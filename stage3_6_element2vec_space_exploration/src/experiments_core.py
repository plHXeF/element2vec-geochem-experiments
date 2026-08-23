from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/element2vec-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .analysis_utils import (
    centroid_by_label,
    continuous_neighbor_smoothness,
    cosine_matrix,
    cv_balanced_accuracy,
    cv_ridge_r2,
    effect_status,
    group_distance_effect,
    knn_indices,
    knn_purity,
    pca_frame,
    permutation_pvalue,
    plot_heatmap,
    plot_scatter,
    row_normalize,
    safe_spearman,
)
from .catalogs import EXPECTED_ELEMENT_PAIRS, numeric_tag_catalog
from .experiment_context import ExperimentContext
from .registry import ExperimentRegistry, ExperimentSpec


def _result(
    *,
    status: str,
    metrics: dict[str, Any],
    primary: str,
    value: float | None,
    baseline: float | None,
    conclusion: str,
    risks: str,
    next_step: str,
    figures: list[Path] | None = None,
    stability: str = "single breadth-first scan",
) -> dict[str, Any]:
    effect = None
    if value is not None and baseline is not None and np.isfinite(value) and np.isfinite(baseline):
        effect = float(value - baseline)
    return {
        "status": status,
        "quantitative_metrics": metrics,
        "figure_paths": [str(p) for p in (figures or [])],
        "conclusion_summary": conclusion,
        "risks_and_confounders": risks,
        "metric_primary": primary,
        "metric_value": value,
        "baseline_value": baseline,
        "effect_size": effect,
        "stability": stability,
        "main_observation": conclusion,
        "possible_confounder": risks,
        "recommended_next_step": next_step,
    }


def _spec(
    experiment_id: str,
    name: str,
    family: str,
    level: str,
    relation: str,
    hypothesis: str,
    ctx: ExperimentContext,
    sample_size: int,
    control: str,
    input_fields: str = "34 trained numeric tags",
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        experiment_name=name,
        family=family,
        representation_level=level,
        dataset="v_high_quality_dataset stratified exploration sample",
        input_fields=input_fields,
        target_geological_relation=relation,
        seed=ctx.seed,
        sample_size=sample_size,
        control_group=control,
        hypothesis=hypothesis,
    )


def experiment_a1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    tags = ctx.numeric_tags
    catalog = numeric_tag_catalog(tags)
    catalog.to_csv(out / "element_property_catalog.csv", index=False, encoding="utf-8-sig")
    trained = ctx.static_trained["numeric"][: len(tags)]
    random = ctx.static_random["numeric"][: len(tags)]
    properties = [
        "atomic_number",
        "atomic_weight",
        "periodic_group",
        "period",
        "electronegativity_pauling",
        "common_valence",
        "typical_ionic_radius_A",
        "ionic_potential",
    ]
    rows = []
    for prop in properties:
        values = catalog[prop].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 8:
            continue
        r2_trained = cv_ridge_r2(trained[valid], values[valid], folds=5, seed=ctx.seed)
        r2_random = cv_ridge_r2(random[valid], values[valid], folds=5, seed=ctx.seed)
        prop_diff = pdist(values[valid, None], metric="cityblock")
        rho_trained, p_trained = safe_spearman(pdist(row_normalize(trained[valid]), metric="cosine"), prop_diff)
        rho_random, _ = safe_spearman(pdist(row_normalize(random[valid]), metric="cosine"), prop_diff)
        rows.append(
            {
                "property": prop,
                "n": int(valid.sum()),
                "trained_cv_r2": r2_trained,
                "random_cv_r2": r2_random,
                "delta_cv_r2": r2_trained - r2_random,
                "trained_distance_spearman": rho_trained,
                "trained_distance_p": p_trained,
                "random_distance_spearman": rho_random,
            }
        )
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(out / "property_readout.csv", index=False, encoding="utf-8-sig")
    pca, coords = pca_frame(trained, n_components=3)
    coords = pd.concat([catalog[["numeric_tag", "geochemical_group"]].reset_index(drop=True), coords], axis=1)
    fig_path = out / "figures" / "numeric_embedding_pca.png"
    plot_scatter(coords[["PC1", "PC2"]], coords["geochemical_group"], fig_path, "Static numeric-tag embedding", categorical=True, dpi=ctx.dpi)
    mean_delta = float(result_frame["delta_cv_r2"].mean())
    mean_distance = float(result_frame["trained_distance_spearman"].mean())
    status = effect_status(mean_delta)
    return _result(
        status=status,
        metrics={
            "mean_delta_cv_r2_vs_random": mean_delta,
            "mean_trained_distance_spearman": mean_distance,
            "pca_variance_first3": pca.explained_variance_ratio_.tolist(),
            "properties_tested": len(result_frame),
        },
        primary="mean_delta_cv_r2_vs_random",
        value=mean_delta,
        baseline=0.0,
        conclusion=f"Across {len(result_frame)} approximate chemical properties, trained embeddings changed cross-validated linear readability by {mean_delta:+.3f} R² on average relative to matched random initialization.",
        risks="Only 33 tags map cleanly to a central element; oxide tags duplicate some cations, and common valence/ionic radius are coordination-dependent approximations.",
        next_step="Retest any positive property with a curated, citation-backed mineral-specific property table and leave-one-element-family-out validation.",
        figures=[fig_path],
    )


def experiment_a2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    tags = ctx.numeric_tags
    catalog = numeric_tag_catalog(tags)
    trained = ctx.static_trained["numeric"][: len(tags)]
    random = ctx.static_random["numeric"][: len(tags)]
    rng = np.random.default_rng(ctx.seed)
    group_rows = []
    for column in ["geochemical_group", "goldschmidt_affinity", "compatibility"]:
        labels = catalog[column].astype(str).to_numpy()
        valid = ~np.isin(labels, ["unclassified", "not_applicable"])
        effect_t, within_t, between_t = group_distance_effect(trained[valid], labels[valid])
        effect_r, within_r, between_r = group_distance_effect(random[valid], labels[valid])
        null = []
        for _ in range(int(ctx.config["analysis"]["permutation_repeats"])):
            shuffled = rng.permutation(labels[valid])
            null.append(group_distance_effect(trained[valid], shuffled)[0])
        group_rows.append(
            {
                "grouping": column,
                "trained_effect": effect_t,
                "random_effect": effect_r,
                "delta_effect": effect_t - effect_r,
                "permutation_p": permutation_pvalue(effect_t, np.asarray(null)),
                "trained_within_distance": within_t,
                "trained_between_distance": between_t,
                "random_within_distance": within_r,
                "random_between_distance": between_r,
            }
        )
    groups = pd.DataFrame(group_rows)
    groups.to_csv(out / "geochemical_group_enrichment.csv", index=False, encoding="utf-8-sig")
    sim_t = cosine_matrix(trained)
    sim_r = cosine_matrix(random)
    tag_index = {tag: i for i, tag in enumerate(tags)}
    pair_rows = []
    for left, right, relation in EXPECTED_ELEMENT_PAIRS:
        available = left in tag_index and right in tag_index
        row = {"relation": relation, "left": left, "right": right, "available": available}
        if available:
            i, j = tag_index[left], tag_index[right]
            order_t = np.argsort(-sim_t[i])
            order_r = np.argsort(-sim_r[i])
            rank_t = int(np.flatnonzero(order_t == j)[0])
            rank_r = int(np.flatnonzero(order_r == j)[0])
            row.update(
                trained_cosine=float(sim_t[i, j]),
                random_cosine=float(sim_r[i, j]),
                trained_rank=rank_t,
                random_rank=rank_r,
                trained_reciprocal_rank=1.0 / rank_t if rank_t else 1.0,
                random_reciprocal_rank=1.0 / rank_r if rank_r else 1.0,
            )
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "expected_element_pair_ranks.csv", index=False, encoding="utf-8-sig")
    heatmap_path = out / "figures" / "numeric_embedding_cosine.png"
    plot_heatmap(pd.DataFrame(sim_t, index=tags, columns=tags), heatmap_path, "Trained numeric-tag cosine similarity", dpi=ctx.dpi, cmap="viridis")
    available_pairs = pairs.loc[pairs["available"]]
    mrr_t = float(available_pairs["trained_reciprocal_rank"].mean())
    mrr_r = float(available_pairs["random_reciprocal_rank"].mean())
    group_delta = float(groups["delta_effect"].mean())
    primary = 0.5 * (mrr_t - mrr_r) + 0.5 * group_delta
    status = effect_status(primary)
    return _result(
        status=status,
        metrics={
            "available_expected_pairs": int(available_pairs.shape[0]),
            "unavailable_expected_pairs": int((~pairs["available"]).sum()),
            "trained_expected_pair_mrr": mrr_t,
            "random_expected_pair_mrr": mrr_r,
            "mean_group_effect_delta": group_delta,
            "combined_primary_effect": primary,
        },
        primary="combined_pair_and_group_effect",
        value=primary,
        baseline=0.0,
        conclusion=f"Only {len(available_pairs)}/{len(pairs)} prespecified similarity pairs are testable. The combined trained-over-random pair/group effect is {primary:+.3f}.",
        risks="Geochemical group assignments overlap and are context-dependent; six canonical pairs cannot be tested because one or both tags were never trained.",
        next_step="Prioritize only permutation-stable group effects; extending the vocabulary is required before claiming broad Goldschmidt or substitution structure.",
        figures=[heatmap_path],
    )


def experiment_a3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    ree_tags = [tag for tag in ["La_ppm", "Ce_ppm", "Nd_ppm", "Yb_ppm"] if tag in ctx.numeric_tags]
    idx = [ctx.numeric_tags.index(tag) for tag in ree_tags]
    z = numeric_tag_catalog(ree_tags)["atomic_number"].to_numpy(dtype=float)
    trained = ctx.static_trained["numeric"][idx]
    random = ctx.static_random["numeric"][idx]
    pca_t = PCA(n_components=min(2, len(idx))).fit_transform(StandardScaler().fit_transform(trained))
    pca_r = PCA(n_components=min(2, len(idx))).fit_transform(StandardScaler().fit_transform(random))
    rho_t = abs(safe_spearman(pca_t[:, 0], z)[0])
    rho_r = abs(safe_spearman(pca_r[:, 0], z)[0])
    order = pd.DataFrame({"tag": ree_tags, "atomic_number": z, "trained_pc1": pca_t[:, 0], "random_pc1": pca_r[:, 0]})
    order.to_csv(out / "available_ree_sequence.csv", index=False, encoding="utf-8-sig")
    status = "Weak/ambiguous" if rho_t > rho_r else "Negative"
    return _result(
        status=status,
        metrics={"available_ree_tags": ree_tags, "trained_abs_pc1_atomic_number_rho": rho_t, "random_abs_pc1_atomic_number_rho": rho_r},
        primary="abs_pc1_atomic_number_spearman",
        value=rho_t,
        baseline=rho_r,
        conclusion=f"The trained vocabulary contains only {', '.join(ree_tags)}; their PC1 ordering correlation with atomic number is {rho_t:.3f} versus {rho_r:.3f} at random initialization.",
        risks="Four irregularly spaced REE tags cannot establish an LREE–MREE–HREE sequence or test Ce/Eu anomalies. PCA sign and axis selection are unstable at n=4.",
        next_step="Treat this as a coverage audit, not evidence of a learned REE manifold; retraining with a complete REE panel is required.",
    )


def _trajectory_metrics(points: np.ndarray, grid: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = float(steps.sum())
    endpoint = float(np.linalg.norm(points[-1] - points[0]))
    coords = PCA(n_components=min(2, points.shape[0], points.shape[1])).fit_transform(points)
    monotonic = abs(safe_spearman(coords[:, 0], grid)[0])
    mid = len(steps) // 2
    center = float(np.mean(steps[max(0, mid - 2) : min(len(steps), mid + 2)]))
    outer = float(np.mean(np.r_[steps[:3], steps[-3:]]))
    return {
        "arc_length": arc,
        "endpoint_distance": endpoint,
        "straightness": endpoint / arc if arc > 0 else np.nan,
        "pc1_monotonicity": monotonic,
        "step_cv": float(steps.std() / steps.mean()) if steps.mean() > 0 else np.nan,
        "outer_to_center_step_ratio": outer / center if center > 0 else np.nan,
    }


def experiment_b1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    grid = np.linspace(-3.0, 3.0, 25, dtype=np.float32)
    trained = ctx.project_numeric_grid(grid, trained=True)
    random = ctx.project_numeric_grid(grid, trained=False)
    rows = []
    for i, tag in enumerate(ctx.numeric_tags):
        row = {"field": tag}
        row.update({f"trained_{k}": v for k, v in _trajectory_metrics(trained[i], grid).items()})
        row.update({f"random_{k}": v for k, v in _trajectory_metrics(random[i], grid).items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "numeric_token_trajectory_metrics.csv", index=False, encoding="utf-8-sig")
    representatives = [tag for tag in ["SiO2", "MgO", "K2O", "Ni_ppm", "Rb_ppm", "Zr_ppm", "La_ppm", "Yb_ppm"] if tag in ctx.numeric_tags]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, tag in zip(axes.ravel(), representatives):
        i = ctx.numeric_tags.index(tag)
        coords = PCA(n_components=2).fit_transform(trained[i])
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=grid, cmap="coolwarm", s=18)
        ax.plot(coords[:, 0], coords[:, 1], linewidth=0.7, alpha=0.7)
        ax.set_title(tag)
    for ax in axes.ravel()[len(representatives) :]:
        ax.axis("off")
    fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.7, label="robust-z input")
    fig.suptitle("MLP token trajectories across a common normalized-value grid")
    fig_path = out / "figures" / "representative_numeric_trajectories.png"
    fig.savefig(fig_path, dpi=ctx.dpi, bbox_inches="tight")
    plt.close(fig)
    monotonic_t = float(frame["trained_pc1_monotonicity"].median())
    monotonic_r = float(frame["random_pc1_monotonicity"].median())
    straight_t = float(frame["trained_straightness"].median())
    saturation = float(frame["trained_outer_to_center_step_ratio"].median())
    status = "Strong positive" if monotonic_t >= 0.9 else effect_status(monotonic_t - monotonic_r)
    return _result(
        status=status,
        metrics={
            "median_trained_pc1_monotonicity": monotonic_t,
            "median_random_pc1_monotonicity": monotonic_r,
            "median_trained_straightness": straight_t,
            "median_outer_to_center_step_ratio": saturation,
            "fields": len(frame),
        },
        primary="median_pc1_monotonicity",
        value=monotonic_t,
        baseline=monotonic_r,
        conclusion=f"Across all {len(frame)} numeric tags, the median trained MLP trajectory has PC1 monotonicity {monotonic_t:.3f}, straightness {straight_t:.3f}, and outer/central sensitivity ratio {saturation:.3f}.",
        risks="The grid is in per-field robust-z space, so equal steps do not mean equal absolute or log-fold concentration changes. This tests encoder geometry, not natural-sample frequency.",
        next_step="Investigate fields with low monotonicity, low straightness, or strong edge compression against empirical quantiles and detection-limit coverage.",
        figures=[fig_path],
    )


def experiment_b2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    pairs = [("K2O", "Rb_ppm"), ("MgO", "Ni_ppm"), ("La_ppm", "Yb_ppm"), ("CaO", "Sr_ppm")]
    grid = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    trained = ctx.project_numeric_grid(grid, trained=True)
    random = ctx.project_numeric_grid(grid, trained=False)
    rows = []
    for left, right in pairs:
        if left not in ctx.numeric_tags or right not in ctx.numeric_tags:
            continue
        i, j = ctx.numeric_tags.index(left), ctx.numeric_tags.index(right)
        for name, arr in [("trained", trained), ("random", random)]:
            base = 0.5 * (arr[i, 1] + arr[j, 1])
            common = 0.5 * (arr[i, 2] + arr[j, 2])
            ratio = 0.5 * (arr[i, 2] + arr[j, 0])
            common_delta = float(np.linalg.norm(common - base))
            ratio_delta = float(np.linalg.norm(ratio - base))
            rows.append(
                {
                    "pair": f"{left}|{right}",
                    "model": name,
                    "common_shift_norm": common_delta,
                    "ratio_shift_norm": ratio_delta,
                    "ratio_to_common": ratio_delta / common_delta if common_delta else np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "paired_absolute_relative_perturbations.csv", index=False, encoding="utf-8-sig")
    trained_ratio = float(frame.loc[frame["model"] == "trained", "ratio_to_common"].mean())
    random_ratio = float(frame.loc[frame["model"] == "random", "ratio_to_common"].mean())
    return _result(
        status=effect_status(trained_ratio - random_ratio),
        metrics={"mean_trained_ratio_to_common_displacement": trained_ratio, "mean_random_ratio_to_common_displacement": random_ratio},
        primary="ratio_to_common_displacement",
        value=trained_ratio,
        baseline=random_ratio,
        conclusion=f"In the pre-Transformer two-token mean, ratio-changing perturbations move {trained_ratio:.3f} times as far as common positive shifts, compared with {random_ratio:.3f} for the matched random encoder.",
        risks="A token-wise MLP has no access to other elements or closure; this is a geometric superposition test, not evidence that the model explicitly represents geochemical ratios.",
        next_step="Repeat the same perturbations through the Transformer on closed synthetic compositions (F3) before assigning relative-composition semantics.",
    )


def experiment_b3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    grid = np.asarray([-0.5, 0.5], dtype=np.float32)
    trained = ctx.project_numeric_grid(grid, trained=True)
    random = ctx.project_numeric_grid(grid, trained=False)
    dir_t = trained[:, 1] - trained[:, 0]
    dir_r = random[:, 1] - random[:, 0]
    norm_t = np.linalg.norm(dir_t, axis=1)
    norm_r = np.linalg.norm(dir_r, axis=1)
    sim_t = cosine_matrix(dir_t)
    sim_r = cosine_matrix(dir_r)
    tri = np.triu_indices(len(dir_t), k=1)
    metrics = pd.DataFrame(
        {
            "field": ctx.numeric_tags,
            "trained_unit_z_displacement": norm_t,
            "random_unit_z_displacement": norm_r,
        }
    )
    metrics.to_csv(out / "numeric_scale_consistency.csv", index=False, encoding="utf-8-sig")
    cohesion_t = float(np.mean(sim_t[tri]))
    cohesion_r = float(np.mean(sim_r[tri]))
    cv_t = float(norm_t.std() / norm_t.mean())
    cv_r = float(norm_r.std() / norm_r.mean())
    effect = (cohesion_t - cohesion_r) + (cv_r - cv_t)
    return _result(
        status=effect_status(effect),
        metrics={
            "trained_direction_mean_cosine": cohesion_t,
            "random_direction_mean_cosine": cohesion_r,
            "trained_displacement_cv": cv_t,
            "random_displacement_cv": cv_r,
            "combined_consistency_gain": effect,
        },
        primary="combined_direction_and_scale_consistency_gain",
        value=effect,
        baseline=0.0,
        conclusion=f"Trained field value directions have mean cross-field cosine {cohesion_t:.3f} and norm CV {cv_t:.3f}; random values are {cohesion_r:.3f} and {cv_r:.3f}.",
        risks="A shared MLP mechanically encourages some common direction even before training. Per-field robust scaling also makes equal z steps distributional rather than physically equivalent.",
        next_step="Compare equal empirical quantiles and equal log-fold changes separately for major and trace fields.",
    )


def _joined_token_frame(ctx: ExperimentContext) -> pd.DataFrame:
    keep = [
        "sample_id", "sio2", "mg_number", "mali", "cia", "total_alkali",
        "log_sr_y", "num_numeric_tokens", "missing_fraction", "rock_group",
        "rock_origin", "rock_type", "data_source", "canonical_ref_id",
    ]
    source = ctx.meta[keep].copy()
    token = ctx.token_meta.drop(columns=[c for c in ["rock_group", "rock_origin", "rock_type", "data_source", "canonical_ref_id"] if c in ctx.token_meta])
    return token.merge(source, on="sample_id", how="left", validate="many_to_one")


def _pca_probe(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    n = min(16, x.shape[0] - 1, x.shape[1])
    coords = PCA(n_components=n, random_state=seed).fit_transform(StandardScaler().fit_transform(x))
    return cv_balanced_accuracy(coords, labels, folds=5, seed=seed)


def experiment_c1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    meta = _joined_token_frame(ctx)
    arrays = ctx.token_arrays()
    representatives = ["Ni_ppm", "Sr_ppm", "Zr_ppm", "Rb_ppm", "K2O", "CaO"]
    rows = []
    for field in representatives:
        mask = meta["field"].eq(field).to_numpy() & meta["normalized_value"].abs().le(0.30).to_numpy()
        if mask.sum() < 120:
            continue
        sio2 = meta.loc[mask, "sio2"].to_numpy(dtype=float)
        background = np.where(sio2 < 52, "mafic", np.where(sio2 > 63, "felsic", "intermediate"))
        projected_score = _pca_probe(arrays["projected"][mask], background, ctx.seed)
        contextual_score = _pca_probe(arrays["contextual"][mask], background, ctx.seed)
        rows.append(
            {
                "field": field,
                "n": int(mask.sum()),
                "projected_background_balanced_accuracy": projected_score,
                "contextual_background_balanced_accuracy": contextual_score,
                "contextual_gain": contextual_score - projected_score,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "same_value_different_background.csv", index=False, encoding="utf-8-sig")
    gain = float(frame["contextual_gain"].mean()) if len(frame) else np.nan
    ctx_score = float(frame["contextual_background_balanced_accuracy"].mean()) if len(frame) else np.nan
    proj_score = float(frame["projected_background_balanced_accuracy"].mean()) if len(frame) else np.nan
    return _result(
        status=effect_status(gain),
        metrics={"fields_tested": len(frame), "mean_projected_score": proj_score, "mean_contextual_score": ctx_score, "mean_contextual_gain": gain},
        primary="mean_contextual_background_probe_gain",
        value=ctx_score,
        baseline=proj_score,
        conclusion=f"At nearly fixed element value, contextual tokens improve mafic/intermediate/felsic background readability by {gain:+.3f} balanced accuracy on average across {len(frame)} fields.",
        risks="SiO₂ background is a coarse proxy, and contextual tokens can exploit reporting patterns or correlated fields rather than mineralogical role.",
        next_step="For fields with the largest gain, match samples additionally on token coverage and publication before interpreting context as geological role.",
    )


def experiment_c2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    meta = _joined_token_frame(ctx)
    arrays = ctx.token_arrays()
    definitions = [
        ("Sr_ppm", "log_sr_y"),
        ("Ni_ppm", "mg_number"),
        ("Cr_ppm", "mg_number"),
        ("Rb_ppm", "sio2"),
        ("Zr_ppm", "sio2"),
        ("K2O", "mali"),
        ("CaO", "total_alkali"),
    ]
    rows = []
    for field, proxy in definitions:
        field_mask = meta["field"].eq(field) & meta[proxy].notna()
        subset = meta.loc[field_mask]
        if len(subset) < 150:
            continue
        bins = pd.qcut(subset[proxy], q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str).to_numpy()
        mask = field_mask.to_numpy()
        proj = _pca_probe(arrays["projected"][mask], bins, ctx.seed)
        contextual = _pca_probe(arrays["contextual"][mask], bins, ctx.seed)
        rows.append({"field": field, "process_proxy": proxy, "n": int(mask.sum()), "projected_score": proj, "contextual_score": contextual, "gain": contextual - proj})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "contextual_role_proxies.csv", index=False, encoding="utf-8-sig")
    gain = float(frame["gain"].mean()) if len(frame) else np.nan
    return _result(
        status=effect_status(gain),
        metrics={"role_proxies": len(frame), "mean_contextual_gain": gain, "largest_gain_field": str(frame.loc[frame["gain"].idxmax(), "field"]) if len(frame) else None},
        primary="mean_contextual_role_proxy_gain",
        value=gain,
        baseline=0.0,
        conclusion=f"Across {len(frame)} predeclared element/process proxies, contextualization changes low/mid/high proxy readability by {gain:+.3f} balanced accuracy on average.",
        risks="Mg#, Sr/Y, MALI, and SiO₂ are process proxies, not unique process labels; high readability can reflect direct compositional covariance.",
        next_step="Only follow fields whose contextual gain remains after value-bin, publication, and missingness matching.",
    )


def experiment_c3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    meta = _joined_token_frame(ctx)
    arrays = ctx.token_arrays()
    rows = []
    for field in ctx.numeric_tags:
        mask = meta["field"].eq(field).to_numpy()
        if mask.sum() < 100:
            continue
        m = meta.loc[mask].reset_index(drop=True)
        for level in ["projected", "contextual"]:
            x = arrays[level][mask]
            idx = knn_indices(x, k=15)
            row = {
                "field": field,
                "level": level,
                "n": len(m),
                "value_neighbor_absdiff": float(np.nanmean(np.abs(m["normalized_value"].to_numpy()[idx] - m["normalized_value"].to_numpy()[:, None]))),
                "rock_group_purity": float(np.mean(m["rock_group"].fillna("[MISSING]").astype(str).to_numpy()[idx] == m["rock_group"].fillna("[MISSING]").astype(str).to_numpy()[:, None])),
                "rock_type_purity": float(np.mean(m["rock_type"].fillna("[MISSING]").astype(str).to_numpy()[idx] == m["rock_type"].fillna("[MISSING]").astype(str).to_numpy()[:, None])),
                "source_purity": float(np.mean(m["data_source"].fillna("[MISSING]").astype(str).to_numpy()[idx] == m["data_source"].fillna("[MISSING]").astype(str).to_numpy()[:, None])),
                "reference_purity": float(np.mean(m["canonical_ref_id"].fillna(-1).astype(str).to_numpy()[idx] == m["canonical_ref_id"].fillna(-1).astype(str).to_numpy()[:, None])),
                "coverage_neighbor_absdiff": float(np.mean(np.abs(m["num_numeric_tokens"].to_numpy()[idx] - m["num_numeric_tokens"].to_numpy()[:, None]))),
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "contextual_neighbor_controls.csv", index=False, encoding="utf-8-sig")
    pivot = frame.pivot(index="field", columns="level", values=["rock_group_purity", "rock_type_purity", "source_purity", "reference_purity", "value_neighbor_absdiff"])
    gains = {}
    for metric in ["rock_group_purity", "rock_type_purity", "source_purity", "reference_purity"]:
        gains[metric] = float((pivot[(metric, "contextual")] - pivot[(metric, "projected")]).mean())
    geology_gain = np.mean([gains["rock_group_purity"], gains["rock_type_purity"]])
    source_gain = np.mean([gains["source_purity"], gains["reference_purity"]])
    net = float(geology_gain - source_gain)
    status = effect_status(net)
    if source_gain > geology_gain and source_gain > 0.05:
        status = "Invalid due to leakage/confounding"
    return _result(
        status=status,
        metrics={"mean_contextual_gains": gains, "geology_gain": geology_gain, "source_gain": source_gain, "geology_minus_source_gain": net},
        primary="geology_minus_source_neighbor_gain",
        value=float(geology_gain),
        baseline=float(source_gain),
        conclusion=f"Contextualization increases geology-label neighbor purity by {geology_gain:+.3f} and source/reference purity by {source_gain:+.3f}; net geology-over-source gain is {net:+.3f}.",
        risks="Token reservoir is capped uniformly by field and the model was trained with hierarchical rock labels; neighbor purity is descriptive, not independent generalization.",
        next_step="Inspect per-field cases where source gain dominates and repeat with publication-grouped matching.",
    )


def _space_subset(ctx: ExperimentContext) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    max_n = min(int(ctx.config["analysis"].get("max_pairwise_samples", 3000)), len(ctx.meta))
    rng = np.random.default_rng(ctx.seed)
    indices = np.sort(rng.choice(len(ctx.meta), size=max_n, replace=False))
    latents = ctx.latents("numeric_only")
    spaces = {
        "raw_log_robust_plus_mask": ctx.raw_baseline()[indices],
        "projected_mean_numeric": latents["projected_mean_numeric"][indices],
        "encoded_mean_numeric": latents["encoded_mean_numeric"][indices],
        "cls": latents["cls"][indices],
    }
    return indices, spaces


def experiment_d1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    indices, spaces = _space_subset(ctx)
    meta = ctx.meta.iloc[indices].reset_index(drop=True)
    meta[["sample_id", "rock_group", "rock_origin", "rock_type"]].to_csv(out / "samples.csv", index=False, encoding="utf-8-sig")
    rows = []
    for name, x in spaces.items():
        rows.append(
            {
                "space": name,
                "rock_group_knn15": knn_purity(x, meta["rock_group"].fillna("[MISSING]").astype(str).to_numpy(), 15),
                "rock_origin_knn15": knn_purity(x, meta["rock_origin"].fillna("[MISSING]").astype(str).to_numpy(), 15),
                "rock_type_knn15": knn_purity(x, meta["rock_type"].fillna("[MISSING]").astype(str).to_numpy(), 15),
                "sio2_smoothness": continuous_neighbor_smoothness(x, meta["sio2"].to_numpy(), 15),
                "mg_number_smoothness": continuous_neighbor_smoothness(x, meta["mg_number"].to_numpy(), 15),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "rock_structure_metrics.csv", index=False, encoding="utf-8-sig")
    encoded = spaces["encoded_mean_numeric"]
    pca, coords = pca_frame(encoded, n_components=3)
    fig1 = out / "figures" / "encoded_pca_rock_group.png"
    fig2 = out / "figures" / "encoded_pca_sio2.png"
    plot_scatter(coords, meta["rock_group"], fig1, "Encoded mean latent by rock group", categorical=True, dpi=ctx.dpi)
    plot_scatter(coords, meta["sio2"], fig2, "Encoded mean latent by SiO2", categorical=False, dpi=ctx.dpi)
    raw_score = float(frame.loc[frame["space"] == "raw_log_robust_plus_mask", "rock_type_knn15"].iloc[0])
    encoded_score = float(frame.loc[frame["space"] == "encoded_mean_numeric", "rock_type_knn15"].iloc[0])
    gain = encoded_score - raw_score
    return _result(
        status=effect_status(gain),
        metrics={"sample_size": len(meta), "raw_rock_type_knn15": raw_score, "encoded_rock_type_knn15": encoded_score, "encoded_minus_raw": gain, "encoded_pca_first3_variance": pca.explained_variance_ratio_.tolist()},
        primary="rock_type_knn15",
        value=encoded_score,
        baseline=raw_score,
        conclusion=f"Chemistry-only encoded-mean latent has rock-type kNN15 purity {encoded_score:.3f} versus {raw_score:.3f} for transformed values plus observation mask.",
        risks="The checkpoint used rock labels during full-data training, so rock-label organization is decodability, not out-of-sample discovery. Missingness is present in both spaces.",
        next_step="Interpret only gradients or neighborhoods that improve over raw chemistry and remain stable under publication/source controls.",
        figures=[fig1, fig2],
    )


def experiment_d2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    indices, spaces = _space_subset(ctx)
    meta = ctx.meta.iloc[indices].reset_index(drop=True)
    rows = []
    for name, x in spaces.items():
        idx = knn_indices(x, 15)
        for field, kind in [
            ("rock_type", "categorical"),
            ("rock_group", "categorical"),
            ("data_source", "categorical"),
            ("canonical_ref_id", "categorical"),
            ("sio2", "continuous"),
            ("mg_number", "continuous"),
            ("total_alkali", "continuous"),
            ("num_numeric_tokens", "continuous"),
        ]:
            values = meta[field]
            if kind == "categorical":
                arr = values.fillna("[MISSING]").astype(str).to_numpy()
                metric = float(np.mean(arr[idx] == arr[:, None]))
                metric_name = "purity"
            else:
                arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                metric = float(np.nanmedian(np.abs(arr[idx] - arr[:, None])))
                metric_name = "median_neighbor_absdiff"
            rows.append({"space": name, "attribute": field, "metric": metric_name, "value": metric})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "neighborhood_attribute_metrics.csv", index=False, encoding="utf-8-sig")
    anchors = meta.groupby(meta["rock_type"].fillna("[MISSING]")).head(1).head(20).index.to_numpy()
    encoded_idx = knn_indices(spaces["encoded_mean_numeric"], 5)
    cases = []
    for i in anchors:
        for rank, j in enumerate(encoded_idx[i], start=1):
            cases.append(
                {
                    "anchor_sample_id": int(meta.iloc[i]["sample_id"]),
                    "anchor_type": meta.iloc[i]["rock_type"],
                    "neighbor_rank": rank,
                    "neighbor_sample_id": int(meta.iloc[j]["sample_id"]),
                    "neighbor_type": meta.iloc[j]["rock_type"],
                    "same_reference": str(meta.iloc[i]["canonical_ref_id"]) == str(meta.iloc[j]["canonical_ref_id"]),
                    "sio2_difference": abs(float(meta.iloc[i]["sio2"]) - float(meta.iloc[j]["sio2"])),
                }
            )
    pd.DataFrame(cases).to_csv(out / "representative_neighbor_cases.csv", index=False, encoding="utf-8-sig")
    pivot = frame.pivot(index="attribute", columns="space", values="value")
    encoded_geo = float(np.mean([pivot.loc["rock_type", "encoded_mean_numeric"], pivot.loc["rock_group", "encoded_mean_numeric"]]))
    encoded_source = float(np.mean([pivot.loc["data_source", "encoded_mean_numeric"], pivot.loc["canonical_ref_id", "encoded_mean_numeric"]]))
    raw_geo = float(np.mean([pivot.loc["rock_type", "raw_log_robust_plus_mask"], pivot.loc["rock_group", "raw_log_robust_plus_mask"]]))
    net_gain = (encoded_geo - raw_geo) - encoded_source
    status = effect_status(encoded_geo - raw_geo)
    if encoded_source > encoded_geo:
        status = "Invalid due to leakage/confounding"
    return _result(
        status=status,
        metrics={"encoded_geology_purity_mean": encoded_geo, "raw_geology_purity_mean": raw_geo, "encoded_source_reference_purity_mean": encoded_source, "net_gain_after_source_purity": net_gain},
        primary="geology_neighbor_purity",
        value=encoded_geo,
        baseline=raw_geo,
        conclusion=f"Encoded-mean geology purity is {encoded_geo:.3f} versus raw {raw_geo:.3f}; encoded source/reference purity is {encoded_source:.3f} and is reported separately as a confounder.",
        risks="Source purity and geology purity are not on identical class-cardinality baselines; publication IDs also proxy real suites and localities.",
        next_step="Use explicit publication-grouped retrieval and spatial distance bins for any neighborhood selected as a candidate result.",
    )


def experiment_d3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    n = min(6000, len(ctx.meta))
    indices = np.sort(rng.choice(len(ctx.meta), size=n, replace=False))
    meta = ctx.meta.iloc[indices].reset_index(drop=True)
    latent = ctx.latents("numeric_only")
    spaces = {
        "raw": ctx.raw_baseline()[indices],
        "projected": latent["projected_mean_numeric"][indices],
        "encoded": latent["encoded_mean_numeric"][indices],
        "cls": latent["cls"][indices],
    }
    targets = ["sio2", "mg_number", "total_alkali", "asi", "mali", "cia", "wip", "maficity", "qtzindex", "log_sr_y", "log_la_yb", "age", "num_numeric_tokens"]
    rows = []
    figures = []
    for name, x in spaces.items():
        n_comp = min(12, x.shape[1])
        coords = PCA(n_components=n_comp, random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(x))
        for pc in range(n_comp):
            for target in targets:
                rho, p = safe_spearman(coords[:, pc], meta[target].to_numpy(dtype=float))
                rows.append({"space": name, "axis": f"PC{pc+1}", "target": target, "spearman": rho, "pvalue": p})
        if name == "encoded":
            corr = pd.DataFrame(rows).query("space == 'encoded'").pivot(index="axis", columns="target", values="spearman")
            fig_path = out / "figures" / "encoded_pca_geochemical_correlations.png"
            plot_heatmap(corr, fig_path, "Encoded latent PCA-axis geochemical correlations", dpi=ctx.dpi)
            figures.append(fig_path)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "global_axis_correlations.csv", index=False, encoding="utf-8-sig")
    best = frame.assign(abs_rho=frame["spearman"].abs()).groupby(["space", "target"])["abs_rho"].max().reset_index()
    encoded_mean = float(best.loc[best["space"] == "encoded", "abs_rho"].mean())
    raw_mean = float(best.loc[best["space"] == "raw", "abs_rho"].mean())
    return _result(
        status=effect_status(encoded_mean - raw_mean),
        metrics={"mean_best_axis_abs_rho_encoded": encoded_mean, "mean_best_axis_abs_rho_raw": raw_mean, "targets": targets, "sample_size": n, "nonlinear_axis_scan": "split into D3b and recorded as technical failure after native exit 139"},
        primary="mean_best_axis_abs_spearman",
        value=encoded_mean,
        baseline=raw_mean,
        conclusion=f"For the predeclared geochemical gradients, the mean strongest single PCA-axis |ρ| is {encoded_mean:.3f} in encoded latent versus {raw_mean:.3f} in the raw transformed baseline.",
        risks="PCA axes are rotation-dependent and several derived indices are deterministic functions of model inputs. The attempted spectral nonlinear scan crashed in native code and is recorded separately rather than silently removed.",
        next_step="Promote only gradients that are smooth in high-dimensional neighborhoods and repeat across seeds/sample subsets before fitting a geological principal curve.",
        figures=figures,
    )


def experiment_d3b_native_failure(_ctx: ExperimentContext, _out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    return _result(
        status="Failed technically",
        metrics={"exit_code": 139, "method": "sklearn SpectralEmbedding", "sample_size": 2000},
        primary="technical_success",
        value=0.0,
        baseline=None,
        conclusion="The low-cost spectral/diffusion proxy terminated the Python process with native exit code 139 after the linear D3 tables had been written.",
        risks="Likely BLAS/OpenMP/native eigensolver interaction in the current macOS CPU environment; no geological conclusion can be drawn from this failure.",
        next_step="Retry the nonlinear graph analysis in an isolated process with one BLAS thread, or use a validated diffusion-map environment.",
        stability="reproduced once; not rerun to avoid another process-level crash",
    )


def run_core_experiments(ctx: ExperimentContext, registry: ExperimentRegistry, only: set[str] | None = None) -> None:
    definitions = [
        (_spec("A1_periodic_properties", "Periodic and chemical property readability", "A", "static numeric-tag embedding", "periodic/chemical properties", "Training should make elementary periodic and ionic properties more readable than matched random initialization.", ctx, len(ctx.numeric_tags), "same-seed random initialization"), experiment_a1),
        (_spec("A2_geochemical_affinity", "Geochemical affinity and expected element neighbors", "A", "static numeric-tag embedding", "Goldschmidt, compatibility, and element-pair structure", "Trained embeddings should enrich prespecified geochemical groups and canonical similar-element pairs.", ctx, len(ctx.numeric_tags), "random embeddings and label permutations"), experiment_a2),
        (_spec("A3_ree_sequence", "Available REE sequence structure", "A", "static numeric-tag embedding", "REE atomic-number sequence", "Available REE tags should form a continuous atomic-number sequence if the vocabulary is sufficient.", ctx, 4, "same-seed random initialization", "La, Ce, Nd, Yb only"), experiment_a3),
        (_spec("B1_numeric_trajectories", "Single-field numeric MLP trajectories", "B", "MLP-projected numeric token", "continuity, monotonicity, saturation", "Each field should trace a continuous, mostly monotonic path as normalized value changes.", ctx, len(ctx.numeric_tags) * 25, "same-seed random MLP and linear input", "all numeric tags on z=-3..3 grid"), experiment_b1),
        (_spec("B2_absolute_relative", "Absolute versus relative two-token changes", "B", "mean of MLP-projected tokens", "common abundance versus ratio change", "Paired ratio changes and common changes should have distinguishable geometry after training.", ctx, 4 * 3, "same-seed random MLP"), experiment_b2),
        (_spec("B3_numeric_scale_consistency", "Cross-field numeric encoding consistency", "B", "MLP-projected numeric token", "shared numeric scale", "Equal normalized changes should induce more coherent directions and scales after training.", ctx, len(ctx.numeric_tags), "same-seed random MLP"), experiment_b3),
        (_spec("C1_same_value_context", "Same value in different rock backgrounds", "C", "projected versus contextual token", "mafic/intermediate/felsic context at fixed value", "Contextual tokens should distinguish rock background beyond the value-only projected token.", ctx, len(ctx.token_meta), "projected token at the same normalized-value window"), experiment_c1),
        (_spec("C2_element_role_proxies", "Contextual element-role proxy branches", "C", "projected versus contextual token", "process-proxy tertiles", "Contextualization should add information about predeclared element-role proxies beyond token value.", ctx, len(ctx.token_meta), "projected token"), experiment_c2),
        (_spec("C3_contextual_neighbors", "Contextual token neighborhood controls", "C", "projected versus contextual token", "value, geology, source, and missingness control", "Geological-neighborhood gains should exceed source/reference gains.", ctx, len(ctx.token_meta), "projected-token neighborhoods"), experiment_c3),
        (_spec("D1_rock_continuity", "Rock classes and continuous transitions", "D", "sample pooled latent", "rock hierarchy and compositional gradients", "Chemistry-only latent should organize rock labels and continuous composition at least as well as raw transformed chemistry.", ctx, int(ctx.config["analysis"]["max_pairwise_samples"]), "raw transformed values plus observation mask"), experiment_d1),
        (_spec("D2_local_geological_neighbors", "Local-neighborhood geology versus source", "D", "sample pooled latent", "geological consistency and source leakage", "Latent neighbors should be geologically coherent without being dominated by publication/source identity.", ctx, int(ctx.config["analysis"]["max_pairwise_samples"]), "raw/projected/CLS spaces"), experiment_d2),
        (_spec("D3_global_axes", "Global linear and nonlinear latent directions", "D", "sample pooled latent", "acidity, differentiation, alkalinity, weathering, age", "Major latent directions should align with multiple predeclared geochemical gradients beyond raw-space PCA.", ctx, 6000, "raw transformed PCA and spectral embedding"), experiment_d3),
        (_spec("D3b_nonlinear_axis_failure", "Nonlinear spectral-axis scan", "D", "sample pooled latent", "nonlinear global structure", "A low-cost graph embedding may reveal curved global gradients not captured by PCA.", ctx, 2000, "PCA global-axis scan"), experiment_d3b_native_failure),
    ]
    for spec, function in definitions:
        if only is not None and spec.experiment_id not in only:
            continue
        print(f"[experiment] {spec.experiment_id}")
        registry.run(spec, lambda out, s, f=function: f(ctx, out, s))
