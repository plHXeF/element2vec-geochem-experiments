from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from stage3_5_element2vec_v2.src.data.collate import collate_token_batch
from stage3_5_element2vec_v2.src.training.train_v2 import move_batch_to_device

from .analysis_utils import (
    centroid_by_label,
    cosine_matrix,
    cv_ridge_r2,
    effect_status,
    knn_purity,
    row_normalize,
    safe_spearman,
)
from .catalogs import EVOLUTION_SEQUENCES
from .experiment_context import ExperimentContext
from .extract_cache import _make_item
from .normative import OXIDES, cipw_like_nnls, ideal_mineral_mixtures
from .registry import ExperimentRegistry, ExperimentSpec


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
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        experiment_name=name,
        family=family,
        representation_level=level,
        dataset="V2 stratified chemistry sample and declared synthetic compositions",
        input_fields="34 numeric chemistry tokens; pure labels omitted",
        target_geological_relation=relation,
        seed=ctx.seed,
        sample_size=sample_size,
        control_group=control,
        hypothesis=hypothesis,
    )


def _result(
    status: str,
    metrics: dict[str, Any],
    primary: str,
    value: float | None,
    baseline: float | None,
    conclusion: str,
    risks: str,
    next_step: str,
    stability: str = "single breadth-first scan",
) -> dict[str, Any]:
    effect = None
    if value is not None and baseline is not None and np.isfinite(value) and np.isfinite(baseline):
        effect = float(value - baseline)
    return {
        "status": status,
        "quantitative_metrics": metrics,
        "figure_paths": [],
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


def _numeric_db_cols(ctx: ExperimentContext) -> list[str]:
    return [item["db_column"] for item in ctx.source_config["data"]["numeric_fields"]]


def _median_type(ctx: ExperimentContext, rock_type: str) -> pd.Series:
    subset = ctx.meta.loc[ctx.meta["rock_type"].fillna("").astype(str).str.lower().eq(rock_type.lower())]
    if len(subset) < 10:
        raise ValueError(f"Need >=10 sampled rows for rock_type={rock_type}; found {len(subset)}")
    return subset[_numeric_db_cols(ctx)].median(numeric_only=True)


def _close_major(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    majors = ["sio2", "tio2", "al2o3", "fe2o3", "feo", "mgo", "cao", "na2o", "k2o", "p2o5", "mno"]
    totals = frame[majors].sum(axis=1, skipna=True)
    valid = totals > 0
    frame.loc[valid, majors] = frame.loc[valid, majors].div(totals.loc[valid], axis=0) * 100.0
    return frame


def _path_metrics(latent: np.ndarray, parameter: np.ndarray) -> dict[str, float]:
    steps = np.linalg.norm(np.diff(latent, axis=0), axis=1)
    arc = float(steps.sum())
    endpoint = float(np.linalg.norm(latent[-1] - latent[0]))
    pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(latent))[:, 0]
    return {
        "parameter_monotonicity": abs(safe_spearman(pc1, parameter)[0]),
        "straightness": endpoint / arc if arc else np.nan,
        "step_cv": float(steps.std() / steps.mean()) if steps.mean() else np.nan,
        "arc_length": arc,
    }


def experiment_e1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    pairs = [
        ("subalkalic basalt", "rhyolite"),
        ("subalkalic basalt", "granite"),
        ("wacke", "limestone"),
        ("shale", "limestone"),
    ]
    rows, compositions = [], []
    alphas = np.linspace(0, 1, 21)
    sample_id = -1
    for left, right in pairs:
        a, b = _median_type(ctx, left), _median_type(ctx, right)
        for closure in ["linear", "closed_major"]:
            path = pd.DataFrame([(1 - alpha) * a + alpha * b for alpha in alphas])
            if closure == "closed_major":
                path = _close_major(path)
            path["sample_id"] = np.arange(sample_id - len(path) + 1, sample_id + 1)
            sample_id -= len(path)
            path["mix_pair"] = f"{left}|{right}"
            path["closure"] = closure
            path["alpha"] = alphas
            compositions.append(path)
    composition = pd.concat(compositions, ignore_index=True)
    latent = ctx.infer_raw_frame(composition)["encoded_mean_numeric"]
    composition.to_csv(out / "synthetic_mixing_compositions.csv", index=False, encoding="utf-8-sig")
    start = 0
    for (pair, closure), group in composition.groupby(["mix_pair", "closure"], sort=False):
        n = len(group)
        metrics = _path_metrics(latent[start : start + n], group["alpha"].to_numpy())
        rows.append({"pair": pair, "closure": closure, **metrics})
        start += n
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "two_endmember_path_metrics.csv", index=False, encoding="utf-8-sig")
    monotonicity = float(frame["parameter_monotonicity"].mean())
    straightness = float(frame["straightness"].mean())
    closure_delta = float(
        frame.pivot(index="pair", columns="closure", values="straightness").eval("closed_major - linear").mean()
    )
    return _result(
        "Strong positive" if monotonicity >= 0.9 else effect_status(monotonicity - 0.5),
        {"paths": len(frame), "mean_parameter_monotonicity": monotonicity, "mean_straightness": straightness, "closure_straightness_delta": closure_delta},
        "mean_mixing_parameter_monotonicity",
        monotonicity,
        0.5,
        f"Eight linear/closed two-endmember paths have mean latent ordering {monotonicity:.3f} and straightness {straightness:.3f}; closure changes straightness by {closure_delta:+.3f} on average.",
        "Median endmembers combine unrelated publications and are not physical coexisting endmembers. Linear oxide mixing and major closure omit density, volatile, and mineralogical constraints.",
        "Promote only curved or branch-specific behavior that is reproducible with natural suites or physically constrained endmembers.",
    )


def experiment_e2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    names = ["subalkalic basalt", "rhyolite", "limestone"]
    endmembers = [_median_type(ctx, name) for name in names]
    rows = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            weights = np.asarray([i, j, k], dtype=float) / 10.0
            comp = sum(w * end for w, end in zip(weights, endmembers))
            row = comp.to_dict()
            row.update(w_basalt=weights[0], w_rhyolite=weights[1], w_limestone=weights[2])
            rows.append(row)
    frame = _close_major(pd.DataFrame(rows))
    frame["sample_id"] = np.arange(-20000 - len(frame), -20000)
    latent = ctx.infer_raw_frame(frame)["encoded_mean_numeric"]
    weights = frame[["w_basalt", "w_rhyolite", "w_limestone"]].to_numpy()
    input_dist = pdist(weights, metric="euclidean")
    latent_dist = pdist(row_normalize(latent), metric="cosine")
    rho, p = safe_spearman(input_dist, latent_dist)
    frame.to_csv(out / "three_endmember_simplex.csv", index=False, encoding="utf-8-sig")
    return _result(
        effect_status(rho, p),
        {"simplex_points": len(frame), "barycentric_vs_latent_distance_spearman": rho, "pvalue": p},
        "simplex_distance_rank_preservation",
        rho,
        0.0,
        f"The closed basalt–rhyolite–limestone simplex preserves barycentric distance ranks at Spearman ρ={rho:.3f} in chemistry-only latent space.",
        "Distance-rank preservation does not prove triangular topology; the three median endmembers are synthetic and may pass through compositions absent from nature.",
        "If retained, visualize edge/interior topology and compare with raw log-ratio space using the same points.",
    )


def experiment_e3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"]
    rows, directions = [], []
    for sequence_id, sequence in enumerate(EVOLUTION_SEQUENCES, start=1):
        centroids = []
        available = []
        for label in sequence:
            mask = ctx.meta["rock_type"].eq(label).to_numpy()
            if mask.sum() >= 20:
                centroids.append(latent[mask].mean(axis=0))
                available.append(label)
        if len(centroids) >= 4:
            centroids = np.stack(centroids)
            metrics = _path_metrics(centroids, np.arange(len(centroids)))
            directions.append(centroids[-1] - centroids[0])
            rows.append({"sequence": sequence_id, "labels": ";".join(available), **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "natural_magmatic_sequence_metrics.csv", index=False, encoding="utf-8-sig")
    direction_cosine = float(cosine_matrix(np.stack(directions))[0, 1]) if len(directions) >= 2 else np.nan
    monotonicity = float(frame["parameter_monotonicity"].mean())
    return _result(
        effect_status(monotonicity - 0.5),
        {"sequences": len(frame), "mean_sequence_monotonicity": monotonicity, "volcanic_plutonic_endpoint_direction_cosine": direction_cosine},
        "natural_sequence_monotonicity",
        monotonicity,
        0.5,
        f"Natural chemistry-only class centroids show mean primitive→evolved ordering {monotonicity:.3f}; volcanic/plutonic endpoint directions have cosine {direction_cosine:.3f}.",
        "Rock_type is chemistry-derived, class centroids mix unrelated suites, and label supervision was used in checkpoint training.",
        "Repeat within publication/locality-defined suites before treating the direction as magmatic fractionation.",
    )


def experiment_e4(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    base = _median_type(ctx, "picrite")
    fields = {item["name"]: item["db_column"] for item in ctx.source_config["data"]["numeric_fields"]}
    d_base = {
        "Rb_ppm": 0.01, "Sr_ppm": 0.2, "Ba_ppm": 0.01, "Zr_ppm": 0.1, "Nb_ppm": 0.01,
        "Y_ppm": 0.2, "V_ppm": 1.0, "Cr_ppm": 3.0, "Ni_ppm": 5.0, "Cu_ppm": 0.5,
        "Zn_ppm": 1.0, "La_ppm": 0.01, "Ce_ppm": 0.02, "Nd_ppm": 0.05, "Th_ppm": 0.001,
        "U_ppm": 0.001, "Pb_ppm": 0.1, "Sc_ppm": 1.0, "Co_ppm": 2.0, "Ga_ppm": 0.5, "Yb_ppm": 0.3,
    }
    incompatible = {"Rb_ppm", "Ba_ppm", "Zr_ppm", "Nb_ppm", "La_ppm", "Ce_ppm", "Nd_ppm", "Th_ppm", "U_ppm", "Pb_ppm"}
    f_values = np.geomspace(0.01, 0.30, 15)
    rows = []
    for source in ["depleted", "enriched"]:
        for residual in ["spinel_proxy", "garnet_proxy"]:
            for melt_fraction in f_values:
                row = base.copy()
                for tag, d_value in d_base.items():
                    col = fields[tag]
                    c0 = float(base[col])
                    if source == "enriched" and tag in incompatible:
                        c0 *= 2.0
                    d = 2.0 if residual == "garnet_proxy" and tag == "Yb_ppm" else d_value
                    row[col] = c0 / (d + melt_fraction * (1.0 - d))
                record = row.to_dict()
                record.update(source=source, residual=residual, melt_fraction=melt_fraction)
                rows.append(record)
    frame = pd.DataFrame(rows)
    frame["sample_id"] = np.arange(-30000 - len(frame), -30000)
    latent = ctx.infer_raw_frame(frame)["encoded_mean_numeric"]
    metrics_rows = []
    start = 0
    for (source, residual), group in frame.groupby(["source", "residual"], sort=False):
        n = len(group)
        metrics_rows.append({"source": source, "residual": residual, **_path_metrics(latent[start : start + n], group["melt_fraction"].to_numpy())})
        start += n
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "partial_melting_path_metrics.csv", index=False, encoding="utf-8-sig")
    frame.to_csv(out / "partial_melting_compositions.csv", index=False, encoding="utf-8-sig")
    monotonicity = float(metrics["parameter_monotonicity"].mean())
    return _result(
        effect_status(monotonicity - 0.5),
        {"branches": len(metrics), "mean_melt_fraction_monotonicity": monotonicity, "mean_straightness": float(metrics["straightness"].mean())},
        "melt_fraction_monotonicity",
        monotonicity,
        0.5,
        f"Four simplified source/residual melting branches have mean F ordering {monotonicity:.3f} in latent space.",
        "Partition coefficients are broad proxies, major elements are held fixed, and the V2 vocabulary lacks a complete REE panel; this is a geometry stress test, not petrological modeling.",
        "Only pursue source/residual branching after replacing proxy coefficients with a documented mineral-mode model.",
    )


def experiment_e5(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    base = _median_type(ctx, "wacke")
    t_values = np.linspace(0, 1, 21)
    rows = []
    for branch in ["leaching", "k_metasomatism"]:
        for t in t_values:
            row = base.copy()
            row["na2o"] *= 1 - 0.75 * t
            row["cao"] *= 1 - 0.70 * t
            row["mgo"] *= 1 - 0.45 * t
            row["sio2"] *= 1 - 0.15 * t
            row["k2o"] *= (1 - 0.25 * t) if branch == "leaching" else (1 + 0.80 * t)
            record = row.to_dict()
            record.update(branch=branch, weathering_intensity=t)
            rows.append(record)
    frame = _close_major(pd.DataFrame(rows))
    frame["sample_id"] = np.arange(-40000 - len(frame), -40000)
    latent = ctx.infer_raw_frame(frame)["encoded_mean_numeric"]
    metrics_rows, start = [], 0
    for branch, group in frame.groupby("branch", sort=False):
        n = len(group)
        metrics_rows.append({"branch": branch, **_path_metrics(latent[start : start + n], group["weathering_intensity"].to_numpy())})
        start += n
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "weathering_path_metrics.csv", index=False, encoding="utf-8-sig")
    monotonicity = float(metrics["parameter_monotonicity"].mean())
    branch_direction = float(cosine_matrix(np.stack([latent[20] - latent[0], latent[41] - latent[21]]))[0, 1])
    return _result(
        effect_status(monotonicity - 0.5),
        {"mean_weathering_monotonicity": monotonicity, "leaching_vs_k_metasomatism_direction_cosine": branch_direction},
        "weathering_intensity_monotonicity",
        monotonicity,
        0.5,
        f"Leaching and K-metasomatism proxy paths have mean ordering {monotonicity:.3f}; endpoint directions have cosine {branch_direction:.3f}.",
        "The perturbation is a compositional proxy, not a mass-balanced weathering model; closure can conflate leaching with relative enrichment.",
        "Compare against natural CIA-matched samples and isocon-style mass-change constraints before geological interpretation.",
    )


def experiment_e6(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    base = _median_type(ctx, "granite")
    t_values = np.linspace(0, 1, 16)
    rows = []
    for style in ["potassic_proxy", "silicification_proxy", "propylitic_proxy"]:
        for t in t_values:
            row = base.copy()
            if style == "potassic_proxy":
                row["k2o"] *= 1 + 1.2 * t
                row["na2o"] *= 1 - 0.5 * t
                row["cao"] *= 1 - 0.4 * t
            elif style == "silicification_proxy":
                row["sio2"] *= 1 + 0.25 * t
            else:
                row["cao"] *= 1 + 0.5 * t
                row["mgo"] *= 1 + 0.4 * t
                row["co2"] = (0.1 if pd.isna(row["co2"]) else row["co2"]) * (1 + 5 * t)
            record = row.to_dict()
            record.update(style=style, intensity=t)
            rows.append(record)
    frame = _close_major(pd.DataFrame(rows))
    frame["sample_id"] = np.arange(-50000 - len(frame), -50000)
    latent = ctx.infer_raw_frame(frame)["encoded_mean_numeric"]
    metrics_rows, directions, start = [], [], 0
    for style, group in frame.groupby("style", sort=False):
        n = len(group)
        path = latent[start : start + n]
        metrics_rows.append({"style": style, **_path_metrics(path, group["intensity"].to_numpy())})
        directions.append(path[-1] - path[0])
        start += n
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "alteration_proxy_paths.csv", index=False, encoding="utf-8-sig")
    direction_sim = cosine_matrix(np.stack(directions))
    separation = float(np.mean(direction_sim[np.triu_indices(len(directions), 1)]))
    monotonicity = float(metrics["parameter_monotonicity"].mean())
    return _result(
        "Weak/ambiguous",
        {"styles": len(metrics), "mean_intensity_monotonicity": monotonicity, "mean_between_style_direction_cosine": separation},
        "alteration_proxy_monotonicity",
        monotonicity,
        0.5,
        f"Three idealized alteration proxies are individually ordered at {monotonicity:.3f}; their mean direction cosine is {separation:.3f}.",
        "No alteration labels or fresh–altered pairs exist in the trained view; closure and arbitrary magnitude choices dominate this synthetic test.",
        "Keep as a stress test only unless paired natural alteration data become available.",
    )


def experiment_e7(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"]
    meta = ctx.meta.copy()
    rows = []
    for rock_type, group in meta.groupby("rock_type", dropna=True):
        metamorphic = group.index[group["rock_group"].eq("metamorphic")].to_numpy()
        nonmeta = group.index[~group["rock_group"].eq("metamorphic")].to_numpy()
        if len(metamorphic) >= 10 and len(nonmeta) >= 10:
            distance = float(1 - cosine_matrix(latent[metamorphic].mean(axis=0, keepdims=True), latent[nonmeta].mean(axis=0, keepdims=True))[0, 0])
            rows.append({"rock_type": rock_type, "metamorphic_n": len(metamorphic), "nonmetamorphic_n": len(nonmeta), "centroid_cosine_distance": distance})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metamorphic_within_rock_type.csv", index=False, encoding="utf-8-sig")
    available = len(frame)
    mean_distance = float(frame["centroid_cosine_distance"].mean()) if available else np.nan
    return _result(
        "Weak/ambiguous" if available >= 3 else "Negative",
        {"matched_rock_types": available, "mean_metamorphic_nonmetamorphic_centroid_distance": mean_distance},
        "matched_rock_types",
        float(available),
        3.0,
        f"There are {available} rock_type strata with at least ten metamorphic and ten non-metamorphic samples; their mean centroid cosine distance is {mean_distance:.3f}.",
        "Rock_group metamorphic may encode protolith inheritance and naming practices; no metamorphic grade or paired protolith sample is available.",
        "If any strata are stable after source matching, analyze them as protolith-preservation cases rather than a universal metamorphic direction.",
    )


def experiment_f1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    meta = ctx.meta
    rhyolite_idx = np.flatnonzero(meta["rock_type"].eq("rhyolite").to_numpy())[:100]
    basalt_mask = meta["rock_type"].eq("subalkalic basalt").to_numpy()
    rhyolite_mask = meta["rock_type"].eq("rhyolite").to_numpy()
    latent_all = ctx.latents("numeric_only")["encoded_mean_numeric"]
    class_direction = latent_all[basalt_mask].mean(axis=0) - latent_all[rhyolite_mask].mean(axis=0)
    target = _median_type(ctx, "subalkalic basalt")
    raw = meta.iloc[rhyolite_idx][_numeric_db_cols(ctx)].copy()
    perturbed = raw.copy()
    major = ["sio2", "tio2", "al2o3", "fe2o3", "feo", "mgo", "cao", "na2o", "k2o", "p2o5", "mno"]
    for col in major:
        perturbed[col] = 0.8 * raw[col] + 0.2 * float(target[col])
    perturbed = _close_major(perturbed)
    perturbed["sample_id"] = np.arange(-60000 - len(perturbed), -60000)
    perturbed_latent = ctx.infer_raw_frame(perturbed)["encoded_mean_numeric"]
    delta = perturbed_latent - latent_all[rhyolite_idx]
    alignment = cosine_matrix(delta, class_direction[None, :])[:, 0]
    rows = pd.DataFrame({"sample_id": meta.iloc[rhyolite_idx]["sample_id"].to_numpy(), "delta_norm": np.linalg.norm(delta, axis=1), "alignment_with_rhyolite_to_basalt_centroid": alignment})
    rows.to_csv(out / "rhyolite_toward_basalt_perturbations.csv", index=False, encoding="utf-8-sig")
    mean_alignment = float(np.mean(alignment))
    return _result(
        effect_status(mean_alignment),
        {"samples": len(rows), "mean_alignment": mean_alignment, "positive_alignment_fraction": float(np.mean(alignment > 0))},
        "perturbation_to_class_direction_cosine",
        mean_alignment,
        0.0,
        f"A 20% closed major-element perturbation from rhyolite toward the basalt median aligns with the natural rhyolite→basalt latent centroid direction at mean cosine {mean_alignment:.3f}.",
        "Class directions are label-contaminated, and the perturbation changes only major oxides while natural basalt/rhyolite differences include trace elements and source effects.",
        "Repeat with multiple perturbation sizes and publication-matched class centroids if alignment is stable.",
    )


def _representative_backgrounds(ctx: ExperimentContext) -> pd.DataFrame:
    candidates = []
    for name, low, high in [("mafic", -np.inf, 52), ("intermediate", 52, 63), ("felsic", 63, np.inf)]:
        mask = ctx.meta["sio2"].between(low, high, inclusive="left") if np.isfinite(low) else ctx.meta["sio2"].lt(high)
        if not np.isfinite(high):
            mask = ctx.meta["sio2"].ge(low)
        subset = ctx.meta.loc[mask]
        # Use a median synthetic background so interventions do not silently
        # disappear merely because one representative publication omitted a
        # trace-element panel. All z fields remain within observed marginals.
        row = subset.median(numeric_only=True)
        for column in ctx.meta.columns:
            if column not in row.index:
                row[column] = np.nan
        row["background"] = name
        candidates.append(row)
    return pd.DataFrame(candidates).reset_index(drop=True)


def experiment_f2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    base = _representative_backgrounds(ctx)
    z_cols = [f"z__{tag}" for tag in ctx.numeric_tags]
    frames, mapping = [], []
    for background_i, row in base.iterrows():
        base_row = row.copy()
        base_row["sample_id"] = -70000 - len(mapping)
        frames.append(base_row)
        mapping.append((background_i, "BASE"))
        for tag in ctx.numeric_tags:
            perturbed = row.copy()
            perturbed[f"z__{tag}"] = float(perturbed[f"z__{tag}"]) + 0.5
            perturbed["sample_id"] = -70000 - len(mapping)
            frames.append(perturbed)
            mapping.append((background_i, tag))
    frame = pd.DataFrame(frames)
    latent = ctx.infer_transformed_frame(frame)["encoded_mean_numeric"]
    rows, directions = [], {}
    cursor = 0
    for background_i, background in enumerate(base["background"]):
        base_latent = latent[cursor]
        cursor += 1
        for tag in ctx.numeric_tags:
            delta = latent[cursor] - base_latent
            directions[(background, tag)] = delta
            rows.append({"background": background, "field": tag, "response_norm": float(np.linalg.norm(delta))})
            cursor += 1
    for tag in ctx.numeric_tags:
        vec = np.stack([directions[(background, tag)] for background in base["background"]])
        sim = cosine_matrix(vec)
        consistency = float(sim[np.triu_indices(3, 1)].mean())
        for row in rows:
            if row["field"] == tag:
                row["cross_background_direction_cosine"] = consistency
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(out / "single_element_finite_differences.csv", index=False, encoding="utf-8-sig")
    consistency = float(result_frame.groupby("field")["cross_background_direction_cosine"].first().mean())
    return _result(
        effect_status(consistency),
        {"fields": len(ctx.numeric_tags), "backgrounds": 3, "mean_cross_background_direction_cosine": consistency, "median_response_norm": float(result_frame["response_norm"].median())},
        "cross_background_single_element_direction_cosine",
        consistency,
        0.0,
        f"Single-field +0.5z interventions have mean direction cosine {consistency:.3f} across mafic, intermediate, and felsic backgrounds.",
        "Changing one transformed token ignores closure and may create impossible compositions; response directions include attention redistribution and missingness context.",
        "Fields with low consistency are candidates for genuine contextual role changes, but require compositionally valid perturbations.",
    )


def experiment_f3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    base = _representative_backgrounds(ctx)
    groups = {
        "mafic_shift": {"SiO2": -0.5, "MgO": 0.5, "FeO": 0.5, "CaO": 0.5},
        "incompatible_enrichment": {"Rb_ppm": 0.5, "Th_ppm": 0.5, "U_ppm": 0.5, "La_ppm": 0.5, "Ce_ppm": 0.5},
        "compatible_suite": {"Ni_ppm": 0.5, "Cr_ppm": 0.5, "Co_ppm": 0.5},
        "weathering_proxy": {"Na2O": -0.5, "CaO": -0.5, "Al2O3": 0.5},
    }
    frames, descriptors = [], []
    for _, row in base.iterrows():
        frames.append(row.copy())
        descriptors.append((row["background"], "BASE", "BASE"))
        for group, changes in groups.items():
            for tag, delta in changes.items():
                modified = row.copy()
                modified[f"z__{tag}"] = float(modified[f"z__{tag}"]) + delta
                frames.append(modified)
                descriptors.append((row["background"], group, tag))
            combined = row.copy()
            for tag, delta in changes.items():
                combined[f"z__{tag}"] = float(combined[f"z__{tag}"]) + delta
            frames.append(combined)
            descriptors.append((row["background"], group, "COMBINED"))
    frame = pd.DataFrame(frames)
    frame["sample_id"] = np.arange(-80000 - len(frame), -80000)
    latent = ctx.infer_transformed_frame(frame)["encoded_mean_numeric"]
    rows = []
    descriptors = pd.DataFrame(descriptors, columns=["background", "group", "component"])
    cursor = 0
    for background in base["background"]:
        block_mask = descriptors["background"].eq(background)
        block_desc = descriptors.loc[block_mask].reset_index(drop=True)
        block_latent = latent[cursor : cursor + len(block_desc)]
        cursor += len(block_desc)
        baseline = block_latent[0]
        for group, changes in groups.items():
            positions = block_desc.index[block_desc["group"].eq(group)].to_numpy()
            component_positions = [p for p in positions if block_desc.iloc[p]["component"] != "COMBINED"]
            combined_position = [p for p in positions if block_desc.iloc[p]["component"] == "COMBINED"][0]
            additive = np.sum([block_latent[p] - baseline for p in component_positions], axis=0)
            actual = block_latent[combined_position] - baseline
            cosine = float(cosine_matrix(actual[None, :], additive[None, :])[0, 0])
            residual_ratio = float(np.linalg.norm(actual - additive) / max(np.linalg.norm(actual), 1e-12))
            rows.append({"background": background, "group": group, "additive_direction_cosine": cosine, "interaction_residual_ratio": residual_ratio, "actual_norm": float(np.linalg.norm(actual))})
    result_frame = pd.DataFrame(rows)
    result_frame.to_csv(out / "combined_intervention_additivity.csv", index=False, encoding="utf-8-sig")
    additivity = float(result_frame["additive_direction_cosine"].mean())
    residual = float(result_frame["interaction_residual_ratio"].mean())
    unavailable = pd.DataFrame(
        [
            {"intervention": "Eu anomaly", "available": False, "reason": "Eu_ppm is not a trained tag"},
            {"intervention": "Nb-Ta anomaly", "available": False, "reason": "Ta_ppm is not a trained tag"},
        ]
    )
    unavailable.to_csv(out / "unavailable_combined_interventions.csv", index=False, encoding="utf-8-sig")
    return _result(
        effect_status(additivity),
        {"groups": len(groups), "backgrounds": 3, "mean_additive_direction_cosine": additivity, "mean_interaction_residual_ratio": residual, "unavailable_interventions": 2},
        "combined_vs_additive_direction_cosine",
        additivity,
        0.0,
        f"Four geochemical multi-field interventions have mean actual-versus-additive direction cosine {additivity:.3f} and interaction residual ratio {residual:.3f} across three backgrounds.",
        "Interventions operate in independently standardized z units and do not enforce closure; high additivity can be architectural rather than geological.",
        "Retain only strongly non-additive, background-stable combinations for compositionally closed follow-up tests.",
    )


def _norm_subset(ctx: ExperimentContext, n: int = 3000) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = ctx.meta.loc[ctx.meta["rock_group"].eq("igneous")].copy()
    candidates = candidates.loc[candidates[OXIDES].notna().sum(axis=1) >= 8]
    if len(candidates) > n:
        candidates = candidates.sample(n=n, random_state=ctx.seed).sort_index()
    norms, qc = cipw_like_nnls(candidates)
    return candidates.index.to_numpy(dtype=int), candidates.reset_index(drop=True), norms.reset_index(drop=True), qc.reset_index(drop=True)


def experiment_i0(_ctx: ExperimentContext, _out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    """Record the strict-CIPW boundary instead of substituting the NNLS proxy.

    The proxy experiments below are useful for representation benchmarking, but
    their simultaneous NNLS phase allocation is not the sequential CIPW norm.
    Raising here ensures the missing strict calculation remains a visible,
    reproducible technical failure with an errors.log entry.
    """
    raise NotImplementedError(
        "Strict sequential CIPW norm is not implemented or independently "
        "validated in this repository/environment. I1-I4 deliberately use a "
        "separately named CIPW-like stoichiometric NNLS proxy and must not be "
        "interpreted as formal CIPW results."
    )


def experiment_i1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    idx, meta, norms, qc = _norm_subset(ctx, 3000)
    valid = qc["reconstruction_rmse"].lt(2.0).to_numpy()
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"][idx]
    probe_components = min(24, latent.shape[1], len(latent) - 1)
    latent_pca = PCA(n_components=probe_components, random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(latent))
    raw = StandardScaler().fit_transform(meta[OXIDES].fillna(meta[OXIDES].median()).to_numpy(dtype=float))
    rows = []
    for mineral in norms.columns:
        y = norms[mineral].to_numpy(dtype=float)
        rows.append({"mineral": mineral, "latent_cv_r2": cv_ridge_r2(latent_pca[valid], y[valid], 5, ctx.seed), "raw_major_cv_r2": cv_ridge_r2(raw[valid], y[valid], 5, ctx.seed)})
    frame = pd.DataFrame(rows)
    frame["latent_minus_raw"] = frame["latent_cv_r2"] - frame["raw_major_cv_r2"]
    frame.to_csv(out / "normative_mineral_readout.csv", index=False, encoding="utf-8-sig")
    pd.concat([meta[["sample_id", "rock_type"]], norms, qc], axis=1).to_csv(out / "cipw_like_norms.csv", index=False, encoding="utf-8-sig")
    latent_mean = float(frame["latent_cv_r2"].mean())
    raw_mean = float(frame["raw_major_cv_r2"].mean())
    status = effect_status(latent_mean - raw_mean)
    return _result(
        status,
        {"samples": int(valid.sum()), "minerals": len(frame), "mean_latent_cv_r2": latent_mean, "mean_raw_major_cv_r2": raw_mean, "median_reconstruction_rmse": float(qc["reconstruction_rmse"].median()), "strict_cipw": False},
        "mean_normative_mineral_cv_r2",
        latent_mean,
        raw_mean,
        f"The {probe_components}D latent predicts the CIPW-like NNLS mineral proxies at mean CV R²={latent_mean:.3f}, versus {raw_mean:.3f} from the ten input oxides.",
        "This is not a strict CIPW allocation: NNLS permits non-unique and mutually incompatible normative phases. It is a broad mineralogical proxy and is explicitly labeled as such.",
        "Install or validate a formal CIPW implementation before promoting mineral-specific claims; retain this result only as a baseline audit.",
    )


def experiment_i2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    idx, meta, norms, qc = _norm_subset(ctx, 3000)
    groups = pd.DataFrame(
        {
            "quartz_feldspar": norms[["Q", "Or", "Ab"]].sum(axis=1),
            "plagioclase_pyroxene": norms[["An", "DiMg", "DiFe", "HyMg", "HyFe"]].sum(axis=1),
            "olivine": norms[["OlMg", "OlFe"]].sum(axis=1),
            "nepheline": norms["Ne"],
        }
    )
    assemblage = groups.idxmax(axis=1).to_numpy()
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"][idx]
    raw = meta[OXIDES].fillna(meta[OXIDES].median()).to_numpy(dtype=float)
    latent_purity = knn_purity(latent, assemblage, 15)
    raw_purity = knn_purity(raw, assemblage, 15)
    pd.concat([meta[["sample_id", "rock_type"]], groups, pd.Series(assemblage, name="dominant_proxy_assemblage")], axis=1).to_csv(out / "normative_assemblages.csv", index=False, encoding="utf-8-sig")
    return _result(
        effect_status(latent_purity - raw_purity),
        {"latent_assemblage_knn15": latent_purity, "raw_assemblage_knn15": raw_purity, "samples": len(meta)},
        "normative_assemblage_knn15",
        latent_purity,
        raw_purity,
        f"Latent kNN15 purity for four dominant normative-proxy assemblages is {latent_purity:.3f}, versus {raw_purity:.3f} in oxide space.",
        "Assemblage labels are derived by argmax from an approximate and non-unique NNLS norm, so they are not mineral observations.",
        "Use only as a relative latent-versus-raw check until strict CIPW norms are available.",
    )


def experiment_i3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    ideal = ideal_mineral_mixtures()
    for col in _numeric_db_cols(ctx):
        if col not in ideal:
            ideal[col] = np.nan
    ideal["sample_id"] = np.arange(-90000 - len(ideal), -90000)
    latent = ctx.infer_raw_frame(ideal)["encoded_mean_numeric"]
    natural = ctx.latents("numeric_only")["encoded_mean_numeric"]
    centroids, names, counts = centroid_by_label(natural, ctx.meta["rock_type"], min_count=20)
    sim = cosine_matrix(latent, centroids)
    expected = {
        "quartz_feldspar_rich": ["granite", "rhyolite"],
        "plagioclase_pyroxene_rich": ["subalkalic basalt", "subalkalic gabbro"],
        "olivine_rich": ["picrite", "alkali picrite"],
        "nepheline_alkaline": ["tephrite", "phonolite", "mafic foidite"],
    }
    rows = []
    reciprocal = []
    for i, mixture in enumerate(ideal["ideal_mixture"]):
        order = np.argsort(-sim[i])
        for rank, j in enumerate(order[:10], start=1):
            rows.append({"ideal_mixture": mixture, "rank": rank, "retrieved_rock_type": names[j], "cosine": float(sim[i, j]), "class_count": int(counts[j])})
        expected_ranks = [int(np.flatnonzero(order == names.index(label))[0]) + 1 for label in expected[mixture] if label in names]
        reciprocal.append(1.0 / min(expected_ranks) if expected_ranks else 0.0)
    pd.DataFrame(rows).to_csv(out / "ideal_normative_retrieval.csv", index=False, encoding="utf-8-sig")
    mrr = float(np.mean(reciprocal))
    chance = float(np.mean(1.0 / np.arange(1, len(names) + 1)))
    return _result(
        effect_status(mrr - chance),
        {"ideal_mixtures": len(ideal), "expected_family_mrr": mrr, "chance_mrr": chance},
        "ideal_mixture_expected_family_mrr",
        mrr,
        chance,
        f"Four ideal normative mixtures retrieve an expected rock family with mean reciprocal rank {mrr:.3f}, versus chance-rank MRR {chance:.3f}.",
        "Ideal oxide mixtures omit trace elements and may be out of the training distribution; expected families are broad and the norm construction is approximate.",
        "Inspect retrieved neighbors and input-space distances before interpreting failures as model semantic failures.",
    )


def experiment_i4(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    idx, meta, norms, qc = _norm_subset(ctx, 2000)
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"][idx]
    pure_names = ctx.pure_tokens
    pure_projected = ctx.project_pure_labels(trained=True)
    token_map = {name: pure_projected[i] for i, name in enumerate(pure_names)}
    class_centroids, class_names, _ = centroid_by_label(latent, meta["rock_type"], min_count=10)
    common = [name for name in class_names if f"rock_type:{name}" in token_map]
    x = np.stack([token_map[f"rock_type:{name}"] for name in common])
    y = np.stack([class_centroids[class_names.index(name)] for name in common])
    xpca = PCA(n_components=min(24, len(common) - 2, x.shape[1])).fit_transform(StandardScaler().fit_transform(x))
    mapper = Ridge(alpha=10.0).fit(xpca, y)
    label_pca = PCA(n_components=xpca.shape[1]).fit(StandardScaler().fit_transform(x))
    mapped = {}
    for name in common:
        coord = label_pca.transform(StandardScaler().fit_transform(x))[common.index(name) : common.index(name) + 1]
        mapped[name] = mapper.predict(coord)[0]
    valid = meta["rock_type"].isin(common).to_numpy()
    residual = latent[valid] - np.stack([mapped[name] for name in meta.loc[valid, "rock_type"].astype(str)])
    residual_components = min(12, residual.shape[1], len(residual) - 1)
    residual_pc = PCA(n_components=residual_components, random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(residual))
    rows = []
    for mineral in norms.columns:
        best = 0.0
        best_axis = None
        for axis in range(residual_pc.shape[1]):
            rho, p = safe_spearman(residual_pc[:, axis], norms.loc[valid, mineral].to_numpy(dtype=float))
            if np.isfinite(rho) and abs(rho) > abs(best):
                best, best_axis = rho, axis + 1
        rows.append({"mineral": mineral, "best_residual_pc": best_axis, "best_abs_spearman": abs(best), "signed_spearman": best})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "label_residual_normative_correlations.csv", index=False, encoding="utf-8-sig")
    mean_best = float(frame["best_abs_spearman"].mean())
    return _result(
        "Weak/ambiguous",
        {"samples": int(valid.sum()), "classes": len(common), "mean_best_residual_pc_abs_spearman": mean_best},
        "label_residual_normative_abs_spearman",
        mean_best,
        0.0,
        f"After a simple label→centroid map, residual PCs have mean best |ρ|={mean_best:.3f} with CIPW-like mineral proxies.",
        "The alignment is fit on the same classes and the norm is approximate; selecting the best of 12 PCs inflates correlations and is exploratory only.",
        "Use held-out classes and strict CIPW norms before interpreting any residual direction.",
    )


@torch.no_grad()
def _encode_items(ctx: ExperimentContext, items: list[dict]) -> np.ndarray:
    model, device = ctx.full_model()
    batch = collate_token_batch(items)
    moved = move_batch_to_device(batch, device)
    encoded = model.encode(moved)
    token = encoded["encoded"][:, 1:, :]
    numeric_mask = moved["numeric_token_mask"] & moved["attention_mask"]
    denom = numeric_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    pooled = (token * numeric_mask.unsqueeze(-1)).sum(dim=1) / denom
    return pooled.cpu().numpy().astype(np.float32)


def experiment_k3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    frame = ctx.meta.iloc[:128].copy()
    base_items = [_make_item(row, ctx.source_config, ctx.vocab, "numeric_only") for _, row in frame.iterrows()]
    base = _encode_items(ctx, base_items)
    shuffled_items = copy.deepcopy(base_items)
    for item in shuffled_items:
        rng.shuffle(item["tokens"])
    shuffled = _encode_items(ctx, shuffled_items)
    order_cosine = np.sum(row_normalize(base) * row_normalize(shuffled), axis=1)
    rows = []
    for fraction in [0.10, 0.30]:
        dropped_items = copy.deepcopy(base_items)
        for item in dropped_items:
            n_drop = max(1, int(round(len(item["tokens"]) * fraction)))
            keep = np.ones(len(item["tokens"]), dtype=bool)
            keep[rng.choice(len(item["tokens"]), size=min(n_drop, len(item["tokens"]) - 1), replace=False)] = False
            item["tokens"] = [token for token, retained in zip(item["tokens"], keep) if retained]
        dropped = _encode_items(ctx, dropped_items)
        cosine = np.sum(row_normalize(base) * row_normalize(dropped), axis=1)
        rows.append({"drop_fraction": fraction, "mean_cosine": float(cosine.mean()), "p05_cosine": float(np.quantile(cosine, 0.05))})
    dropout = pd.DataFrame(rows)
    dropout.to_csv(out / "token_dropout_stability.csv", index=False, encoding="utf-8-sig")
    order_mean = float(order_cosine.mean())
    drop30 = float(dropout.loc[dropout["drop_fraction"] == 0.30, "mean_cosine"].iloc[0])
    status = "Strong positive" if order_mean > 0.99999 and drop30 > 0.8 else "Weak/ambiguous"
    return _result(
        status,
        {"samples": len(frame), "token_order_mean_cosine": order_mean, "token_order_min_cosine": float(order_cosine.min()), "drop10_mean_cosine": float(dropout.iloc[0]["mean_cosine"]), "drop30_mean_cosine": drop30},
        "token_order_invariance_cosine",
        order_mean,
        1.0,
        f"Random token permutation preserves encoded-mean latent at mean cosine {order_mean:.8f}; random 30% token removal retains mean cosine {drop30:.3f}.",
        "Dropout is uniform over observed fields and does not mimic element-specific analytical panels or detection limits.",
        "Repeat dropout by realistic analytical panels and field coverage strata for any robustness claim.",
    )


def run_process_experiments(ctx: ExperimentContext, registry: ExperimentRegistry, only: set[str] | None = None) -> None:
    definitions = [
        (_spec("E1_two_endmember_mixing", "Linear and closed two-endmember mixing", "E", "chemistry-only encoded sample latent", "mixing continuity and curvature", "Geologically motivated endmember mixtures should trace continuous, ordered latent paths.", ctx, 168, "linear versus major-closed mixing"), experiment_e1),
        (_spec("E2_multiendmember_simplex", "Three-endmember mixture topology", "E", "chemistry-only encoded sample latent", "barycentric simplex topology", "Latent distance should preserve the broad topology of a basalt–rhyolite–limestone simplex.", ctx, 66, "barycentric input geometry"), experiment_e2),
        (_spec("E3_magmatic_differentiation", "Natural magmatic differentiation sequences", "E", "chemistry-only class-centroid latent", "primitive-to-evolved ordering", "Volcanic and plutonic class centroids should show ordered, partly shared differentiation directions.", ctx, 10, "volcanic versus plutonic sequence"), experiment_e3),
        (_spec("E4_partial_melting", "Simplified batch-melting paths", "E", "chemistry-only encoded sample latent", "melt fraction, source enrichment, residual mineral proxy", "Melt-fraction trajectories should be ordered and distinguish source/residual branches.", ctx, 60, "depleted/enriched and spinel/garnet proxies"), experiment_e4),
        (_spec("E5_weathering_sediment", "Weathering and K-metasomatism proxies", "E", "chemistry-only encoded sample latent", "weathering intensity and branch divergence", "Closed weathering proxies should form ordered paths and distinguish K metasomatism.", ctx, 42, "leaching versus K-metasomatism branch"), experiment_e5),
        (_spec("E6_hydrothermal", "Hydrothermal alteration proxy paths", "E", "chemistry-only encoded sample latent", "potassic, silicification, propylitic proxy geometry", "Different alteration-style perturbations should produce distinguishable ordered directions.", ctx, 48, "three synthetic alteration styles"), experiment_e6),
        (_spec("E7_metamorphic", "Metamorphic inheritance within matched rock type", "E", "chemistry-only encoded sample latent", "metamorphic versus nonmetamorphic within type", "Where labels permit, metamorphic samples may retain protolith identity with a secondary contextual shift.", ctx, 12000, "within-rock_type matched groups"), experiment_e7),
        (_spec("F1_lithology_perturbation", "Compositionally closed rhyolite→basalt perturbation", "F", "chemistry-only encoded sample latent", "local perturbation versus natural class direction", "A constrained mafic-direction perturbation should align with the natural rhyolite→basalt latent direction.", ctx, 100, "natural class-centroid direction"), experiment_f1),
        (_spec("F2_single_element_jacobian", "Single-element finite-difference responses", "F", "chemistry-only encoded sample latent", "field sensitivity and background dependence", "Single-field response directions should reveal both shared and background-specific contextual roles.", ctx, 102, "three compositional backgrounds"), experiment_f2),
        (_spec("F3_combined_interventions", "Combined geochemical intervention additivity", "F", "chemistry-only encoded sample latent", "multi-field nonlinear interactions", "Combined interventions may deviate from the sum of individual latent displacements.", ctx, 75, "sum of single-field displacements"), experiment_f3),
        (_spec("I0_strict_cipw_unavailable", "Strict sequential CIPW norm availability", "I", "major-oxide composition", "formal sequential CIPW normative minerals", "Applicable igneous samples should receive an independently validated strict CIPW norm before mineral-specific latent claims.", ctx, 0, "CIPW-like NNLS proxy kept separate in I1-I4"), experiment_i0),
        (_spec("I1_normative_readability", "CIPW-like normative mineral readability", "I", "chemistry-only encoded sample latent", "normative Q/Or/Ab/An/Di/Hy/Ol/Ne/Mt/Il/Ap", "Low-capacity latent probes should recover broad normative mineral variation.", ctx, 3000, "raw major oxides"), experiment_i1),
        (_spec("I2_normative_assemblages", "Normative assemblage neighborhood relation", "I", "chemistry-only encoded sample latent", "dominant normative assemblage", "Latent neighborhoods should organize broad normative assemblage proxies.", ctx, 3000, "raw major-oxide neighborhoods"), experiment_i2),
        (_spec("I3_ideal_norm_retrieval", "Idealized normative mixture retrieval", "I", "chemistry-only encoded sample latent", "ideal mineral mixtures to natural rock centroids", "Ideal mineral assemblages should retrieve broadly compatible rock families.", ctx, 4, "chance class rank"), experiment_i3),
        (_spec("I4_label_residual_norms", "Label-aligned residual mineral variation", "I", "sample latent residual after label mapping", "within-label normative variation", "Residual latent directions within a label may retain normative mineral variation.", ctx, 2000, "label-mapped class component"), experiment_i4),
        (_spec("K3_order_dropout_robustness", "Permutation invariance and token-dropout robustness", "K", "chemistry-only encoded sample latent", "set invariance and missing-panel robustness", "No-position encoding should be order-invariant and degrade smoothly under token removal.", ctx, 128, "original token sets"), experiment_k3),
    ]
    for spec, function in definitions:
        if only is not None and spec.experiment_id not in only:
            continue
        print(f"[experiment] {spec.experiment_id}")
        registry.run(spec, lambda out, s, f=function: f(ctx, out, s))
