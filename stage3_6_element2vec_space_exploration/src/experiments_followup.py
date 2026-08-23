from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, pairwise_distances, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis_utils import centroid_by_label, cosine_matrix, effect_status, knn_indices, knn_purity, permutation_pvalue, row_normalize, safe_spearman
from .catalogs import VOLCANIC_PLUTONIC_PAIRS
from .experiment_context import ExperimentContext
from .registry import ExperimentRegistry, ExperimentSpec


def _spec(experiment_id: str, name: str, family: str, level: str, relation: str, hypothesis: str, ctx: ExperimentContext, sample_size: int, control: str) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        experiment_name=name,
        family=family,
        representation_level=level,
        dataset="V2 stratified chemistry-only exploration sample",
        input_fields="34 numeric tokens plus external audit metadata",
        target_geological_relation=relation,
        seed=ctx.seed,
        sample_size=sample_size,
        control_group=control,
        hypothesis=hypothesis,
    )


def _result(status: str, metrics: dict[str, Any], primary: str, value: float | None, baseline: float | None, conclusion: str, risks: str, next_step: str, stability: str = "targeted follow-up to first-round anomaly") -> dict[str, Any]:
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


def _group_balanced_accuracy(x: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int, pca_components: int | None = None) -> float:
    labels = np.asarray(labels).astype(str)
    groups = pd.Series(groups).fillna("[MISSING]").astype(str).to_numpy()
    valid = (labels != "") & (labels != "nan")
    x, labels, groups = x[valid], labels[valid], groups[valid]
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3 or len(np.unique(labels)) < 2:
        return float("nan")
    n_splits = min(5, len(unique_groups))
    steps = [StandardScaler()]
    if pca_components:
        steps.append(PCA(n_components=min(pca_components, x.shape[1], len(x) - 1), random_state=seed))
    steps.append(LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1))
    model = make_pipeline(*steps)
    predictions = np.empty(len(labels), dtype=object)
    for train, test in GroupKFold(n_splits=n_splits).split(x, labels, groups):
        model.fit(x[train], labels[train])
        predictions[test] = model.predict(x[test])
    return float(balanced_accuracy_score(labels, predictions))


def _group_ridge_r2(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    groups = pd.Series(groups).fillna("[MISSING]").astype(str).to_numpy()
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y, groups = x[valid], y[valid], groups[valid]
    if len(y) < 50 or len(np.unique(groups)) < 3:
        return float("nan")
    predictions = np.zeros(len(y), dtype=float)
    for train, test in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(x, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
    return float(r2_score(y, predictions))


def _token_join(ctx: ExperimentContext) -> pd.DataFrame:
    columns = ["sample_id", "sio2", "mg_number", "mali", "total_alkali", "log_sr_y", "num_numeric_tokens", "canonical_ref_id"]
    token = ctx.token_meta.drop(columns=[c for c in ["canonical_ref_id"] if c in ctx.token_meta])
    return token.merge(ctx.meta[columns], on="sample_id", how="left", validate="many_to_one")


def experiment_x1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    meta = _token_join(ctx)
    arrays = ctx.token_arrays()
    z_cols = [f"z__{tag}" for tag in ctx.numeric_tags]
    sample_features = ctx.meta.set_index("sample_id")[z_cols]
    representatives = ["Ni_ppm", "Sr_ppm", "Zr_ppm", "Rb_ppm", "K2O", "CaO"]
    rows = []
    for field in representatives:
        mask = meta["field"].eq(field).to_numpy() & meta["normalized_value"].abs().le(0.30).to_numpy()
        subset = meta.loc[mask].reset_index(drop=True)
        if len(subset) < 120:
            continue
        sio2 = subset["sio2"].to_numpy(dtype=float)
        labels = np.where(sio2 < 52, "mafic", np.where(sio2 > 63, "felsic", "intermediate"))
        groups = subset["canonical_ref_id"].to_numpy()
        raw = sample_features.loc[subset["sample_id"]].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.0)
        projected = _group_balanced_accuracy(arrays["projected"][mask], labels, groups, ctx.seed, 16)
        contextual = _group_balanced_accuracy(arrays["contextual"][mask], labels, groups, ctx.seed, 16)
        raw_score = _group_balanced_accuracy(raw, labels, groups, ctx.seed, None)
        rows.append({"field": field, "n": len(subset), "projected_grouped_score": projected, "contextual_grouped_score": contextual, "raw_context_grouped_score": raw_score, "contextual_minus_raw": contextual - raw_score})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "same_value_grouped_raw_context_control.csv", index=False, encoding="utf-8-sig")
    contextual = float(frame["contextual_grouped_score"].mean())
    raw = float(frame["raw_context_grouped_score"].mean())
    delta = contextual - raw
    return _result(
        effect_status(delta),
        {"fields": len(frame), "mean_grouped_contextual_score": contextual, "mean_grouped_raw_context_score": raw, "contextual_minus_raw": delta},
        "grouped_background_balanced_accuracy",
        contextual,
        raw,
        f"The large C1 contextual signal survives publication grouping at {contextual:.3f}, but full raw chemistry reaches {raw:.3f}; contextual-minus-raw is {delta:+.3f}.",
        "The background label is defined directly by SiO₂, which is present in raw context and attended by the contextual token. This tests context transmission, not information beyond chemistry.",
        "Treat C1 as evidence that attention writes whole-sample context into a token, not as a novel geological discovery unless it exceeds raw-context controls on a nontrivial target.",
    )


def experiment_x2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    meta = _token_join(ctx)
    arrays = ctx.token_arrays()
    z_cols = [f"z__{tag}" for tag in ctx.numeric_tags]
    raw_by_id = ctx.meta.set_index("sample_id")[z_cols]
    definitions = [("Sr_ppm", "log_sr_y"), ("Ni_ppm", "mg_number"), ("Cr_ppm", "mg_number"), ("Rb_ppm", "sio2"), ("Zr_ppm", "sio2"), ("K2O", "mali"), ("CaO", "total_alkali")]
    rows = []
    for field, proxy in definitions:
        field_mask = meta["field"].eq(field) & meta[proxy].notna()
        subset = meta.loc[field_mask].reset_index(drop=True)
        if len(subset) < 150:
            continue
        labels = pd.qcut(subset[proxy], q=3, labels=["low", "mid", "high"], duplicates="drop").astype(str).to_numpy()
        groups = subset["canonical_ref_id"].to_numpy()
        mask = field_mask.to_numpy()
        raw = np.nan_to_num(raw_by_id.loc[subset["sample_id"]].to_numpy(dtype=float), nan=0.0)
        contextual = _group_balanced_accuracy(arrays["contextual"][mask], labels, groups, ctx.seed, 16)
        raw_score = _group_balanced_accuracy(raw, labels, groups, ctx.seed, None)
        rows.append({"field": field, "proxy": proxy, "n": len(subset), "contextual_grouped_score": contextual, "raw_grouped_score": raw_score, "contextual_minus_raw": contextual - raw_score})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "contextual_role_grouped_raw_control.csv", index=False, encoding="utf-8-sig")
    contextual = float(frame["contextual_grouped_score"].mean())
    raw = float(frame["raw_grouped_score"].mean())
    return _result(
        effect_status(contextual - raw),
        {"proxies": len(frame), "mean_contextual_grouped_score": contextual, "mean_raw_grouped_score": raw, "contextual_minus_raw": contextual - raw},
        "grouped_role_proxy_balanced_accuracy",
        contextual,
        raw,
        f"After publication grouping, contextual role-proxy readability is {contextual:.3f}, versus {raw:.3f} from the full transformed chemical panel.",
        "The process proxies are algebraic or strongly correlated functions of input chemistry, so matching raw performance is expected and exceeding it would require careful validation.",
        "Promote only individual fields with a stable contextual-over-raw gain and a process target not algebraically defined by the inputs.",
    )


def experiment_x3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    pure = (
        ctx.project_pure_labels_for_sample_space(trained=True)
        if hasattr(ctx, "project_pure_labels_for_sample_space")
        else ctx.project_pure_labels(trained=True)
    )
    token_map = dict(zip(ctx.pure_tokens, pure))
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"]
    sample_centroids, sample_names, counts = centroid_by_label(latent, ctx.meta["rock_type"], min_count=20)
    sample_map = dict(zip(sample_names, sample_centroids))
    label_rel, sample_rel, names = [], [], []
    for volcanic, plutonic in VOLCANIC_PLUTONIC_PAIRS:
        left, right = f"rock_type:{volcanic}", f"rock_type:{plutonic}"
        if left in token_map and right in token_map and volcanic in sample_map and plutonic in sample_map:
            label_rel.append(token_map[right] - token_map[left])
            sample_rel.append(sample_map[plutonic] - sample_map[volcanic])
            names.append(f"{volcanic}->{plutonic}")
    label_rel, sample_rel = np.stack(label_rel), np.stack(sample_rel)
    label_sim = cosine_matrix(label_rel)
    sample_sim = cosine_matrix(sample_rel)
    tri = np.triu_indices(len(names), 1)
    rho, p = safe_spearman(label_sim[tri], sample_sim[tri])
    rng = np.random.default_rng(ctx.seed)
    null = []
    for _ in range(int(ctx.config["analysis"]["permutation_repeats"])):
        perm = rng.permutation(len(names))
        null.append(safe_spearman(label_sim[tri], sample_sim[perm][:, perm][tri])[0])
    perm_p = permutation_pvalue(rho, np.asarray(null), two_sided=True)
    pd.DataFrame(label_sim, index=names, columns=names).to_csv(out / "label_relation_similarity.csv", encoding="utf-8-sig")
    pd.DataFrame(sample_sim, index=names, columns=names).to_csv(out / "sample_centroid_relation_similarity.csv", encoding="utf-8-sig")
    return _result(
        effect_status(rho, perm_p),
        {"pairs": len(names), "relation_similarity_matrix_spearman": rho, "permutation_p": perm_p, "label_relation_cohesion": float(label_sim[tri].mean()), "sample_relation_cohesion": float(sample_sim[tri].mean())},
        "cross_space_relation_structure_spearman",
        rho,
        0.0,
        f"Volcanic–plutonic relation similarity matrices agree between label and chemistry-only class-centroid spaces at ρ={rho:.3f} (permutation p={perm_p:.3g}).",
        "Both spaces inherit the same label-supervised checkpoint; this comparison avoids direct cross-space vector subtraction but not training-label contamination.",
        "Only if relation structure is significant should individual analogy directions be examined within publication-held-out centroids.",
    )


def experiment_x4(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    n = min(3000, len(ctx.latents("random_numeric_only")["encoded_mean_numeric"]))
    meta = ctx.meta.iloc[:n].reset_index(drop=True)
    trained = ctx.latents("numeric_only")["encoded_mean_numeric"][:n]
    random = ctx.latents("random_numeric_only")["encoded_mean_numeric"][:n]
    raw = ctx.raw_baseline()[:n]
    mask_only = raw[:, 34:]
    spaces = {"mask_only": mask_only, "raw_values_plus_mask": raw, "random_encoder": random, "trained_encoder": trained}
    subsets: dict[int, list[tuple[int, np.ndarray]]] = {}
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        subsets[seed] = [
            (sample_size, np.sort(rng.choice(n, size=min(sample_size, n), replace=False)))
            for sample_size in [1000, 2000, 3000]
        ]

    rock_labels = meta["rock_type"].fillna("[MISSING]").astype(str).to_numpy()
    source_labels = meta["data_source"].fillna("[MISSING]").astype(str).to_numpy()

    def evaluate_space(name: str, x: np.ndarray) -> list[dict[str, Any]]:
        space_rows = []
        for metric in ["cosine", "euclidean"]:
            # Compute the 3000x3000 distance matrix once, then slice it for all
            # three sample-size controls. This preserves the original subsets
            # and metrics while avoiding three repeated neighbor searches.
            distances = pairwise_distances(x, metric=metric, n_jobs=1)
            np.fill_diagonal(distances, np.inf)
            for seed, seed_subsets in subsets.items():
                for requested_size, idx in seed_subsets:
                    local = distances[np.ix_(idx, idx)]
                    k = min(15, len(idx) - 1)
                    neighbors_local = np.argpartition(local, kth=k - 1, axis=1)[:, :k]
                    neighbors = idx[neighbors_local]
                    space_rows.append(
                        {
                            "seed": seed,
                            "sample_size": len(idx),
                            "space": name,
                            "distance": metric,
                            "rock_type_purity": float(np.mean(rock_labels[neighbors] == rock_labels[idx, None])),
                            "source_purity": float(np.mean(source_labels[neighbors] == source_labels[idx, None])),
                        }
                    )
            del distances
        return space_rows

    rows = []
    shared_path = None
    if hasattr(ctx, "dimension"):
        shared_path = Path(ctx.config["project"]["reports_dir"]).parent / "shared_controls" / "x4_raw_mask_metrics.csv"
    if shared_path is not None and shared_path.exists():
        rows.extend(pd.read_csv(shared_path).to_dict("records"))
    else:
        invariant_rows = []
        for name in ["mask_only", "raw_values_plus_mask"]:
            invariant_rows.extend(evaluate_space(name, spaces[name]))
        rows.extend(invariant_rows)
        if shared_path is not None:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(invariant_rows).to_csv(shared_path, index=False, encoding="utf-8-sig")
    for name in ["random_encoder", "trained_encoder"]:
        rows.extend(evaluate_space(name, spaces[name]))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "missingness_random_seed_distance_controls.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby("space")[["rock_type_purity", "source_purity"]].agg(["mean", "std"])
    summary.to_csv(out / "missingness_random_summary.csv", encoding="utf-8-sig")
    trained_geo = float(frame.loc[frame["space"] == "trained_encoder", "rock_type_purity"].mean())
    random_geo = float(frame.loc[frame["space"] == "random_encoder", "rock_type_purity"].mean())
    trained_source = float(frame.loc[frame["space"] == "trained_encoder", "source_purity"].mean())
    random_source = float(frame.loc[frame["space"] == "random_encoder", "source_purity"].mean())
    mask_source = float(frame.loc[frame["space"] == "mask_only", "source_purity"].mean())
    net = (trained_geo - random_geo) - (trained_source - random_source)
    status = effect_status(net)
    if random_source > 0.8 or mask_source > 0.8:
        status = "Invalid due to leakage/confounding"
    return _result(
        status,
        {"trained_rock_type_purity_mean": trained_geo, "random_rock_type_purity_mean": random_geo, "trained_source_purity_mean": trained_source, "random_source_purity_mean": random_source, "mask_only_source_purity_mean": mask_source, "net_trained_geology_minus_source_gain": net, "seeds": 3, "sample_sizes": [1000, 2000, 3000], "distance_metrics": ["cosine", "euclidean"]},
        "net_geology_over_source_gain_vs_random",
        trained_geo - random_geo,
        trained_source - random_source,
        f"On exactly matched samples, training changes rock-type purity by {trained_geo-random_geo:+.3f} and source purity by {trained_source-random_source:+.3f}; mask-only source purity is {mask_source:.3f}.",
        "A random set encoder can preserve or amplify analytical-panel patterns; sample_id ordering and source imbalance also affect this control despite matched rows.",
        "Any promoted latent neighborhood result must report mask-only and random-encoder source purity on the same samples.",
    )


def _haversine_matrix_to_neighbors(lat: np.ndarray, lon: np.ndarray, idx: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat)[:, None]
    lon1 = np.radians(lon)[:, None]
    lat2 = np.radians(lat[idx])
    lon2 = np.radians(lon[idx])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def experiment_d4(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    idx = np.sort(rng.choice(len(ctx.meta), size=min(3000, len(ctx.meta)), replace=False))
    meta = ctx.meta.iloc[idx].reset_index(drop=True)
    spaces = {"raw": ctx.raw_baseline()[idx], "encoded": ctx.latents("numeric_only")["encoded_mean_numeric"][idx], "random": ctx.latents("random_numeric_only")["encoded_mean_numeric"][: min(3000, len(ctx.latents("random_numeric_only")["encoded_mean_numeric"]))]}
    rows = []
    for name, x in list(spaces.items())[:2]:
        neighbors = knn_indices(x, 15)
        distance = _haversine_matrix_to_neighbors(meta["latitude"].to_numpy(dtype=float), meta["longitude"].to_numpy(dtype=float), neighbors)
        rows.append({"space": name, "median_neighbor_km": float(np.nanmedian(distance)), "p25_neighbor_km": float(np.nanquantile(distance, 0.25)), "same_reference_purity": float(np.mean(meta["canonical_ref_id"].astype(str).to_numpy()[neighbors] == meta["canonical_ref_id"].astype(str).to_numpy()[:, None]))})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "geographic_neighbor_audit.csv", index=False, encoding="utf-8-sig")
    raw_km = float(frame.loc[frame["space"] == "raw", "median_neighbor_km"].iloc[0])
    encoded_km = float(frame.loc[frame["space"] == "encoded", "median_neighbor_km"].iloc[0])
    return _result(
        "Weak/ambiguous",
        {"raw_median_neighbor_km": raw_km, "encoded_median_neighbor_km": encoded_km, "encoded_minus_raw_km": encoded_km - raw_km},
        "median_geographic_neighbor_distance_km",
        encoded_km,
        raw_km,
        f"Encoded neighbors have median geographic separation {encoded_km:.1f} km versus {raw_km:.1f} km in raw chemistry.",
        "Coordinates may be rounded or publication-level, and geographic proximity can represent real suites as well as leakage.",
        "Use spatial cells jointly with canonical publication IDs for a stricter neighborhood audit.",
    )


def experiment_d5(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    n = min(6000, len(ctx.meta))
    idx = np.sort(rng.choice(len(ctx.meta), size=n, replace=False))
    meta = ctx.meta.iloc[idx].reset_index(drop=True)
    groups = meta["canonical_ref_id"].to_numpy()
    raw_input = ctx.raw_baseline()[idx]
    latent_input = ctx.latents("numeric_only")["encoded_mean_numeric"][idx]
    raw = PCA(n_components=min(24, raw_input.shape[1], len(raw_input) - 1), random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(raw_input))
    latent = PCA(n_components=min(24, latent_input.shape[1], len(latent_input) - 1), random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(latent_input))
    targets = ["sr87_sr86", "epsilon_nd", "epsilon_hf", "pb206_pb204", "pb207_pb204", "pb208_pb204", "age"]
    rows = []
    for target in targets:
        y = meta[target].to_numpy(dtype=float)
        rows.append({"target": target, "coverage": float(np.isfinite(y).mean()), "raw_grouped_r2": _group_ridge_r2(raw, y, groups), "latent_grouped_r2": _group_ridge_r2(latent, y, groups)})
    frame = pd.DataFrame(rows)
    frame["latent_minus_raw"] = frame["latent_grouped_r2"] - frame["raw_grouped_r2"]
    frame.to_csv(out / "external_isotope_age_readout.csv", index=False, encoding="utf-8-sig")
    valid = frame["coverage"].ge(0.05) & frame["latent_grouped_r2"].notna()
    latent_mean = float(frame.loc[valid, "latent_grouped_r2"].mean()) if valid.any() else np.nan
    raw_mean = float(frame.loc[valid, "raw_grouped_r2"].mean()) if valid.any() else np.nan
    return _result(
        effect_status(latent_mean - raw_mean),
        {"targets_with_coverage": int(valid.sum()), "mean_latent_grouped_r2": latent_mean, "mean_raw_grouped_r2": raw_mean},
        "external_isotope_age_grouped_r2",
        latent_mean,
        raw_mean,
        f"For external isotope/age targets with at least 5% coverage, publication-grouped latent R² averages {latent_mean:.3f}, versus {raw_mean:.3f} for raw transformed chemistry.",
        "Isotope coverage is sparse and publication-specific; grouped folds reduce but do not eliminate analytical-source confounding.",
        "Inspect only targets with positive held-publication R² and adequate coverage before geological interpretation.",
    )


def experiment_j2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    n = min(6000, len(ctx.meta))
    idx = np.sort(rng.choice(len(ctx.meta), size=n, replace=False))
    meta = ctx.meta.iloc[idx].reset_index(drop=True)
    q_cols = ["sio2", "mg_number", "total_alkali", "cia", "log_sr_y", "log_la_yb", "log_th_sc", "log_nb_u"]
    q = meta[q_cols].copy()
    for col in q_cols:
        q[col] = q[col].fillna(q[col].median())
    qx = q.to_numpy(dtype=float)
    latent_input = ctx.latents("numeric_only")["encoded_mean_numeric"][idx]
    latent = PCA(n_components=min(24, latent_input.shape[1], len(latent_input) - 1), random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(latent_input))
    combined = np.concatenate([qx, latent], axis=1)
    groups = meta["canonical_ref_id"].to_numpy()
    rows = []
    for target in ["rock_group", "rock_origin"]:
        labels = meta[target].fillna("[MISSING]").astype(str).to_numpy()
        base = _group_balanced_accuracy(qx, labels, groups, ctx.seed)
        augmented = _group_balanced_accuracy(combined, labels, groups, ctx.seed)
        rows.append({"target": target, "indicator_grouped_score": base, "indicator_plus_latent_grouped_score": augmented, "increment": augmented - base})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "publication_grouped_incremental_probes.csv", index=False, encoding="utf-8-sig")
    increment = float(frame["increment"].mean())
    return _result(
        effect_status(increment),
        {"targets": len(frame), "mean_publication_grouped_increment": increment},
        "publication_grouped_balanced_accuracy_increment",
        increment,
        0.0,
        f"Under canonical-publication grouped folds, adding latent changes indicator-only balanced accuracy by {increment:+.3f} on average.",
        "Encoder weights still saw all publications and labels during pretraining; grouped probes therefore remain transductive decodability checks.",
        "A publishable generalization test requires fold-specific pretraining or a disjoint external corpus.",
    )


def _unavailable(reason: str, next_step: str) -> dict[str, Any]:
    return _result("Negative", {"available": False}, "available", 0.0, 1.0, reason, "Required state is absent from the trained model/data environment.", next_step, stability="deterministic availability audit")


def run_followup_experiments(ctx: ExperimentContext, registry: ExperimentRegistry, only: set[str] | None = None) -> None:
    definitions = [
        (_spec("X1_context_raw_control", "Automatic follow-up: C1 versus raw context under publication grouping", "X", "contextual token", "same-value background transmission beyond raw chemistry", "The strong C1 signal should be compared with the raw chemical context that the token attends to.", ctx, 1800, "publication-grouped full raw context"), experiment_x1),
        (_spec("X2_role_raw_control", "Automatic follow-up: contextual role proxies versus raw chemistry", "X", "contextual token", "process-proxy readability beyond raw chemistry", "C2 gains over a value-only token may disappear against full raw chemistry.", ctx, 7000, "publication-grouped full raw context"), experiment_x2),
        (_spec("X3_cross_space_relation_followup", "Automatic follow-up: volcanic–plutonic relation structure across spaces", "X", "label relations and chemistry class-centroid relations", "relation-structure agreement", "Strong G2/G4 label relations should have a corresponding structure in chemistry-only sample centroids.", ctx, 7, "relation-matrix label permutations"), experiment_x3),
        (_spec("X4_random_missingness_followup", "Automatic follow-up: random encoder and missingness/source structure", "X", "raw mask, random encoder, trained encoder", "source versus geology neighborhood structure", "The unexpectedly high random-encoder source purity should be explained by observed-panel structure and matched-sample controls.", ctx, 3000, "mask-only, raw, random, trained; seeds/sample sizes/distances"), experiment_x4),
        (_spec("D4_geographic_neighbors", "Geographic proximity in latent neighborhoods", "D", "sample latent", "geographic versus chemical neighborhood", "Latent neighbor geography should be quantified separately from geological consistency.", ctx, 3000, "raw chemistry neighborhoods"), experiment_d4),
        (_spec("D5_external_isotopes_age", "External isotope and age readability", "D", "sample latent", "isotope and age structure not used as tokens", "Chemistry-only latent may organize external isotope and age variation beyond raw chemistry.", ctx, 6000, "raw chemistry PCA and grouped ridge"), experiment_d5),
        (_spec("J2_grouped_increment", "Publication-grouped conditional incremental probes", "J", "sample latent", "increment beyond traditional indicators under grouped folds", "J1 incremental value should survive canonical-publication grouped probes.", ctx, 6000, "indicator-only grouped probes"), experiment_j2),
        (_spec("B4_detection_limit_unavailable", "Detection-limit trajectory audit", "B", "numeric token", "near-detection-limit geometry", "Detection-limit states should be tested if preserved." , ctx, 0, "availability audit"), lambda _c, _o, _s: _unavailable("The source view and V2 tokenization do not preserve detection-limit/censoring states, so near-limit geometry cannot be identified.", "Add explicit censored-value state tokens in a future model before testing detection-limit semantics.")),
        (_spec("K4_checkpoint_sensitivity_unavailable", "Checkpoint sensitivity audit", "K", "all representation levels", "stability across training checkpoints", "Representation findings should be stable across checkpoints.", ctx, 0, "availability audit"), lambda _c, _o, _s: _unavailable("Only the final full_model.pt and its same-step component state dicts are available; there are no epoch checkpoints for sensitivity analysis.", "Save periodic full checkpoints in future training runs.")),
        (_spec("K5_umap_parameter_unavailable", "UMAP parameter sensitivity audit", "K", "low-dimensional visualization", "UMAP stability", "Visual conclusions should be stable to UMAP parameters.", ctx, 0, "availability audit"), lambda _c, _o, _s: _unavailable("umap-learn is absent from the Element2Vec environment; PCA remained the primary reproducible visualization and no UMAP claim was made.", "Install a pinned UMAP environment only for auxiliary visualization, not primary evidence.")),
    ]
    for spec, function in definitions:
        if only is not None and spec.experiment_id not in only:
            continue
        print(f"[experiment] {spec.experiment_id}")
        registry.run(spec, lambda out, s, f=function: f(ctx, out, s))
