from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .analysis_utils import (
    centroid_by_label,
    cosine_matrix,
    cv_balanced_accuracy,
    effect_status,
    knn_indices,
    knn_purity,
    permutation_pvalue,
    plot_heatmap,
    row_normalize,
    safe_spearman,
)
from .catalogs import EVOLUTION_SEQUENCES, METAMORPHIC_PAIRS, VOLCANIC_PLUTONIC_PAIRS
from .experiment_context import ExperimentContext
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
        dataset="V2 trained vocabulary and/or stratified chemistry-only sample",
        input_fields="rock_group, rock_origin, rock_type pure tokens; 34 numeric chemistry tokens",
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
    figures: list[Path] | None = None,
    stability: str = "single breadth-first scan",
) -> dict[str, Any]:
    effect = None
    if value is not None and baseline is not None and np.isfinite(value) and np.isfinite(baseline):
        effect = float(value - baseline)
    return {
        "status": status,
        "quantitative_metrics": metrics,
        "figure_paths": [str(p) for p in figures or []],
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


def _nonmask_pure(ctx: ExperimentContext, projected: bool = False) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = ctx.pure_tokens
    valid = np.asarray([not name.endswith(":[MASK]") for name in names])
    if projected:
        trained = ctx.project_pure_labels(trained=True)[valid]
        random = ctx.project_pure_labels(trained=False)[valid]
    else:
        trained = ctx.static_trained["pure"][valid]
        random = ctx.static_random["pure"][valid]
    return [name for name, keep in zip(names, valid) if keep], trained, random


def experiment_g1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    names, trained, random = _nonmask_pure(ctx, projected=False)
    fields = np.asarray([name.split(":", 1)[0] for name in names])
    idx_t = knn_indices(trained, k=5)
    idx_r = knn_indices(random, k=5)
    purity_t = float(np.mean(fields[idx_t] == fields[:, None]))
    purity_r = float(np.mean(fields[idx_r] == fields[:, None]))
    rows = []
    sim = cosine_matrix(trained)
    for i, name in enumerate(names):
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i][:10]
        for rank, j in enumerate(order, start=1):
            rows.append({"query": name, "neighbor": names[j], "rank": rank, "cosine": float(sim[i, j]), "same_field": fields[i] == fields[j]})
    pd.DataFrame(rows).to_csv(out / "pure_label_neighbors.csv", index=False, encoding="utf-8-sig")
    field_effect = purity_t - purity_r
    return _result(
        effect_status(field_effect),
        {"trained_field_knn5_purity": purity_t, "random_field_knn5_purity": purity_r, "tokens": len(names)},
        "pure_label_field_knn5_purity",
        purity_t,
        purity_r,
        f"Pure-label embeddings have field-identity kNN5 purity {purity_t:.3f}, versus {purity_r:.3f} before training.",
        "Field identity is encoded directly in token names and separate field-level mask tasks, so field clustering is not geological semantic evidence by itself.",
        "Inspect within-rock_type neighbors and compare their distance structure with chemistry-only class centroids (H3).",
    )


def _relation_vectors(ctx: ExperimentContext) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names, trained, random = _nonmask_pure(ctx, projected=True)
    return dict(zip(names, trained)), dict(zip(names, random))


def experiment_g2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    trained, random = _relation_vectors(ctx)
    rows = []
    vectors_t, vectors_r, available_pairs = [], [], []
    for volcanic, plutonic in VOLCANIC_PLUTONIC_PAIRS:
        vt, pt = f"rock_type:{volcanic}", f"rock_type:{plutonic}"
        available = vt in trained and pt in trained
        rows.append({"volcanic": volcanic, "plutonic": plutonic, "available": available})
        if available:
            vectors_t.append(trained[pt] - trained[vt])
            vectors_r.append(random[pt] - random[vt])
            available_pairs.append((vt, pt))
    if len(vectors_t) < 3:
        pd.DataFrame(rows).to_csv(out / "volcanic_plutonic_pairs.csv", index=False, encoding="utf-8-sig")
        return _result("Negative", {"available_pairs": len(vectors_t)}, "available_pairs", float(len(vectors_t)), 3.0, "Too few volcanic–plutonic label pairs are available.", "Vocabulary coverage limitation.", "Extend or normalize rock_type labels before analogy analysis.")
    vectors_t = np.stack(vectors_t)
    vectors_r = np.stack(vectors_r)
    sim_t = cosine_matrix(vectors_t)
    sim_r = cosine_matrix(vectors_r)
    tri = np.triu_indices(len(vectors_t), k=1)
    cohesion_t = float(sim_t[tri].mean())
    cohesion_r = float(sim_r[tri].mean())
    # Leave-one-pair-out translation retrieval among rock_type labels.
    rock_names = [name for name in trained if name.startswith("rock_type:")]
    rock_matrix = np.stack([trained[name] for name in rock_names])
    retrieval = []
    for held, (v_name, p_name) in enumerate(available_pairs):
        direction = np.mean(np.delete(vectors_t, held, axis=0), axis=0)
        query = trained[v_name] + direction
        similarities = cosine_matrix(query[None, :], rock_matrix)[0]
        order = np.argsort(-similarities)
        true_idx = rock_names.index(p_name)
        rank = int(np.flatnonzero(order == true_idx)[0]) + 1
        retrieval.append({"volcanic": v_name, "target_plutonic": p_name, "rank": rank, "reciprocal_rank": 1.0 / rank})
    retrieval_frame = pd.DataFrame(retrieval)
    retrieval_frame.to_csv(out / "volcanic_plutonic_analogy_retrieval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out / "volcanic_plutonic_pairs.csv", index=False, encoding="utf-8-sig")
    mrr = float(retrieval_frame["reciprocal_rank"].mean())
    chance_mrr = float(np.mean(1.0 / np.arange(1, len(rock_names) + 1)))
    combined = 0.5 * (cohesion_t - cohesion_r) + 0.5 * (mrr - chance_mrr)
    return _result(
        effect_status(combined),
        {"available_pairs": len(vectors_t), "trained_relation_mean_cosine": cohesion_t, "random_relation_mean_cosine": cohesion_r, "leave_one_pair_out_mrr": mrr, "chance_rank_mrr": chance_mrr, "combined_effect": combined},
        "combined_relation_cohesion_and_mrr",
        combined,
        0.0,
        f"Across {len(vectors_t)} available pairs, the trained volcanic→plutonic relation has mean pairwise cosine {cohesion_t:.3f} and leave-one-pair-out MRR {mrr:.3f}.",
        "Rock-type labels are partly computed from chemistry and share lexical/classification structure; a global translation may obscure local nonlinear relations.",
        "Repeat any positive relation in chemistry-only class-centroid space and test whether relation structure agrees across spaces (H3).",
    )


def experiment_g3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    trained, _ = _relation_vectors(ctx)
    rows = []
    for protolith, metamorphic in METAMORPHIC_PAIRS:
        left = f"rock_type:{protolith}"
        right = f"rock_type:{metamorphic}"
        rows.append({"protolith": protolith, "metamorphic": metamorphic, "left_available": left in trained, "right_available": right in trained, "pair_available": left in trained and right in trained})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "metamorphic_pair_coverage.csv", index=False, encoding="utf-8-sig")
    available = int(frame["pair_available"].sum())
    return _result(
        "Negative",
        {"prespecified_pairs": len(frame), "available_pairs": available},
        "available_metamorphic_pairs",
        float(available),
        3.0,
        f"Only {available}/{len(frame)} prespecified protolith–metamorphic rock_type pairs are present; the relation algebra is not testable in the trained pure-label vocabulary.",
        "The database rock_type is a chemistry-derived classification and does not preserve metamorphic grade/name labels such as slate, phyllite, schist, or marble.",
        "Use rock_group/protolith metadata in a separately trained vocabulary if metamorphic relation semantics are a research target.",
    )


def experiment_g4(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    trained, random = _relation_vectors(ctx)
    rows = []
    for sequence_id, sequence in enumerate(EVOLUTION_SEQUENCES, start=1):
        names = [f"rock_type:{name}" for name in sequence]
        available = [name for name in names if name in trained]
        if len(available) < 4:
            rows.append({"sequence": sequence_id, "available": len(available), "trained_monotonicity": np.nan, "random_monotonicity": np.nan})
            continue
        x_t = np.stack([trained[name] for name in available])
        x_r = np.stack([random[name] for name in available])
        pc_t = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(x_t))[:, 0]
        pc_r = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(x_r))[:, 0]
        order = np.arange(len(available))
        rows.append({"sequence": sequence_id, "labels": ";".join(available), "available": len(available), "trained_monotonicity": abs(safe_spearman(pc_t, order)[0]), "random_monotonicity": abs(safe_spearman(pc_r, order)[0])})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "rock_evolution_sequences.csv", index=False, encoding="utf-8-sig")
    trained_mean = float(frame["trained_monotonicity"].mean())
    random_mean = float(frame["random_monotonicity"].mean())
    return _result(
        effect_status(trained_mean - random_mean),
        {"sequences": len(frame), "trained_mean_monotonicity": trained_mean, "random_mean_monotonicity": random_mean},
        "mean_sequence_pc1_monotonicity",
        trained_mean,
        random_mean,
        f"The two prespecified volcanic and plutonic evolution sequences have mean trained PC1 monotonicity {trained_mean:.3f}, compared with {random_mean:.3f} at initialization.",
        "With only five labels per sequence, PCA ordering is unstable and does not prove a globally linear magmatic-evolution axis.",
        "Compare sequence ordering with chemistry-only sample centroids and natural-suite trajectories.",
    )


def experiment_g5(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    trained, random = _relation_vectors(ctx)
    pairs = [
        ("subalkalic basalt", "alkalic basalt"),
        ("subalkalic gabbro", "alkalic gabbro"),
        ("subalkalic basalt", "trachybasalt"),
    ]
    vectors_t, vectors_r, rows = [], [], []
    for sub, alk in pairs:
        a, b = f"rock_type:{sub}", f"rock_type:{alk}"
        available = a in trained and b in trained
        rows.append({"subalkaline": sub, "alkaline": alk, "available": available})
        if available:
            vectors_t.append(trained[b] - trained[a])
            vectors_r.append(random[b] - random[a])
    pd.DataFrame(rows).to_csv(out / "alkaline_subalkaline_pairs.csv", index=False, encoding="utf-8-sig")
    if len(vectors_t) < 2:
        return _result("Negative", {"available_pairs": len(vectors_t)}, "available_pairs", float(len(vectors_t)), 2.0, "Insufficient alkaline/subalkaline pairs.", "Vocabulary coverage.", "Add normalized series labels.")
    tri = np.triu_indices(len(vectors_t), 1)
    cohesion_t = float(cosine_matrix(np.stack(vectors_t))[tri].mean())
    cohesion_r = float(cosine_matrix(np.stack(vectors_r))[tri].mean())
    return _result(
        effect_status(cohesion_t - cohesion_r),
        {"available_pairs": len(vectors_t), "trained_relation_cosine": cohesion_t, "random_relation_cosine": cohesion_r},
        "alkaline_relation_cosine",
        cohesion_t,
        cohesion_r,
        f"Available alkaline-shift relation vectors have mean cosine {cohesion_t:.3f} versus {cohesion_r:.3f} at random initialization.",
        "The three label pairs do not represent identical petrogenetic comparisons and one is not a strict paired equivalent.",
        "Treat as exploratory; validate against total-alkali-matched chemistry-only class centroids.",
    )


def _rock_type_label_vectors(ctx: ExperimentContext, shared_sample_basis: bool = False) -> tuple[list[str], np.ndarray]:
    names = ctx.pure_tokens
    valid = np.asarray([not name.endswith(":[MASK]") for name in names])
    if shared_sample_basis and hasattr(ctx, "project_pure_labels_for_sample_space"):
        vectors = ctx.project_pure_labels_for_sample_space(trained=True)[valid]
    else:
        vectors = ctx.project_pure_labels(trained=True)[valid]
    names = [name for name, keep in zip(names, valid) if keep]
    keep = [i for i, name in enumerate(names) if name.startswith("rock_type:")]
    return [names[i].split(":", 1)[1] for i in keep], vectors[keep]


def experiment_h1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    label_names, label_vectors = _rock_type_label_vectors(ctx, shared_sample_basis=True)
    latents = ctx.latents("numeric_only")["encoded_mean_numeric"]
    meta = ctx.meta
    valid = meta["rock_type"].isin(label_names).to_numpy()
    indices = np.flatnonzero(valid)[:3000]
    sim = cosine_matrix(latents[indices], label_vectors)
    rows = []
    for local_i, sample_i in enumerate(indices):
        target = str(meta.iloc[sample_i]["rock_type"])
        target_idx = label_names.index(target)
        order = np.argsort(-sim[local_i])
        rank = int(np.flatnonzero(order == target_idx)[0]) + 1
        rows.append({"sample_id": int(meta.iloc[sample_i]["sample_id"]), "rock_type": target, "rank": rank, "top1": rank == 1, "top5": rank <= 5, "reciprocal_rank": 1.0 / rank})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "direct_sample_label_retrieval.csv", index=False, encoding="utf-8-sig")
    top1, top5, mrr = float(frame["top1"].mean()), float(frame["top5"].mean()), float(frame["reciprocal_rank"].mean())
    chance_top1 = 1.0 / len(label_names)
    status = effect_status(top1 - chance_top1)
    return _result(
        status,
        {"n": len(frame), "classes": len(label_names), "top1": top1, "top5": top5, "mrr": mrr, "chance_top1": chance_top1},
        "direct_sample_to_label_top1",
        top1,
        chance_top1,
        f"Without an alignment map, direct cosine retrieval places the true rock_type first for {top1:.3%} of chemistry-only samples and in the top five for {top5:.3%}.",
        "Projected pure-label tokens and contextual pooled sample latents are different representation levels; direct cosine is a diagnostic and may be geometrically invalid.",
        "Use the constrained class-centroid alignment experiment H2 before interpreting cross-space similarity.",
    )


def _class_centroids(ctx: ExperimentContext, min_count: int = 20) -> tuple[list[str], np.ndarray, np.ndarray]:
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"]
    centroids, names, counts = centroid_by_label(latent, ctx.meta["rock_type"], min_count=min_count)
    return names, centroids, counts


def experiment_h2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    label_names, label_vectors = _rock_type_label_vectors(ctx, shared_sample_basis=True)
    class_names, centroids, counts = _class_centroids(ctx, min_count=20)
    common = [name for name in class_names if name in label_names]
    x = np.stack([label_vectors[label_names.index(name)] for name in common])
    y = np.stack([centroids[class_names.index(name)] for name in common])
    n_components = min(24, len(common) - 2, x.shape[1], y.shape[1])
    x_coords = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(x))
    y_pca = PCA(n_components=n_components)
    y_coords = y_pca.fit_transform(StandardScaler().fit_transform(y))
    predicted = np.zeros_like(y_coords)
    for held in range(len(common)):
        train = np.arange(len(common)) != held
        model = Ridge(alpha=10.0).fit(x_coords[train], y_coords[train])
        predicted[held] = model.predict(x_coords[held : held + 1])[0]
    sim = cosine_matrix(predicted, y_coords)
    ranks = []
    for i, name in enumerate(common):
        order = np.argsort(-sim[i])
        rank = int(np.flatnonzero(order == i)[0]) + 1
        ranks.append({"rock_type": name, "rank": rank, "top1": rank == 1, "top5": rank <= 5, "reciprocal_rank": 1.0 / rank, "class_count": int(counts[class_names.index(name)])})
    frame = pd.DataFrame(ranks)
    frame.to_csv(out / "leave_one_class_out_alignment.csv", index=False, encoding="utf-8-sig")
    top1 = float(frame["top1"].mean())
    mrr = float(frame["reciprocal_rank"].mean())
    chance = 1.0 / len(common)
    return _result(
        effect_status(top1 - chance),
        {"classes": len(common), "leave_one_class_out_top1": top1, "top5": float(frame["top5"].mean()), "mrr": mrr, "chance_top1": chance},
        "leave_one_class_out_alignment_top1",
        top1,
        chance,
        f"A 24D ridge map trained on all other classes retrieves the held-out class centroid at top-1 rate {top1:.3f} across {len(common)} rock types.",
        "Class centroids and label embeddings both inherit training labels; leave-one-class-out limits mapping capacity but is not an independent test set.",
        "Compare relation-distance preservation without fitted mapping (H3) and rerun on publication-held-out class centroids if pursued.",
    )


def experiment_h3(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    label_names, label_vectors = _rock_type_label_vectors(ctx, shared_sample_basis=True)
    class_names, centroids, _ = _class_centroids(ctx, min_count=20)
    common = [name for name in class_names if name in label_names]
    labels = np.stack([label_vectors[label_names.index(name)] for name in common])
    samples = np.stack([centroids[class_names.index(name)] for name in common])
    label_dist = pdist(row_normalize(labels), metric="cosine")
    sample_dist = pdist(row_normalize(samples), metric="cosine")
    rho, p = safe_spearman(label_dist, sample_dist)
    rng = np.random.default_rng(ctx.seed)
    null = []
    for _ in range(int(ctx.config["analysis"]["permutation_repeats"])):
        perm = rng.permutation(len(common))
        null.append(safe_spearman(pdist(row_normalize(labels[perm]), metric="cosine"), sample_dist)[0])
    perm_p = permutation_pvalue(rho, np.asarray(null))
    matrix = pd.DataFrame(cosine_matrix(labels), index=common, columns=common)
    fig = out / "figures" / "rock_type_label_cosine.png"
    plot_heatmap(matrix, fig, "Projected rock_type label similarities", dpi=ctx.dpi, cmap="viridis")
    return _result(
        effect_status(rho, perm_p),
        {"classes": len(common), "cross_space_distance_spearman": rho, "spearman_p": p, "permutation_p": perm_p},
        "cross_space_relation_distance_spearman",
        rho,
        0.0,
        f"Pairwise rock-type relation distances agree between projected label space and chemistry-only class-centroid space at Spearman ρ={rho:.3f} (permutation p={perm_p:.3g}).",
        "Both spaces were produced by one label-supervised full-data checkpoint; relation agreement may reflect the supervised objective rather than emergent unsupervised geology.",
        "Inspect which class pairs drive agreement and test against raw-chemistry centroid distances.",
        [fig],
    )


def experiment_j1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    rng = np.random.default_rng(ctx.seed)
    n = min(6000, len(ctx.meta))
    idx = np.sort(rng.choice(len(ctx.meta), size=n, replace=False))
    meta = ctx.meta.iloc[idx].reset_index(drop=True)
    q_cols = ["sio2", "mg_number", "total_alkali", "cia", "log_sr_y", "log_la_yb", "log_th_sc", "log_nb_u"]
    q = meta[q_cols].copy()
    for col in q_cols:
        q[col] = q[col].fillna(q[col].median())
    qx = q.to_numpy(dtype=float)
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"][idx]
    probe_components = min(24, latent.shape[1], len(latent) - 1)
    z = PCA(n_components=probe_components, random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(latent))
    combined = np.concatenate([qx, z], axis=1)
    targets = {
        "rock_group": meta["rock_group"].fillna("[MISSING]").astype(str).to_numpy(),
        "rock_origin": meta["rock_origin"].fillna("[MISSING]").astype(str).to_numpy(),
    }
    top_types = meta["rock_type"].value_counts().head(15).index
    type_mask = meta["rock_type"].isin(top_types).to_numpy()
    rows = []
    for target, labels in targets.items():
        base = cv_balanced_accuracy(qx, labels, folds=5, seed=ctx.seed)
        augmented = cv_balanced_accuracy(combined, labels, folds=5, seed=ctx.seed)
        rows.append({"target": target, "n": n, "baseline_balanced_accuracy": base, "q_plus_latent_balanced_accuracy": augmented, "increment": augmented - base})
    base = cv_balanced_accuracy(qx[type_mask], meta.loc[type_mask, "rock_type"].astype(str).to_numpy(), folds=5, seed=ctx.seed)
    augmented = cv_balanced_accuracy(combined[type_mask], meta.loc[type_mask, "rock_type"].astype(str).to_numpy(), folds=5, seed=ctx.seed)
    rows.append({"target": "rock_type_top15", "n": int(type_mask.sum()), "baseline_balanced_accuracy": base, "q_plus_latent_balanced_accuracy": augmented, "increment": augmented - base})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "conditional_incremental_probes.csv", index=False, encoding="utf-8-sig")
    increment = float(frame["increment"].mean())
    return _result(
        effect_status(increment),
        {"targets": len(frame), "mean_increment": increment, "max_increment": float(frame["increment"].max()), "sample_size": n, "conditioning_variables": q_cols},
        "mean_balanced_accuracy_increment_over_traditional_indicators",
        increment,
        0.0,
        f"Adding a {probe_components}D chemistry-only latent to eight traditional indicators changes balanced accuracy by {increment:+.3f} on average across rock_group, rock_origin, and top-15 rock_type.",
        "The checkpoint was trained with these labels on the same full dataset; this measures conditional decodability and cannot establish independent predictive value.",
        "Repeat only promising targets with an encoder trained outside each publication-held-out evaluation fold.",
    )


def experiment_k1(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    n = min(3000, len(ctx.meta))
    # The random model cache was deliberately extracted for the first n rows.
    # Use exactly those same samples for every representation in this control.
    idx = np.arange(n)
    meta = ctx.meta.iloc[idx].reset_index(drop=True)
    numeric = ctx.latents("numeric_only")
    masked = ctx.latents("masked_labels")
    random_latent = ctx.latents("random_numeric_only")
    random_idx = np.arange(min(n, len(random_latent["cls"])))
    # Random controls were extracted from the first rows, so evaluate their matching metadata separately.
    spaces = {
        "raw_log_robust_plus_mask": (ctx.raw_baseline()[idx], meta),
        "major_clr": (ctx.major_clr()[idx], meta),
        "projected_mean_numeric": (numeric["projected_mean_numeric"][idx], meta),
        "encoded_mean_numeric": (numeric["encoded_mean_numeric"][idx], meta),
        "cls_numeric_only": (numeric["cls"][idx], meta),
        "encoded_masked_labels": (masked["encoded_mean_numeric"][idx], meta),
        "cls_masked_labels": (masked["cls"][idx], meta),
        "random_encoded_numeric_only": (random_latent["encoded_mean_numeric"][random_idx], ctx.meta.iloc[random_idx].reset_index(drop=True)),
        "random_cls_numeric_only": (random_latent["cls"][random_idx], ctx.meta.iloc[random_idx].reset_index(drop=True)),
    }
    raw_pca = PCA(n_components=12, random_state=ctx.seed).fit_transform(StandardScaler().fit_transform(spaces["raw_log_robust_plus_mask"][0]))
    spaces["raw_pca12"] = (raw_pca, meta)
    rows = []
    for name, (x, m) in spaces.items():
        rows.append(
            {
                "space": name,
                "n": len(x),
                "rock_group_knn15": knn_purity(x, m["rock_group"].fillna("[MISSING]").astype(str).to_numpy(), 15),
                "rock_type_knn15": knn_purity(x, m["rock_type"].fillna("[MISSING]").astype(str).to_numpy(), 15),
                "source_knn15": knn_purity(x, m["data_source"].fillna("[MISSING]").astype(str).to_numpy(), 15),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "representation_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    encoded = float(frame.loc[frame["space"] == "encoded_mean_numeric", "rock_type_knn15"].iloc[0])
    traditional = float(frame.loc[frame["space"].isin(["raw_log_robust_plus_mask", "major_clr", "raw_pca12"]), "rock_type_knn15"].max())
    random_score = float(frame.loc[frame["space"] == "random_encoded_numeric_only", "rock_type_knn15"].iloc[0])
    return _result(
        effect_status(encoded - traditional),
        {"encoded_rock_type_knn15": encoded, "best_traditional_rock_type_knn15": traditional, "random_encoded_rock_type_knn15": random_score, "spaces": len(frame)},
        "rock_type_knn15_against_best_traditional",
        encoded,
        traditional,
        f"Encoded mean has rock-type kNN15 purity {encoded:.3f}; the best raw/CLR/PCA traditional baseline reaches {traditional:.3f}, and the random contextual encoder reaches {random_score:.3f}.",
        "Rock-type purity is label-contaminated by full-data multi-task training and can also be inflated by source/missingness structure.",
        "Use this table as the common baseline for all promoted results; do not cite model metrics without the source-purity column.",
    )


def experiment_k2(ctx: ExperimentContext, out: Path, _specification: ExperimentSpec) -> dict[str, Any]:
    numeric = ctx.latents("numeric_only")
    masked = ctx.latents("masked_labels")
    visible = ctx.latents("visible_labels")
    rows = []
    for pooling in ["projected_mean_numeric", "encoded_mean_numeric", "cls"]:
        a = row_normalize(numeric[pooling])
        b = row_normalize(masked[pooling])
        masked_cos = np.sum(a * b, axis=1)
        visible_n = len(visible[pooling])
        v = row_normalize(visible[pooling])
        visible_cos = np.sum(a[:visible_n] * v, axis=1)
        rows.append(
            {
                "pooling": pooling,
                "numeric_vs_masked_mean_cosine": float(masked_cos.mean()),
                "numeric_vs_masked_p05": float(np.quantile(masked_cos, 0.05)),
                "numeric_vs_visible_mean_cosine": float(visible_cos.mean()),
                "numeric_vs_visible_p05": float(np.quantile(visible_cos, 0.05)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "input_mode_stability.csv", index=False, encoding="utf-8-sig")
    encoded_row = frame.loc[frame["pooling"] == "encoded_mean_numeric"].iloc[0]
    masked_stability = float(encoded_row["numeric_vs_masked_mean_cosine"])
    visible_stability = float(encoded_row["numeric_vs_visible_mean_cosine"])
    status = "Strong positive" if masked_stability >= 0.95 else "Weak/ambiguous" if masked_stability >= 0.8 else "Negative"
    return _result(
        status,
        {"encoded_numeric_vs_masked_cosine": masked_stability, "encoded_numeric_vs_visible_cosine": visible_stability, "poolings": len(frame)},
        "numeric_only_vs_masked_label_cosine",
        masked_stability,
        1.0,
        f"Encoded-mean latents from numeric-only and all-label-masked inference have mean cosine {masked_stability:.3f}; numeric-only versus visible-label control is {visible_stability:.3f}.",
        "All modes use a model trained mostly with visible labels. Numeric-only and all-masked inputs are both distribution shifts, so agreement is reassuring but disagreement is not uniquely attributable.",
        "Combine with explicit token-dropout and order-invariance tests in K3.",
    )


def run_relation_experiments(ctx: ExperimentContext, registry: ExperimentRegistry, only: set[str] | None = None) -> None:
    definitions = [
        (_spec("G1_label_neighbors", "Pure-label neighborhoods", "G", "static pure-label embedding", "field and rock-label semantic neighborhoods", "Training should organize pure labels beyond field identity and random initialization.", ctx, 93, "same-seed random label embeddings"), experiment_g1),
        (_spec("G2_volcanic_plutonic", "Volcanic–plutonic relation vectors", "G", "MLP-projected pure-label token", "paired volcanic/plutonic analogies", "Volcanic→plutonic pairs may share a stable local relation direction.", ctx, len(VOLCANIC_PLUTONIC_PAIRS), "random relation vectors and chance retrieval"), experiment_g2),
        (_spec("G3_protolith_metamorphic", "Protolith–metamorphic relation coverage", "G", "pure-label vocabulary", "shared metamorphic transformation directions", "Metamorphic products and protoliths should support relation-vector tests if labels exist.", ctx, len(METAMORPHIC_PAIRS), "coverage audit"), experiment_g3),
        (_spec("G4_magmatic_sequences", "Magmatic evolution label sequences", "G", "MLP-projected pure-label token", "basalt→andesite→dacite→rhyolite and plutonic equivalent", "Rock labels may form continuous evolution sequences rather than isolated classes.", ctx, 10, "same-seed random label embeddings"), experiment_g4),
        (_spec("G5_other_geological_relations", "Alkaline–subalkaline relation directions", "G", "MLP-projected pure-label token", "alkaline series shift", "Multiple alkaline/subalkaline pairs may share a relation direction.", ctx, 3, "same-seed random label embeddings"), experiment_g5),
        (_spec("H1_sample_label_retrieval", "Direct sample-to-rock-label retrieval", "H", "chemistry-only sample latent versus projected label token", "true rock_type rank", "If spaces are naturally aligned, direct cosine should retrieve the true rock label above chance.", ctx, 3000, "chance rank"), experiment_h1),
        (_spec("H2_centroid_label_alignment", "Low-capacity label-to-centroid alignment", "H", "label and sample-centroid spaces", "held-out rock_type alignment", "A constrained linear map may align label tokens with chemistry-only class centroids.", ctx, 81, "leave-one-class-out and chance rank"), experiment_h2),
        (_spec("H3_cross_space_relations", "Cross-space relation preservation", "H", "label and sample-centroid distance structures", "rock_type pairwise relations", "Label-space pairwise relations should agree with chemistry-only sample-centroid relations without vector subtraction across spaces.", ctx, 81, "label permutations"), experiment_h3),
        (_spec("J1_beyond_traditional_indicators", "Conditional value beyond traditional indicators", "J", "chemistry-only encoded sample latent", "incremental rock-label readability", "Latent should add low-capacity information after conditioning on traditional geochemical indicators.", ctx, 6000, "indicator-only regularized probes"), experiment_j1),
        (_spec("K1_representation_baselines", "Traditional, random, projected, contextual, and pooling baselines", "K", "all major representation levels", "rock and source neighborhood structure", "Trained contextual latent should outperform traditional and random controls without uniquely amplifying source structure.", ctx, 3000, "raw/log+mask/CLR/PCA/random/input-only/label modes"), experiment_k1),
        (_spec("K2_input_mode_stability", "Label-input mode and pooling stability", "K", "sample pooled latent", "numeric-only, masked-label, visible-label stability", "Main chemistry-only latents should remain close to masked-label inference and diverge predictably from visible-label leakage control.", ctx, 12000, "masked and visible pure-label modes"), experiment_k2),
    ]
    for spec, function in definitions:
        if only is not None and spec.experiment_id not in only:
            continue
        print(f"[experiment] {spec.experiment_id}")
        registry.run(spec, lambda out, s, f=function: f(ctx, out, s))
