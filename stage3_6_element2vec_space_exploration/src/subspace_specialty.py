from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .catalogs import VOLCANIC_PLUTONIC_PAIRS
from .common import load_yaml, resolve_path, set_reproducible_seed
from .experiment_context import ExperimentContext
from .experiments_process import _representative_backgrounds
from .subspace_projection import ProjectionManager, ProjectionSpec


def _pool_indices(meta: pd.DataFrame, path: Path) -> tuple[np.ndarray, np.ndarray]:
    mapping = pd.Series(np.arange(len(meta)), index=meta["sample_id"]).to_dict()
    fit = pd.read_csv(path / "sample_fit_pool.csv")["sample_id"]
    evaluate = pd.read_csv(path / "sample_evaluation_pool.csv")["sample_id"]
    return np.asarray([mapping[x] for x in fit]), np.asarray([mapping[x] for x in evaluate])


def _baseline(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(np.sum(p * p))


def _knn_indicators(x: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    valid = pd.notna(labels)
    x, labels = x[valid], np.asarray(labels[valid]).astype(str)
    if len(x) <= k + 1:
        return np.asarray([]), np.nan
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric="cosine").fit(x)
    idx = nn.kneighbors(return_distance=False)[:, 1:]
    indicators = (labels[idx] == labels[:, None]).mean(axis=1)
    return indicators, _baseline(labels)


def _bootstrap_ci(values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(repeats)]
    return tuple(np.quantile(means, [0.025, 0.975]))


def _eta_squared(values: np.ndarray, labels: pd.Series, min_group: int = 10) -> float:
    valid = np.isfinite(values) & labels.notna().to_numpy()
    if valid.sum() < 30:
        return np.nan
    y = labels.iloc[np.flatnonzero(valid)].astype(str)
    counts = y.value_counts()
    y = y.where(y.isin(counts[counts >= min_group].index), "__rare__")
    x = values[valid]
    total = np.sum((x - x.mean()) ** 2)
    if total <= 0:
        return np.nan
    between = sum(len(ix) * (x[ix].mean() - x.mean()) ** 2 for _, ix in y.groupby(y).indices.items())
    return float(between / total)


def _grouped_probe(x: np.ndarray, labels: pd.Series, groups: pd.Series, seed: int) -> float:
    valid = labels.notna() & groups.notna()
    x = x[valid.to_numpy()]
    y = labels[valid].astype(str).to_numpy()
    g = groups[valid].astype(str).to_numpy()
    counts = pd.Series(y).value_counts()
    keep = np.isin(y, counts[counts >= 30].index)
    x, y, g = x[keep], y[keep], g[keep]
    if len(np.unique(y)) < 2 or len(np.unique(g)) < 3:
        return np.nan
    scores = []
    for train, test in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(x, y, g):
        if len(np.unique(y[train])) != len(np.unique(y)):
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed))
        model.fit(x[train], y[train])
        scores.append(balanced_accuracy_score(y[test], model.predict(x[test])))
    return float(np.mean(scores)) if scores else np.nan


def _specs(settings: dict[str, Any]) -> list[ProjectionSpec]:
    specs = [ProjectionSpec("FULL", "FULL")]
    specs += [ProjectionSpec(f"PCA{d}", "PCA", d) for d in settings["pca_dimensions"]]
    specs += [ProjectionSpec(f"PCA{d}_WHITEN", "PCA", d, whiten=True) for d in settings["whitening_dimensions"]]
    specs += [ProjectionSpec(f"DROP_PC{d}", "DROP_PC", drop_count=d) for d in settings["drop_pc_counts"]]
    specs += [
        ProjectionSpec(f"RP{d}_seed{s:02d}", "RP", d, seed=s)
        for d in settings["random_projection_dimensions"] for s in settings["random_projection_seeds"]
    ]
    return specs


def _sample_signal_curves(
    ctx: ExperimentContext, manager: ProjectionManager, fit_idx: np.ndarray, eval_idx: np.ndarray,
    settings: dict[str, Any], out: Path,
) -> pd.DataFrame:
    latent = ctx.latents("numeric_only")
    representations = {
        "contextual_mean": ("sample_contextual_mean_numeric", latent["encoded_mean_numeric"]),
        "projected_mean": ("sample_projected_mean_numeric", latent["projected_mean_numeric"]),
        "cls": ("sample_cls", latent["cls"]),
    }
    meta = ctx.meta.iloc[eval_idx].reset_index(drop=True).copy()
    meta["missingness_quartile"] = pd.qcut(meta["missing_fraction"], q=4, duplicates="drop").astype(str)
    label_cols = {
        "rock_type": "geology", "rock_group": "geology", "rock_origin": "geology",
        "data_source": "source", "canonical_ref_id": "source", "missingness_quartile": "missingness",
    }
    rows = []
    for rep_name, (basis_name, full) in representations.items():
        reference = full[fit_idx]
        rep_specs = _specs(settings)
        if rep_name != "contextual_mean":
            rep_specs = [spec for spec in rep_specs if spec.method != "RP"]
        for spec in rep_specs:
            try:
                x = manager.transform(basis_name, reference, full[eval_idx], spec)
            except ValueError:
                continue
            nn = NearestNeighbors(n_neighbors=min(int(settings["neighbor_k"]) + 1, len(x)), metric="cosine").fit(x)
            neighbor_idx = nn.kneighbors(return_distance=False)[:, 1:]
            metrics = {}
            for label, signal in label_cols.items():
                raw_labels = meta[label]
                valid = raw_labels.notna().to_numpy()
                labels = raw_labels.fillna("__missing_target__").astype(str).to_numpy()
                indicators = (labels[neighbor_idx] == labels[:, None]).mean(axis=1)[valid]
                baseline = _baseline(labels[valid]) if valid.any() else np.nan
                low, high = _bootstrap_ci(indicators, int(settings["bootstrap_repeats"]), int(settings["seed"]) + len(rows))
                metrics[label] = float(np.mean(indicators)) if len(indicators) else np.nan
                rows.append({
                    "experiment_id": f"S4_{rep_name}_{label}", "family": "S", "representation_level": rep_name,
                    "projection_name": spec.name, "projection_method": spec.method,
                    "target_dimension": spec.target_dimension if spec.target_dimension else full.shape[1],
                    "random_seed": spec.seed, "signal_type": signal, "label": label,
                    "metric": "knn15_purity", "metric_value": metrics[label], "baseline": baseline,
                    "metric_lift": metrics[label] - baseline, "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                })
            geology = np.nanmean([metrics.get("rock_type"), metrics.get("rock_group"), metrics.get("rock_origin")])
            source = np.nanmean([metrics.get("data_source"), metrics.get("canonical_ref_id")])
            missing = metrics.get("missingness_quartile", np.nan)
            rows.append({
                "experiment_id": f"S4_{rep_name}_signal_ratio", "family": "S", "representation_level": rep_name,
                "projection_name": spec.name, "projection_method": spec.method,
                "target_dimension": spec.target_dimension if spec.target_dimension else full.shape[1],
                "random_seed": spec.seed, "signal_type": "ratio", "label": "geology_vs_confounding",
                "metric": "geology_minus_source", "metric_value": geology - source,
                "baseline": 0.0, "metric_lift": geology - source,
                "geology_metric": geology, "source_metric": source, "missingness_metric": missing,
                "bootstrap_ci_low": np.nan, "bootstrap_ci_high": np.nan,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "sample_signal_dimension_curves.csv", index=False)

    # Grouped probes use the fixed evaluation pool and publication/reference groups.
    probe_rows = []
    basis_name, full = representations["contextual_mean"]
    for spec in [s for s in _specs(settings) if s.method in {"FULL", "PCA"} and not s.whiten]:
        try:
            x = manager.transform(basis_name, full[fit_idx], full[eval_idx], spec)
        except ValueError:
            continue
        for label in ("rock_group", "rock_origin", "rock_type"):
            score = _grouped_probe(x, meta[label], meta["canonical_ref_id"], spec.seed)
            probe_rows.append({"representation_level": "contextual_mean", "projection_name": spec.name,
                               "projection_method": spec.method, "target": label,
                               "grouping": "canonical_ref_id", "balanced_accuracy": score, "seed": spec.seed})
    pd.DataFrame(probe_rows).to_csv(out / "grouped_probe_dimension_curves.csv", index=False)
    return frame


def _pc_attribution(ctx: ExperimentContext, manager: ProjectionManager, fit_idx: np.ndarray, eval_idx: np.ndarray, out: Path) -> pd.DataFrame:
    lat = ctx.latents("numeric_only")
    reps = {
        "contextual_mean": ("sample_contextual_mean_numeric", lat["encoded_mean_numeric"]),
        "projected_mean": ("sample_projected_mean_numeric", lat["projected_mean_numeric"]),
        "cls": ("sample_cls", lat["cls"]),
    }
    meta = ctx.meta.iloc[eval_idx].reset_index(drop=True)
    continuous = {
        "missing_fraction": "confounder", "num_numeric_tokens": "confounder",
        "sio2": "geology", "mg_number": "geology", "total_alkali": "geology",
        "cia": "geology", "log_la_yb": "geology", "latitude": "geography", "longitude": "geography",
    }
    categorical = {
        "data_source": "confounder", "canonical_ref_id": "confounder", "author": "confounder",
        "rock_group": "geology", "rock_origin": "geology", "rock_type": "geology",
    }
    rows = []
    for rep, (basis_name, full) in reps.items():
        model = manager.fit_pca(basis_name, full[fit_idx], max_components=64)
        coords = manager.pca_transform(basis_name, full[fit_idx], full[eval_idx], 64)
        evr = np.asarray(model["explained_variance_ratio"])
        for pc in range(64):
            for variable, kind in continuous.items():
                y = pd.to_numeric(meta[variable], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(y) & np.isfinite(coords[:, pc])
                value = abs(float(spearmanr(coords[valid, pc], y[valid]).statistic)) if valid.sum() >= 20 else np.nan
                rows.append({"representation_level": rep, "pc": pc + 1, "explained_variance_ratio": evr[pc],
                             "variable": variable, "variable_kind": kind, "association_method": "abs_spearman", "association": value})
            for variable, kind in categorical.items():
                rows.append({"representation_level": rep, "pc": pc + 1, "explained_variance_ratio": evr[pc],
                             "variable": variable, "variable_kind": kind, "association_method": "eta_squared",
                             "association": _eta_squared(coords[:, pc], meta[variable])})
    frame = pd.DataFrame(rows)
    scores = frame.groupby(["representation_level", "pc", "variable_kind"])["association"].max().unstack(fill_value=np.nan)
    classes = []
    for idx, row in scores.iterrows():
        g = row.get("geology", np.nan)
        c = row.get("confounder", np.nan)
        if g >= 0.15 and c >= 0.15:
            label = "mixed"
        elif c >= max(0.10, g * 1.25 if np.isfinite(g) else 0.10):
            label = "source-dominated"
        elif g >= max(0.10, c * 1.25 if np.isfinite(c) else 0.10):
            label = "geology-dominated"
        else:
            label = "unexplained"
        classes.append((*idx, label, g, c))
    class_df = pd.DataFrame(classes, columns=["representation_level", "pc", "pc_class", "geology_score", "confounder_score"])
    frame = frame.merge(class_df, on=["representation_level", "pc"], how="left")
    frame.to_csv(out / "pc_semantic_attribution.csv", index=False)
    class_df.to_csv(out / "pc_classification.csv", index=False)
    return frame


def _direction_energy(ctx: ExperimentContext, manager: ProjectionManager, fit_idx: np.ndarray, out: Path) -> pd.DataFrame:
    basis_name = "sample_contextual_mean_numeric"
    latent = ctx.latents("numeric_only")["encoded_mean_numeric"]
    model = manager.fit_pca(basis_name, latent[fit_idx], max_components=64)
    components = np.asarray(model["components"])
    directions: list[tuple[str, str, np.ndarray]] = []

    # Reproduce F2's fixed +0.5z finite differences in the three fixed backgrounds.
    base = _representative_backgrounds(ctx)
    frames, descriptors = [], []
    for _, row in base.iterrows():
        frames.append(row.copy()); descriptors.append((row["background"], "BASE"))
        for tag in ctx.numeric_tags:
            changed = row.copy(); changed[f"z__{tag}"] = float(changed[f"z__{tag}"]) + 0.5
            frames.append(changed); descriptors.append((row["background"], tag))
    frame = pd.DataFrame(frames)
    frame["sample_id"] = np.arange(-91000 - len(frame), -91000)
    encoded = ctx.infer_transformed_frame(frame)["encoded_mean_numeric"]
    cursor = 0
    for background in base["background"]:
        origin = encoded[cursor]; cursor += 1
        for tag in ctx.numeric_tags:
            directions.append(("single_element", f"{background}:{tag}", encoded[cursor] - origin)); cursor += 1

    meta = ctx.meta
    low = latent[meta["missing_fraction"].le(meta["missing_fraction"].quantile(.25))].mean(axis=0)
    high = latent[meta["missing_fraction"].ge(meta["missing_fraction"].quantile(.75))].mean(axis=0)
    directions.append(("confounder", "missingness_high_minus_low", high - low))
    for source in meta["data_source"].value_counts().head(8).index:
        mask = meta["data_source"].eq(source).to_numpy()
        directions.append(("source", f"source:{source}", latent[mask].mean(axis=0) - latent.mean(axis=0)))

    rows = []
    for family, name, direction in directions:
        norm2 = float(np.dot(direction, direction))
        coeff = components @ direction
        energy = coeff * coeff / max(norm2, 1e-12)
        cumulative = np.cumsum(energy)
        for pc in range(len(components)):
            rows.append({"direction_family": family, "direction": name, "pc": pc + 1,
                         "pc_energy_fraction": energy[pc], "cumulative_energy_fraction": cumulative[pc],
                         "direction_norm": np.sqrt(norm2)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "direction_pc_energy.csv", index=False)
    summary = result.loc[result["pc"].isin([2, 3, 32, 64])].pivot_table(
        index=["direction_family", "direction"], columns="pc", values="cumulative_energy_fraction"
    ).reset_index().rename(columns={2: "energy_top2", 3: "energy_top3", 32: "energy_top32", 64: "energy_top64"})
    summary.to_csv(out / "direction_energy_summary.csv", index=False)

    matrix = np.stack([d for f, _, d in directions if f == "single_element"])
    singular = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
    cumulative = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    pd.DataFrame({"component": np.arange(1, len(singular) + 1), "singular_value": singular,
                  "cumulative_direction_energy": cumulative}).to_csv(out / "perturbation_effective_rank.csv", index=False)
    return result


def _basis_stability(ctx: ExperimentContext, manager: ProjectionManager, fit_idx: np.ndarray, repeats: int, out: Path) -> pd.DataFrame:
    rng = np.random.default_rng(ctx.seed + 911)
    reps = {
        "contextual_mean": ctx.latents("numeric_only")["encoded_mean_numeric"][fit_idx],
        "pure_label_projected": ctx.project_pure_labels(True),
    }
    rows = []
    for name, x in reps.items():
        k = min(32, x.shape[0] - 1, x.shape[1])
        solver = "randomized" if len(x) > 512 else "full"
        fit_n = min(len(x), 2000)
        reference_sample = x[:fit_n]
        reference = PCA(n_components=k, svd_solver=solver, random_state=ctx.seed).fit(reference_sample).components_.T
        for repeat in range(repeats):
            sample = x[rng.integers(0, len(x), fit_n)]
            fitted = PCA(n_components=k, svd_solver=solver, random_state=ctx.seed + repeat + 1).fit(sample).components_.T
            angles = np.degrees(subspace_angles(reference, fitted))
            rows.append({"representation_level": name, "bootstrap_repeat": repeat + 1, "components": k,
                         "mean_principal_angle_deg": float(angles.mean()), "max_principal_angle_deg": float(angles.max())})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "pca_basis_bootstrap_stability.csv", index=False)
    return frame


def _figures(signal: pd.DataFrame, attribution: pd.DataFrame, ctx: ExperimentContext, manager: ProjectionManager,
             fit_idx: np.ndarray, eval_idx: np.ndarray, root: Path) -> None:
    figdir = root / "figures"
    main = signal[(signal["representation_level"] == "contextual_mean") &
                  (signal["metric"] == "knn15_purity") & signal["projection_name"].isin(["FULL", "PCA64", "PCA32", "PCA3", "PCA2"])]
    order = ["FULL", "PCA64", "PCA32", "PCA3", "PCA2"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label in ["rock_type", "data_source", "canonical_ref_id", "missingness_quartile"]:
        d = main[main["label"] == label].set_index("projection_name").reindex(order)
        ax.plot(order, d["metric_value"], marker="o", label=label)
    ax.set_ylabel("kNN15 purity"); ax.set_title("Contextual mean: geology and confounder signals vs dimension")
    ax.grid(alpha=.25); ax.legend(frameon=False, ncol=2); fig.tight_layout()
    fig.savefig(figdir / "dimension_curves" / "sample_signal_dimension_curve.png", dpi=180); plt.close(fig)

    pivot = attribution[attribution["representation_level"] == "contextual_mean"].pivot_table(
        index="variable", columns="pc", values="association", aggfunc="max"
    ).iloc[:, :32]
    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_yticks(np.arange(len(pivot))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(0, 32, 2)); ax.set_xticklabels(np.arange(1, 33, 2))
    ax.set_xlabel("PC"); ax.set_title("PC semantic attribution (association strength)")
    fig.colorbar(image, ax=ax, shrink=.75); fig.tight_layout()
    fig.savefig(figdir / "pc_attribution" / "contextual_mean_pc_attribution.png", dpi=180); plt.close(fig)

    latent = ctx.latents("numeric_only")["encoded_mean_numeric"]
    meta = ctx.meta.iloc[eval_idx].reset_index(drop=True)
    for dim in (2, 3):
        coords = manager.pca_transform("sample_contextual_mean_numeric", latent[fit_idx], latent[eval_idx], dim)
        if dim == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = meta["rock_group"].fillna("missing")
            for label in labels.value_counts().head(6).index:
                mask = labels.eq(label).to_numpy(); ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=.35, label=label)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA2 contextual mean — rock group")
            ax.legend(frameon=False, markerscale=2); fig.tight_layout(); fig.savefig(figdir / "pca2" / "sample_rock_group.png", dpi=180); plt.close(fig)
        else:
            fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
            labels = meta["rock_group"].fillna("missing")
            for label in labels.value_counts().head(6).index:
                mask = labels.eq(label).to_numpy(); ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], s=5, alpha=.25, label=label)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.set_title("PCA3 contextual mean — rock group")
            ax.legend(frameon=False, markerscale=2); fig.tight_layout(); fig.savefig(figdir / "pca3" / "sample_rock_group.png", dpi=180); plt.close(fig)


def run(config: dict[str, Any]) -> None:
    base_config = load_yaml(config["base_config"])
    settings = config["subspace_audit"]
    root = resolve_path(settings["output_root"])
    set_reproducible_seed(int(settings["seed"]), int(base_config["runtime"].get("num_threads", 8)))
    ctx = ExperimentContext(base_config)
    fit_idx, eval_idx = _pool_indices(ctx.meta, root / "projections" / "fitting_pools")
    manager = ProjectionManager(root / "projections", seed=int(settings["seed"]), dtype=settings["numerical_dtype"])
    out = root / "results" / "S"
    out.mkdir(parents=True, exist_ok=True)
    signal = _sample_signal_curves(ctx, manager, fit_idx, eval_idx, settings, out)
    attribution = _pc_attribution(ctx, manager, fit_idx, eval_idx, out)
    _direction_energy(ctx, manager, fit_idx, out)
    _basis_stability(ctx, manager, fit_idx, int(settings["basis_bootstrap_repeats"]), out)
    _figures(signal, attribution, ctx, manager, fit_idx, eval_idx, root)
    manager.write_manifest()
    (out / "specialty_manifest.json").write_text(json.dumps({
        "S1": "dimension curves from paired replay plus RP intervals",
        "S2": "PC energy for F2 finite differences and source/missingness directions",
        "S3": "64-PC semantic attribution for contextual mean, projected mean, and CLS",
        "S4": "geology/source/missingness kNN purity and grouped probes",
        "S5": "bootstrap PCA principal-angle and metric confidence intervals",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[subspace-specialty] signal_rows={len(signal)} attribution_rows={len(attribution)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml")
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
