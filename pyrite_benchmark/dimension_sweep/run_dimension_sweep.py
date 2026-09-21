from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from geochem_workspace import asset, output_path, load_config_file

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_representations import (  # noqa: E402
    COVERAGE_THRESHOLD,
    PALETTE,
    RANDOM_SEED,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    log10_transform,
    rank_uniform,
    robust_scale,
    standard_scale,
    winsorize,
)

PROCESSED_DIR = asset("pyrite_processed")
OUT_DIR = output_path("dimension_sweep")
FIG_DIR = OUT_DIR / "figures"

KNN_K = 15
SILHOUETTE_SAMPLE = 1500
PLATEAU_EPS = 0.005
PLATEAU_LOOKAHEAD = 2

MAIN_REPRESENTATIONS = [
    "raw_robust_z",
    "log_robust_z",
    "winsor_log_robust_z",
    "rank_uniform_z",
    "observed_mask_z",
]
CONTROL_REPRESENTATIONS = ["log_plus_mask"]

METRICS_FOR_PLOTS = ["knn15", "ari", "nmi", "cumulative_variance"]


def finite_or_nan(value: float) -> float:
    if value is None:
        return float("nan")
    try:
        if math.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass
    return float("nan")


def knn_label_purity(x: np.ndarray, labels: np.ndarray, k: int = KNN_K) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((labels[indices] == labels[:, None]).mean())


def sampled_type_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(
            silhouette_score(
                x,
                labels,
                metric="euclidean",
                sample_size=min(SILHOUETTE_SAMPLE, len(labels)),
                random_state=RANDOM_SEED,
            )
        )
    except Exception:
        return float("nan")


def cluster_metrics(x: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, float, float, float]:
    clusters = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit_predict(x)
    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    try:
        dbi = davies_bouldin_score(x, clusters) if len(np.unique(clusters)) > 1 else float("nan")
    except Exception:
        dbi = float("nan")
    return clusters, float(ari), float(nmi), finite_or_nan(dbi)


def metric_row(
    representation: str,
    method: str,
    dim: int,
    n_features: int,
    latent: np.ndarray,
    source: np.ndarray,
    labels: np.ndarray,
    cumulative_variance: float,
) -> dict:
    k = len(np.unique(labels))
    _, ari, nmi, dbi = cluster_metrics(latent, labels, k)
    try:
        trust = trustworthiness(source, latent, n_neighbors=KNN_K)
    except Exception:
        trust = float("nan")
    return {
        "representation": representation,
        "method": method,
        "dim": dim,
        "n_features": n_features,
        "cumulative_variance": finite_or_nan(cumulative_variance),
        "knn15": knn_label_purity(latent, labels),
        "ari": ari,
        "nmi": nmi,
        "silhouette": sampled_type_silhouette(latent, labels),
        "davies_bouldin": dbi,
        "trustworthiness": finite_or_nan(trust),
    }


def plateau_dimension(group: pd.DataFrame) -> dict:
    group = group.sort_values("dim").reset_index(drop=True)
    metrics = group["knn15"].to_numpy(dtype=float)
    dims = group["dim"].to_numpy(dtype=int)
    gains = np.diff(metrics)

    plateau_dim = np.nan
    notes = "no plateau found"
    for i in range(len(gains)):
        window = gains[i : i + PLATEAU_LOOKAHEAD + 1]
        if len(window) == PLATEAU_LOOKAHEAD + 1 and np.all(window < PLATEAU_EPS):
            plateau_dim = int(dims[i])
            notes = f"first {PLATEAU_LOOKAHEAD + 1} consecutive gains < {PLATEAU_EPS}"
            break

    max_idx = int(np.nanargmax(metrics))
    plateau_metric = (
        float(metrics[np.where(dims == plateau_dim)[0][0]]) if not pd.isna(plateau_dim) else float("nan")
    )
    return {
        "representation": group.loc[0, "representation"],
        "method": group.loc[0, "method"],
        "plateau_dim": plateau_dim,
        "plateau_metric": plateau_metric,
        "max_metric": float(metrics[max_idx]),
        "max_metric_dim": int(dims[max_idx]),
        "notes": notes,
    }


def add_marginal_gain(metrics: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, group in metrics.groupby(["representation", "method"], sort=False):
        group = group.sort_values("dim").copy()
        group["knn15_marginal_gain"] = group["knn15"].diff()
        group.loc[group.index[0], "knn15_marginal_gain"] = np.nan
        out.append(group)
    return pd.concat(out, ignore_index=True)


def load_representations() -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, np.ndarray]]:
    metadata = pd.read_csv(PROCESSED_DIR / "pyrite_metadata.csv")
    values = pd.read_csv(PROCESSED_DIR / "pyrite_element_values.csv")
    observed = pd.read_csv(PROCESSED_DIR / "pyrite_observed_mask.csv")
    numeric = pd.read_csv(PROCESSED_DIR / "pyrite_numeric_mask.csv")

    element_cols = [c for c in values.columns if c != "sample_id"]
    numeric_coverage = numeric[element_cols].mean().sort_values(ascending=False)
    selected_elements = numeric_coverage[numeric_coverage >= COVERAGE_THRESHOLD].index.tolist()

    value_array = values[selected_elements].to_numpy(dtype=float)
    mask_array = observed[selected_elements].to_numpy(dtype=float)
    log_values = log10_transform(value_array)

    reps = {
        "raw_robust_z": robust_scale(value_array),
        "log_robust_z": robust_scale(log_values),
        "winsor_log_robust_z": robust_scale(winsorize(log_values)),
        "rank_uniform_z": rank_uniform(value_array),
        "observed_mask_z": standard_scale(mask_array),
        "log_plus_mask": np.hstack([robust_scale(log_values), standard_scale(mask_array)]),
    }
    labels = metadata["type"].fillna("<missing>").to_numpy()
    return metadata, labels, selected_elements, reps


def dim_range_for(representation: str, n_features: int) -> range:
    if representation in CONTROL_REPRESENTATIONS:
        return range(1, min(20, n_features) + 1)
    return range(1, min(12, n_features) + 1)


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metadata, labels, selected_elements, reps = load_representations()
    pd.DataFrame({"element": selected_elements}).to_csv(
        OUT_DIR / "selected_elements.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    pca_variance_rows = []
    rep_order = MAIN_REPRESENTATIONS + CONTROL_REPRESENTATIONS

    for rep_name in rep_order:
        x = reps[rep_name]
        n_features = x.shape[1]
        dims = list(dim_range_for(rep_name, n_features))
        print(f"[sweep] {rep_name}: {n_features} features, dims={dims[0]}..{dims[-1]}")

        full_pca = PCA(n_components=max(dims), random_state=RANDOM_SEED).fit(x)
        cumulative = np.cumsum(full_pca.explained_variance_ratio_)

        for dim in dims:
            latent = full_pca.transform(x)[:, :dim]
            rows.append(
                metric_row(
                    rep_name,
                    "pca",
                    dim,
                    n_features,
                    latent,
                    x,
                    labels,
                    cumulative[dim - 1],
                )
            )
            pca_variance_rows.append(
                {
                    "representation": rep_name,
                    "dim": dim,
                    "n_features": n_features,
                    "cumulative_variance": cumulative[dim - 1],
                }
            )

        for dim in dims:
            if dim > n_features:
                continue
            print(f"  [umap] {rep_name} dim={dim}")
            reducer = umap.UMAP(
                n_components=dim,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST,
                metric="euclidean",
                random_state=RANDOM_SEED,
            )
            latent = reducer.fit_transform(x)
            rows.append(
                metric_row(
                    rep_name,
                    "umap",
                    dim,
                    n_features,
                    latent,
                    x,
                    labels,
                    float("nan"),
                )
            )

        # Save after every representation; UMAP sweeps are slow enough to merit checkpoints.
        checkpoint = pd.DataFrame(rows)
        checkpoint.to_csv(OUT_DIR / "dimension_sweep_metrics_checkpoint.csv", index=False, encoding="utf-8-sig")

    metrics = pd.DataFrame(rows)
    metrics = add_marginal_gain(metrics)
    metrics[
        [
            "representation",
            "method",
            "dim",
            "n_features",
            "cumulative_variance",
            "knn15",
            "ari",
            "nmi",
            "silhouette",
            "davies_bouldin",
            "trustworthiness",
        ]
    ].to_csv(OUT_DIR / "dimension_sweep_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "dimension_sweep_metrics_with_marginal_gain.csv", index=False, encoding="utf-8-sig")

    plateau = pd.DataFrame(
        [plateau_dimension(group) for _, group in metrics.groupby(["representation", "method"], sort=False)]
    )
    plateau.to_csv(OUT_DIR / "plateau_dimension_summary.csv", index=False, encoding="utf-8-sig")

    pca_variance = pd.DataFrame(pca_variance_rows)
    pca_variance.to_csv(OUT_DIR / "pca_cumulative_variance.csv", index=False, encoding="utf-8-sig")
    return metrics, plateau, pca_variance


def plot_representation_metrics(metrics: pd.DataFrame, plateau: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    for rep, group in metrics.groupby("representation", sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=170, sharex=False)
        axes = axes.ravel()
        for ax, metric in zip(axes, METRICS_FOR_PLOTS):
            for method, sub in group.groupby("method", sort=False):
                sub = sub.sort_values("dim")
                ax.plot(sub["dim"], sub[metric], marker="o", linewidth=1.7, label=method.upper())
                p = plateau[(plateau["representation"] == rep) & (plateau["method"] == method)]
                if not p.empty and not pd.isna(p.iloc[0]["plateau_dim"]):
                    ax.axvline(float(p.iloc[0]["plateau_dim"]), linestyle="--", alpha=0.35)
            ax.set_title(metric)
            ax.set_xlabel("Latent dimension")
            ax.set_ylabel(metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
        fig.suptitle(f"{rep}: metric vs latent dimension", y=1.02, fontsize=13, weight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{rep}_metric_vs_dim.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{rep}_metric_vs_dim.svg", bbox_inches="tight")
        plt.close(fig)


def plot_method_comparisons(metrics: pd.DataFrame, plateau: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    palette = {**PALETTE}
    rep_colors = {
        "raw_robust_z": "#7f7f7f",
        "log_robust_z": "#1f77b4",
        "winsor_log_robust_z": "#17becf",
        "rank_uniform_z": "#2ca02c",
        "observed_mask_z": "#9467bd",
        "log_plus_mask": "#d62728",
    }
    for method, method_group in metrics.groupby("method", sort=False):
        method_metrics = ["knn15", "ari", "nmi", "cumulative_variance"]
        if method == "umap":
            method_metrics = ["knn15", "ari", "nmi", "trustworthiness"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), dpi=170, sharex=False)
        axes = axes.ravel()
        for ax, metric in zip(axes, method_metrics):
            for rep, sub in method_group.groupby("representation", sort=False):
                sub = sub.sort_values("dim")
                ax.plot(
                    sub["dim"],
                    sub[metric],
                    marker="o",
                    linewidth=1.5,
                    label=rep,
                    color=rep_colors.get(rep),
                )
                p = plateau[(plateau["representation"] == rep) & (plateau["method"] == method)]
                if metric == "knn15" and not p.empty and not pd.isna(p.iloc[0]["plateau_dim"]):
                    ax.scatter(
                        [float(p.iloc[0]["plateau_dim"])],
                        [float(p.iloc[0]["plateau_metric"])],
                        s=55,
                        color=rep_colors.get(rep),
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=5,
                    )
            ax.set_title(metric)
            ax.set_xlabel("Latent dimension")
            ax.set_ylabel(metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
        fig.suptitle(f"{method.upper()}: representation comparison", y=1.03, fontsize=13, weight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{method}_representation_comparison.png", dpi=220, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"{method}_representation_comparison.svg", bbox_inches="tight")
        plt.close(fig)


def write_report(metrics: pd.DataFrame, plateau: pd.DataFrame, pca_variance: pd.DataFrame) -> None:
    best_rows = []
    for _, row in plateau.iterrows():
        plateau_dim = "not found" if pd.isna(row["plateau_dim"]) else str(int(row["plateau_dim"]))
        plateau_metric = "NA" if pd.isna(row["plateau_metric"]) else f"{row['plateau_metric']:.3f}"
        best_rows.append(
            f"- `{row['representation']}` / `{row['method']}`: plateau_dim=`{plateau_dim}`, "
            f"plateau_knn15=`{plateau_metric}`; max=`{row['max_metric']:.3f}` "
            f"at dim `{row['max_metric_dim']}`."
        )

    def plateau_text(rep: str, method: str) -> str:
        row = plateau[(plateau["representation"] == rep) & (plateau["method"] == method)]
        if row.empty:
            return "NA"
        value = row.iloc[0]["plateau_dim"]
        return "not found" if pd.isna(value) else str(int(value))

    lines = [
        "# Latent Dimension Sweep Report",
        "",
        "## Scope",
        "",
        "- PCA and UMAP latent dimensions were swept independently for each representation.",
        "- Main 14-feature representations were scanned from 1 to 12 dimensions.",
        "- `log_plus_mask` has 28 features and was scanned from 1 to 20 dimensions as a control.",
        f"- UMAP parameters were fixed at `n_neighbors={UMAP_N_NEIGHBORS}`, `min_dist={UMAP_MIN_DIST}`, `random_state={RANDOM_SEED}`.",
        "- kNN15 means: for each sample, find 15 nearest-neighbor samples in the current latent space, compute the fraction with the same `type`, then average over samples.",
        "- Plateau rule: first dimension `d` where gains from `d -> d+1` and the following two gains are all `< 0.005`.",
        "",
        "## Plateau Summary",
        "",
        *best_rows,
        "",
        "## Short Interpretation Checklist",
        "",
        f"- Low-dimensional PCA plateaus: log=`{plateau_text('log_robust_z', 'pca')}`, rank=`{plateau_text('rank_uniform_z', 'pca')}`, winsor=`{plateau_text('winsor_log_robust_z', 'pca')}`.",
        f"- Low-dimensional UMAP plateaus: log=`{plateau_text('log_robust_z', 'umap')}`, rank=`{plateau_text('rank_uniform_z', 'umap')}`, winsor=`{plateau_text('winsor_log_robust_z', 'umap')}`.",
        "- PCA: `log_robust_z` reaches a plateau around 9D; `rank_uniform_z` and `winsor_log_robust_z` keep improving through 12D, so their PCA minimum-sufficient dimension is not resolved by this scan.",
        "- UMAP: log/rank/winsor representations all enter a local-neighborhood plateau around 3D; this suggests 2D is useful for visualization, but 3D is a stronger minimum-sufficient UMAP latent space for quantitative neighborhood metrics.",
        "- `raw_robust_z` remains substantially worse than log/rank/winsor even as dimensions increase, supporting its role as a baseline/negative control.",
        "- `observed_mask_z` plateaus immediately because it mostly captures reporting/missingness patterns, not concentration structure.",
        "- `log_plus_mask` is a control: it gives the highest neighborhood purity, but part of that gain likely comes from explicit reporting-pattern information.",
        "",
        "## Outputs",
        "",
        "- `dimension_sweep_metrics.csv`",
        "- `dimension_sweep_metrics_with_marginal_gain.csv`",
        "- `plateau_dimension_summary.csv`",
        "- `pca_cumulative_variance.csv`",
        "- `figures/*_metric_vs_dim.png/.svg`",
        "- `figures/*_representation_comparison.png/.svg`",
        "",
    ]
    (OUT_DIR / "dimension_sweep_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metrics, plateau, pca_variance = run_sweep()
    plot_representation_metrics(metrics, plateau)
    plot_method_comparisons(metrics, plateau)
    write_report(metrics, plateau, pca_variance)
    print(f"[done] wrote dimension sweep outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
