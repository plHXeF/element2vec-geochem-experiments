from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from utils import OUTPUT_DIR, STAGE_DIR, load_config, markdown_table, save_dataframe, write_json


OUT_DIR = OUTPUT_DIR / "raw_embedding_benchmark"
SWEEP_DIR = OUT_DIR / "pca_dim_sweep"
FIG_DIR = SWEEP_DIR / "figures"
SPACE_NAME = "sample_raw_element_token_flat_476"
METRICS = [
    ("cumulative_variance", "Cumulative explained variance"),
    ("knn15", "kNN15 label purity"),
    ("kmeans_ari", "KMeans ARI"),
    ("kmeans_nmi", "KMeans NMI"),
]


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"sample_id", "deposit_type"}]


def knn_label_purity(x: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def run_sweep(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    path = OUT_DIR / "embeddings" / "sample_raw_element_token_flat.csv"
    df = pd.read_csv(path)
    labels_text = df["deposit_type"].fillna("<missing>").astype(str).to_numpy()
    y = LabelEncoder().fit_transform(labels_text)
    cols = feature_cols(df)
    x = df[cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    dims = list(range(8, min(32, x.shape[1]) + 1))
    seed = int(config.get("random_seed", 42))
    pca = PCA(n_components=max(dims), random_state=seed)
    coords_all = pca.fit_transform(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_clusters = len(np.unique(y))

    rows = []
    for dim in dims:
        coords = coords_all[:, :dim]
        clusters = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(coords)
        rows.append(
            {
                "space_name": SPACE_NAME,
                "dim": dim,
                "n_features": int(x.shape[1]),
                "cumulative_variance": float(cumulative[dim - 1]),
                "knn15": knn_label_purity(coords, labels_text, k=15),
                "kmeans_ari": float(adjusted_rand_score(y, clusters)),
                "kmeans_nmi": float(normalized_mutual_info_score(y, clusters)),
            }
        )
    metrics = pd.DataFrame(rows)
    summary = summarize(metrics)
    save_dataframe(metrics, SWEEP_DIR / "raw_embedding_pca_dimension_sweep_metrics.csv")
    save_dataframe(summary, SWEEP_DIR / "raw_embedding_pca_dimension_sweep_summary.csv")
    write_json(metrics.to_dict(orient="records"), SWEEP_DIR / "raw_embedding_pca_dimension_sweep_metrics.json")
    plot_per_metric(metrics)
    plot_panel(metrics)
    write_report(summary)
    print(f"[raw pca sweep] wrote {SWEEP_DIR}")
    return metrics, summary


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    group = metrics.sort_values("dim")
    row12 = group[group["dim"] == 12].iloc[0]
    row32 = group[group["dim"] == group["dim"].max()].iloc[0]
    best_knn = group.loc[group["knn15"].idxmax()]
    best_ari = group.loc[group["kmeans_ari"].idxmax()]
    best_nmi = group.loc[group["kmeans_nmi"].idxmax()]
    return pd.DataFrame(
        [
            {
                "space_name": SPACE_NAME,
                "n_features": int(group["n_features"].iloc[0]),
                "dim12_variance": float(row12["cumulative_variance"]),
                "dim12_knn15": float(row12["knn15"]),
                "dim12_kmeans_ari": float(row12["kmeans_ari"]),
                "dim12_kmeans_nmi": float(row12["kmeans_nmi"]),
                "max_dim": int(row32["dim"]),
                "maxdim_variance": float(row32["cumulative_variance"]),
                "maxdim_knn15": float(row32["knn15"]),
                "maxdim_kmeans_ari": float(row32["kmeans_ari"]),
                "maxdim_kmeans_nmi": float(row32["kmeans_nmi"]),
                "knn15_gap_maxdim_minus_12": float(row32["knn15"] - row12["knn15"]),
                "ari_gap_maxdim_minus_12": float(row32["kmeans_ari"] - row12["kmeans_ari"]),
                "nmi_gap_maxdim_minus_12": float(row32["kmeans_nmi"] - row12["kmeans_nmi"]),
                "best_knn15": float(best_knn["knn15"]),
                "best_knn15_dim": int(best_knn["dim"]),
                "best_kmeans_ari": float(best_ari["kmeans_ari"]),
                "best_kmeans_ari_dim": int(best_ari["dim"]),
                "best_kmeans_nmi": float(best_nmi["kmeans_nmi"]),
                "best_kmeans_nmi_dim": int(best_nmi["dim"]),
            }
        ]
    )


def save_plot(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_per_metric(metrics: pd.DataFrame) -> None:
    metrics = metrics.sort_values("dim")
    for metric, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        ax.plot(metrics["dim"], metrics[metric], marker="o", linewidth=2)
        ax.axvline(12, color="#333333", linestyle="--", linewidth=1.1, alpha=0.8, label="dim=12")
        ax.set_xlabel("PCA dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{SPACE_NAME}: {ylabel} vs PCA dimension")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, FIG_DIR / f"{metric}_vs_dim")


def plot_panel(metrics: pd.DataFrame) -> None:
    metrics = metrics.sort_values("dim")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), METRICS):
        ax.plot(metrics["dim"], metrics[metric], marker="o", linewidth=2)
        ax.axvline(12, color="#333333", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("PCA dimension")
    fig.suptitle(SPACE_NAME)
    save_plot(fig, FIG_DIR / "raw_embedding_pca_dim_sweep_panel")


def write_report(summary: pd.DataFrame) -> None:
    report = f"""# Raw 476D Embedding PCA Dimension Sweep

This sweep evaluates the unprojected sample-level token matrix:

`14 * (x_i + visible_mask_i + learned element embedding 32D) = 476`

Scan range: PCA dimensions 8 to 32.

## Summary

{markdown_table(summary)}

## Outputs

- `raw_embedding_pca_dimension_sweep_metrics.csv`
- `raw_embedding_pca_dimension_sweep_summary.csv`
- `figures/*_vs_dim.png/.svg`
- `figures/raw_embedding_pca_dim_sweep_panel.png/.svg`
"""
    (SWEEP_DIR / "raw_embedding_pca_dimension_sweep_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = parser.parse_args()
    run_sweep(load_config(args.config))


if __name__ == "__main__":
    main()
