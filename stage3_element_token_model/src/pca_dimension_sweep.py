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

from utils import OUTPUT_DIR, STAGE_DIR, ensure_output_dirs, load_config, markdown_table, save_dataframe, write_json


SPACE_SPECS = {
    "sample_input_token_flat": OUTPUT_DIR / "embeddings" / "sample_input_token_flat.csv",
    "sample_encoded_token_flat": OUTPUT_DIR / "embeddings" / "sample_encoded_token_flat.csv",
    "transformer_latent_label_masked_mean_element": OUTPUT_DIR
    / "latents"
    / "transformer_latent_label_masked_mean_element.csv",
}

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


def run_one_space(space_name: str, path: Path, dims: list[int], random_seed: int) -> pd.DataFrame:
    print(f"[pca sweep] loading {space_name}: {path}")
    df = pd.read_csv(path)
    labels_text = df["deposit_type"].fillna("<missing>").astype(str).to_numpy()
    y = LabelEncoder().fit_transform(labels_text)
    cols = feature_cols(df)
    x = df[cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    max_dim = min(max(dims), x.shape[1])
    valid_dims = [d for d in dims if d <= max_dim]
    pca = PCA(n_components=max_dim, random_state=random_seed)
    coords_all = pca.fit_transform(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_clusters = len(np.unique(y))

    rows = []
    for dim in valid_dims:
        coords = coords_all[:, :dim]
        clusters = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_seed).fit_predict(coords)
        rows.append(
            {
                "space_name": space_name,
                "dim": dim,
                "n_features": x.shape[1],
                "cumulative_variance": float(cumulative[dim - 1]),
                "knn15": knn_label_purity(coords, labels_text, k=15),
                "kmeans_ari": float(adjusted_rand_score(y, clusters)),
                "kmeans_nmi": float(normalized_mutual_info_score(y, clusters)),
            }
        )
    return pd.DataFrame(rows)


def save_plot(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    for metric, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(9.6, 5.5))
        for space, group in metrics.groupby("space_name", sort=False):
            group = group.sort_values("dim")
            ax.plot(group["dim"], group[metric], marker="o", linewidth=2, label=space)
        ax.axvline(12, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8, label="dim=12")
        ax.set_xlabel("PCA dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs PCA dimension")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        save_plot(fig, out_dir / f"{metric}_vs_dim_comparison")


def plot_per_space(metrics: pd.DataFrame, out_dir: Path) -> None:
    for space, group in metrics.groupby("space_name", sort=False):
        group = group.sort_values("dim")
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
        for ax, (metric, ylabel) in zip(axes.ravel(), METRICS):
            ax.plot(group["dim"], group[metric], marker="o", linewidth=2)
            ax.axvline(12, color="#333333", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.set_title(ylabel)
            ax.grid(True, alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("PCA dimension")
        fig.suptitle(space)
        save_plot(fig, out_dir / f"{space}_pca_dim_sweep")


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for space, group in metrics.groupby("space_name", sort=False):
        group = group.sort_values("dim")
        row12 = group[group["dim"] == 12]
        row32 = group[group["dim"] == group["dim"].max()]
        best_knn = group.loc[group["knn15"].idxmax()]
        best_ari = group.loc[group["kmeans_ari"].idxmax()]
        best_nmi = group.loc[group["kmeans_nmi"].idxmax()]
        if not row12.empty:
            r12 = row12.iloc[0]
            rmax = row32.iloc[0]
            knn_gap = float(rmax["knn15"] - r12["knn15"])
            ari_gap = float(rmax["kmeans_ari"] - r12["kmeans_ari"])
            nmi_gap = float(rmax["kmeans_nmi"] - r12["kmeans_nmi"])
        else:
            r12 = None
            rmax = row32.iloc[0]
            knn_gap = ari_gap = nmi_gap = float("nan")
        rows.append(
            {
                "space_name": space,
                "n_features": int(group["n_features"].iloc[0]),
                "dim12_variance": float(r12["cumulative_variance"]) if r12 is not None else float("nan"),
                "dim12_knn15": float(r12["knn15"]) if r12 is not None else float("nan"),
                "dim12_kmeans_ari": float(r12["kmeans_ari"]) if r12 is not None else float("nan"),
                "dim12_kmeans_nmi": float(r12["kmeans_nmi"]) if r12 is not None else float("nan"),
                "max_dim": int(rmax["dim"]),
                "maxdim_variance": float(rmax["cumulative_variance"]),
                "maxdim_knn15": float(rmax["knn15"]),
                "maxdim_kmeans_ari": float(rmax["kmeans_ari"]),
                "maxdim_kmeans_nmi": float(rmax["kmeans_nmi"]),
                "knn15_gap_maxdim_minus_12": knn_gap,
                "ari_gap_maxdim_minus_12": ari_gap,
                "nmi_gap_maxdim_minus_12": nmi_gap,
                "best_knn15": float(best_knn["knn15"]),
                "best_knn15_dim": int(best_knn["dim"]),
                "best_kmeans_ari": float(best_ari["kmeans_ari"]),
                "best_kmeans_ari_dim": int(best_ari["dim"]),
                "best_kmeans_nmi": float(best_nmi["kmeans_nmi"]),
                "best_kmeans_nmi_dim": int(best_nmi["dim"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    notes = []
    for row in summary.itertuples(index=False):
        if row.knn15_gap_maxdim_minus_12 > 0.01 or row.ari_gap_maxdim_minus_12 > 0.03 or row.nmi_gap_maxdim_minus_12 > 0.03:
            verdict = "12D likely leaves useful structure on the table."
        else:
            verdict = "12D is close to the upper scan range for the tracked metrics."
        notes.append(f"- `{row.space_name}`: {verdict}")

    report = f"""# Stage 3 PCA Dimension Sweep

Scan range: PCA dimensions 8 to 32.

Spaces:

- `sample_input_token_flat`
- `sample_encoded_token_flat`
- `transformer_latent_label_masked_mean_element`

Metrics:

- cumulative explained variance
- kNN15 label purity
- KMeans ARI
- KMeans NMI

The vertical dashed line in the figures marks 12 dimensions, because the previous Stage3 benchmark used PCA12 for the flattened token matrices.

## Summary

{markdown_table(summary)}

## Reading On Whether 12D Is Too Small

{chr(10).join(notes)}

## Outputs

- `token_pca_dimension_sweep_metrics.csv`
- `token_pca_dimension_sweep_summary.csv`
- `figures/*_comparison.png/.svg`
- `figures/*_pca_dim_sweep.png/.svg`
"""
    (out_dir / "token_pca_dimension_sweep_report.md").write_text(report, encoding="utf-8")


def run_sweep(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs()
    out_dir = OUTPUT_DIR / "pca_dim_sweep"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dims = list(range(8, 33))
    seed = int(config.get("random_seed", 42))
    parts = []
    for space_name, path in SPACE_SPECS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(run_one_space(space_name, path, dims, seed))
    metrics = pd.concat(parts, ignore_index=True)
    summary = summarize(metrics)
    save_dataframe(metrics, out_dir / "token_pca_dimension_sweep_metrics.csv")
    save_dataframe(summary, out_dir / "token_pca_dimension_sweep_summary.csv")
    write_json(metrics.to_dict(orient="records"), out_dir / "token_pca_dimension_sweep_metrics.json")
    plot_comparison(metrics, fig_dir)
    plot_per_space(metrics, fig_dir)
    write_report(metrics, summary, out_dir)
    print(f"[pca sweep] wrote {out_dir}")
    return metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = parser.parse_args()
    run_sweep(load_config(args.config))


if __name__ == "__main__":
    main()
