from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PALETTE = {
    "orogenic": "#1f77b4",
    "sedimentary": "#2ca02c",
    "porphyry": "#ff7f0e",
    "carlin": "#d62728",
    "ni-cu-pge": "#9467bd",
    "vms": "#8c564b",
}


def _save(fig: plt.Figure, path_no_suffix: Path) -> None:
    path_no_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_suffix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_no_suffix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_pca_sweep(metrics: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        ("cumulative_variance", "Cumulative explained variance"),
        ("pca_knn15", "PCA latent kNN15 purity"),
        ("pca_knn30", "PCA latent kNN30 purity"),
        ("kmeans_ari", "KMeans ARI"),
        ("kmeans_nmi", "KMeans NMI"),
        ("gmm_ari", "GMM ARI"),
        ("gmm_nmi", "GMM NMI"),
    ]
    for metric, ylabel in specs:
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        sns.lineplot(
            data=metrics,
            x="dim",
            y=metric,
            hue="representation",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_xlabel("PCA latent dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs dimension")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        _save(fig, out_dir / f"{metric}_vs_dim")

    for rep, group in metrics.groupby("representation", sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
        panel_specs = [
            ("cumulative_variance", "Cumulative variance"),
            ("pca_knn15", "kNN15"),
            ("kmeans_ari", "KMeans ARI"),
            ("kmeans_nmi", "KMeans NMI"),
        ]
        for ax, (metric, title) in zip(axes.ravel(), panel_specs):
            ax.plot(group["dim"], group[metric], marker="o", linewidth=2)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("PCA latent dimension")
        fig.suptitle(rep)
        _save(fig, out_dir / f"{rep}_pca_metrics")


def plot_unsupervised_summary(metrics: pd.DataFrame, out_dir: Path) -> None:
    for metric, label in [
        ("input_knn15", "Input kNN15 purity"),
        ("input_knn30", "Input kNN30 purity"),
        ("kmeans_ari", "KMeans ARI"),
        ("gmm_ari", "GMM ARI"),
        ("silhouette", "Type-label silhouette"),
    ]:
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        order = metrics.sort_values(metric, ascending=False)["representation"]
        sns.barplot(data=metrics, x="representation", y=metric, order=order, ax=ax, color="#5B8DEF")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
        _save(fig, out_dir / f"unsupervised_{metric}")


def plot_umap_projections(coords: pd.DataFrame, out_dir: Path) -> None:
    if coords.empty:
        return
    pairs = [("UMAP1", "UMAP2"), ("UMAP1", "UMAP3"), ("UMAP2", "UMAP3")]
    for rep, group in coords.groupby("representation", sort=False):
        for x_col, y_col in pairs:
            fig, ax = plt.subplots(figsize=(8.2, 6.6))
            sns.scatterplot(
                data=group,
                x=x_col,
                y=y_col,
                hue="type",
                palette=PALETTE,
                s=12,
                alpha=0.72,
                linewidth=0,
                ax=ax,
            )
            ax.set_title(f"{rep}: {x_col} vs {y_col}")
            ax.grid(True, alpha=0.18)
            ax.legend(loc="best", fontsize=8, markerscale=1.8)
            _save(fig, out_dir / f"{rep}_{x_col}_{y_col}")


def save_plotly_umap(coords: pd.DataFrame, out_dir: Path) -> list[str]:
    notes: list[str] = []
    if coords.empty:
        return ["plotly skipped: no UMAP coordinates"]
    try:
        import plotly.express as px  # type: ignore
    except Exception as exc:
        return [f"plotly skipped: {type(exc).__name__}: {exc}"]

    for rep, group in coords.groupby("representation", sort=False):
        fig = px.scatter_3d(
            group,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color="type",
            hover_data=["sample_id"],
            title=f"{rep} UMAP-3D",
            opacity=0.72,
        )
        path = out_dir / f"{rep}_umap3d_interactive.html"
        fig.write_html(path)
        notes.append(f"ok: {path.name}")
    return notes


def plot_feature_importance(importance: pd.DataFrame, out_dir: Path) -> None:
    if importance.empty:
        return
    for rep, group in importance.groupby("representation", sort=False):
        top = group.sort_values("importance_mean", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        sns.barplot(data=top, x="importance_mean", y="feature", ax=ax, color="#4E9F7B")
        ax.set_xlabel("Permutation importance mean")
        ax.set_ylabel("")
        ax.set_title(f"{rep}: permutation importance")
        ax.grid(True, axis="x", alpha=0.25)
        _save(fig, out_dir / f"{rep}_permutation_importance")
