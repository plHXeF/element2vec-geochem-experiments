from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import OUTPUT_DIR, STAGE_DIR, ensure_output_dirs, load_config


DISPLAY_NAMES = {
    "previous_baseline_space": "Stage2 baseline",
    "log_robust_space": "Log robust",
    "log_robust_pca12_space": "Log robust PCA12",
    "sample_input_token_flat_pca12_space": "Input token PCA12",
    "sample_encoded_token_flat_pca12_space": "Encoded token PCA12",
    "transformer_latent_label_masked_space": "Transformer latent",
}

SCENARIOS = {
    "XGB macro F1": "xgb_macro_f1_mean",
    "KNN macro F1": "knn15_macro_f1_mean",
    "Neighborhood": "knn15_neighborhood_purity",
    "KMeans ARI": "kmeans_ari",
    "KMeans NMI": "kmeans_nmi",
    "Silhouette": "silhouette_score",
}


def _out_dir() -> Path:
    path = OUTPUT_DIR / "report_fig"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _space_order(df: pd.DataFrame) -> list[str]:
    return [name for name in DISPLAY_NAMES if name in set(df["space_name"])]


def _display(space: str) -> str:
    return DISPLAY_NAMES.get(space, space)


def _save(fig: plt.Figure, name: str) -> None:
    out = _out_dir()
    fig.savefig(out / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def normalized_capability(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        item = {"space_name": row["space_name"], "display_name": _display(row["space_name"])}
        for label, col in SCENARIOS.items():
            value = float(row[col]) if pd.notna(row[col]) else np.nan
            if col == "silhouette_score":
                # Silhouette can be negative; map [-1, 1] to [0, 1] for visual comparison only.
                value = (value + 1.0) / 2.0
            item[label] = value
        rows.append(item)
    return pd.DataFrame(rows)


def plot_metric_heatmap(summary: pd.DataFrame) -> None:
    cap = normalized_capability(summary).set_index("display_name")
    matrix = cap[list(SCENARIOS.keys())].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(list(SCENARIOS.keys()), rotation=25, ha="right")
    ax.set_yticks(np.arange(len(cap.index)))
    ax.set_yticklabels(cap.index)
    ax.set_title("Representation capability heatmap")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if matrix[i, j] < np.nanmean(matrix) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _save(fig, "capability_heatmap")


def plot_radar(summary: pd.DataFrame) -> None:
    cap = normalized_capability(summary)
    categories = list(SCENARIOS.keys())
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8.5, 8.2))
    ax = fig.add_subplot(111, polar=True)
    for _, row in cap.iterrows():
        values = [float(row[c]) for c in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.8, label=row["display_name"])
        ax.fill(angles, values, alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.0)
    ax.set_title("Scenario capability radar")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    _save(fig, "capability_radar")


def plot_classification(summary: pd.DataFrame) -> None:
    use = summary.copy()
    use["display_name"] = use["space_name"].map(_display)
    order = [_display(s) for s in _space_order(use)]
    x = np.arange(len(order))
    width = 0.32
    indexed = use.set_index("display_name").loc[order]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x - width / 2, indexed["xgb_macro_f1_mean"], width, yerr=indexed["xgb_macro_f1_std"], label="XGB macro F1", color="#4477AA", capsize=3)
    ax.bar(x + width / 2, indexed["knn15_macro_f1_mean"], width, yerr=indexed["knn15_macro_f1_std"], label="KNN15 macro F1", color="#66CCEE", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Macro F1")
    ax.set_title("Classification sanity-check performance")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, "classification_macro_f1")


def plot_structure(summary: pd.DataFrame) -> None:
    use = summary.copy()
    use["display_name"] = use["space_name"].map(_display)
    order = [_display(s) for s in _space_order(use)]
    indexed = use.set_index("display_name").loc[order]
    metrics = [
        ("knn15_neighborhood_purity", "kNN15 purity", "#228833"),
        ("kmeans_ari", "KMeans ARI", "#CC6677"),
        ("kmeans_nmi", "KMeans NMI", "#AA3377"),
    ]
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    for offset, (col, label, color) in zip([-width, 0, width], metrics):
        ax.bar(x + offset, indexed[col], width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Local neighborhood and clustering structure")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, "structure_metrics")


def plot_token_gain(summary: pd.DataFrame) -> None:
    base = summary.loc[summary["space_name"] == "log_robust_space"]
    if base.empty:
        return
    base_row = base.iloc[0]
    focus = summary[summary["space_name"].isin([
        "sample_input_token_flat_pca12_space",
        "sample_encoded_token_flat_pca12_space",
        "transformer_latent_label_masked_space",
    ])].copy()
    focus["display_name"] = focus["space_name"].map(_display)
    cols = {
        "xgb_macro_f1_mean": "XGB macro F1",
        "knn15_neighborhood_purity": "kNN15 purity",
        "kmeans_ari": "KMeans ARI",
        "kmeans_nmi": "KMeans NMI",
    }
    gains = []
    for _, row in focus.iterrows():
        for col, label in cols.items():
            gains.append({"representation": row["display_name"], "metric": label, "delta_vs_log": float(row[col] - base_row[col])})
    df = pd.DataFrame(gains)
    reps = df["representation"].unique().tolist()
    metrics = list(cols.values())
    x = np.arange(len(metrics))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10.6, 5.3))
    colors = ["#EE7733", "#0077BB", "#009988"]
    for i, rep in enumerate(reps):
        vals = df[df["representation"] == rep].set_index("metric").loc[metrics]["delta_vs_log"]
        ax.bar(x + (i - 1) * width, vals, width, label=rep, color=colors[i % len(colors)])
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Delta vs log robust space")
    ax.set_title("Token representation gains relative to log robust")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    _save(fig, "token_gain_vs_log_robust")


def plot_loss_curves() -> None:
    path = OUTPUT_DIR / "logs" / "training_log.csv"
    if not path.exists():
        return
    hist = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(hist["epoch"], hist["loss"], label="total", linewidth=1.6)
    axes[0].plot(hist["epoch"], hist["element_loss"], label="element", linewidth=1.4)
    axes[0].plot(hist["epoch"], hist["type_loss"], label="deposit type", linewidth=1.4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training losses")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(hist["epoch"], hist["learning_rate"], color="#7b3294", linewidth=1.7)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning rate")
    axes[1].set_title("Cosine annealing schedule")
    axes[1].grid(True, alpha=0.25)
    _save(fig, "training_loss_and_lr")


def generate_report_figures(config: dict) -> None:
    ensure_output_dirs()
    summary_path = OUTPUT_DIR / "metrics" / "all_space_benchmark_summary.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[summary["n_features"] > 0].copy()
    plot_metric_heatmap(summary)
    plot_radar(summary)
    plot_classification(summary)
    plot_structure(summary)
    plot_token_gain(summary)
    plot_loss_curves()
    print(f"[report_figures] wrote figures to {OUTPUT_DIR / 'report_fig'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = parser.parse_args()
    generate_report_figures(load_config(args.config))


if __name__ == "__main__":
    main()
