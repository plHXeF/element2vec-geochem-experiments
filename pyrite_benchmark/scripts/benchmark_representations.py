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
from geochem_workspace import asset, output_path, load_config_file, configured_seed

import html
import math
import os
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = asset("pyrite_processed")
RESULT_DIR = output_path("results/representation_benchmark")
EMBED_DIR = RESULT_DIR / "embeddings"
FIG_DIR = RESULT_DIR / "figures"

RANDOM_SEED = configured_seed(42)
COVERAGE_THRESHOLD = 0.20
KNN_K = 15
SILHOUETTE_SAMPLE = 1500
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.10


PALETTE = {
    "orogenic": "#1f77b4",
    "sedimentary": "#2ca02c",
    "porphyry": "#ff7f0e",
    "carlin": "#d62728",
    "ni-cu-pge": "#9467bd",
    "vms": "#8c564b",
    "<missing>": "#7f7f7f",
}

LABEL_ORDER = ["orogenic", "sedimentary", "porphyry", "carlin", "ni-cu-pge", "vms", "<missing>"]


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_latest{path.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        return fallback


def robust_scale(values: np.ndarray) -> np.ndarray:
    med = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = np.nan
    imputed = np.where(np.isnan(values), med, values)
    scaled = (imputed - med) / iqr
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def standard_scale(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std[std == 0] = 1.0
    imputed = np.where(np.isnan(values), mean, values)
    return np.nan_to_num((imputed - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def log10_transform(values: np.ndarray) -> np.ndarray:
    out = values.astype(float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite_positive = col[np.isfinite(col) & (col > 0)]
        pseudo = np.nanmin(finite_positive) / 2.0 if finite_positive.size else 1e-9
        col = np.where(np.isnan(col), np.nan, np.maximum(col, pseudo))
        out[:, j] = np.log10(col)
    return out


def winsorize(values: np.ndarray, lower: float = 1, upper: float = 99) -> np.ndarray:
    out = values.copy()
    lo = np.nanpercentile(out, lower, axis=0)
    hi = np.nanpercentile(out, upper, axis=0)
    return np.minimum(np.maximum(out, lo), hi)


def rank_uniform(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    for j in range(values.shape[1]):
        col = values[:, j]
        valid = np.isfinite(col)
        n = int(valid.sum())
        if n == 0:
            continue
        ranks = pd.Series(col[valid]).rank(method="average").to_numpy()
        out[valid, j] = ranks / (n + 1)
    out = np.where(np.isnan(out), 0.5, out)
    return standard_scale(out)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    coords = u[:, :2] * s[:2]
    eig = (s**2) / max(x.shape[0] - 1, 1)
    explained = eig / eig.sum() if eig.sum() > 0 else np.zeros_like(eig)
    return coords, explained[:2]


def pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2 * a @ b.T, 0.0)


def knn_label_purity(x: np.ndarray, labels: np.ndarray, k: int = KNN_K, chunk: int = 512) -> float:
    purities = []
    n = x.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist = pairwise_sq(x[start:end], x)
        row_idx = np.arange(start, end)
        dist[np.arange(end - start), row_idx] = np.inf
        nn = np.argpartition(dist, kth=k, axis=1)[:, :k]
        purities.append((labels[nn] == labels[start:end, None]).mean(axis=1))
    return float(np.concatenate(purities).mean())


def silhouette_by_label(x: np.ndarray, labels: np.ndarray, sample_size: int = SILHOUETTE_SAMPLE) -> float:
    rng = np.random.default_rng(RANDOM_SEED)
    n = x.shape[0]
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    xs = x[idx]
    ys = labels[idx]
    dist = np.sqrt(pairwise_sq(xs, xs))
    scores = []
    for i, label in enumerate(ys):
        same = ys == label
        same[i] = False
        if same.sum() == 0:
            continue
        a = dist[i, same].mean()
        b = math.inf
        for other in np.unique(ys[ys != label]):
            other_mask = ys == other
            if other_mask.any():
                b = min(b, float(dist[i, other_mask].mean()))
        if math.isfinite(b) and max(a, b) > 0:
            scores.append((b - a) / max(a, b))
    return float(np.mean(scores)) if scores else float("nan")


def kmeans(x: np.ndarray, k: int, restarts: int = 8, iterations: int = 60) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    best_labels = None
    best_inertia = math.inf
    n = x.shape[0]
    for _ in range(restarts):
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iterations):
            dist = pairwise_sq(x, centers)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if mask.any():
                    centers[c] = x[mask].mean(axis=0)
                else:
                    centers[c] = x[rng.integers(0, n)]
        inertia = float(np.min(pairwise_sq(x, centers), axis=1).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    true_vals, true_inv = np.unique(labels_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((len(true_vals), len(pred_vals)), dtype=np.int64)
    np.add.at(table, (true_inv, pred_inv), 1)
    return table


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = contingency(labels_true, labels_pred)
    n = table.sum()
    if n < 2:
        return 0.0

    def comb2(arr):
        return arr * (arr - 1) / 2

    sum_comb = comb2(table).sum()
    row_comb = comb2(table.sum(axis=1)).sum()
    col_comb = comb2(table.sum(axis=0)).sum()
    total_comb = n * (n - 1) / 2
    expected = row_comb * col_comb / total_comb if total_comb else 0.0
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    return float((sum_comb - expected) / denom) if denom else 0.0


def normalized_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = contingency(labels_true, labels_pred).astype(float)
    n = table.sum()
    if n == 0:
        return 0.0
    pxy = table / n
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px[:, None] * py[None, :])[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    return mi / math.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0


def make_svg_scatter(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    explained: np.ndarray,
    output_file: Path,
    width: int = 920,
    height: int = 700,
) -> None:
    margin_left, margin_right, margin_top, margin_bottom = 70, 210, 55, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = np.percentile(x, [1, 99])
    ymin, ymax = np.percentile(y, [1, 99])
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    xs = margin_left + np.clip((x - xmin) / (xmax - xmin), 0, 1) * plot_w
    ys = margin_top + (1 - np.clip((y - ymin) / (ymax - ymin), 0, 1)) * plot_h

    label_order = [label for label in PALETTE if label in set(labels)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="32" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#fafafa" stroke="#cfcfcf"/>',
    ]
    for gx in np.linspace(margin_left, margin_left + plot_w, 6):
        parts.append(f'<line x1="{gx:.1f}" y1="{margin_top}" x2="{gx:.1f}" y2="{margin_top + plot_h}" stroke="#e5e5e5"/>')
    for gy in np.linspace(margin_top, margin_top + plot_h, 6):
        parts.append(f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{margin_left + plot_w}" y2="{gy:.1f}" stroke="#e5e5e5"/>')
    for label in label_order:
        mask = labels == label
        color = PALETTE.get(label, "#555555")
        for px, py in zip(xs[mask], ys[mask]):
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.3" fill="{color}" fill-opacity="0.55"/>')
    xlab = f"PC1 ({explained[0] * 100:.1f}%)"
    ylab = f"PC2 ({explained[1] * 100:.1f}%)"
    parts.append(f'<text x="{margin_left + plot_w / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="14">{xlab}</text>')
    parts.append(f'<text x="22" y="{margin_top + plot_h / 2}" transform="rotate(-90 22 {margin_top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="14">{ylab}</text>')
    lx = margin_left + plot_w + 35
    ly = margin_top + 15
    parts.append(f'<text x="{lx}" y="{ly}" font-family="Arial" font-size="14" font-weight="700">Type</text>')
    for i, label in enumerate(label_order):
        yy = ly + 26 + i * 24
        parts.append(f'<circle cx="{lx + 8}" cy="{yy - 5}" r="6" fill="{PALETTE[label]}"/>')
        parts.append(f'<text x="{lx + 24}" y="{yy}" font-family="Arial" font-size="13">{html.escape(label)}</text>')
    parts.append("</svg>")
    output_file.write_text("\n".join(parts), encoding="utf-8")


def sklearn_knn_label_purity(x: np.ndarray, labels: np.ndarray, k: int = KNN_K) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    indices = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((labels[indices] == labels[:, None]).mean())


def sampled_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    n = x.shape[0]
    sample_size = min(SILHOUETTE_SAMPLE, n)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(
        silhouette_score(
            x,
            labels,
            metric="euclidean",
            sample_size=sample_size,
            random_state=RANDOM_SEED,
        )
    )


def plot_embedding(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    output_stem: Path,
    axis_percentile: tuple[float, float] | None = None,
    duplicate_jitter: bool = True,
) -> None:
    plot_coords = coords.copy()
    rounded = np.round(plot_coords, 8)
    unique_coords = np.unique(rounded, axis=0).shape[0]
    jitter_note = ""
    if duplicate_jitter and unique_coords < max(200, int(0.05 * len(plot_coords))):
        rng = np.random.default_rng(RANDOM_SEED)
        x_span = np.ptp(plot_coords[:, 0]) or 1.0
        y_span = np.ptp(plot_coords[:, 1]) or 1.0
        plot_coords[:, 0] += rng.normal(0, x_span * 0.006, size=len(plot_coords))
        plot_coords[:, 1] += rng.normal(0, y_span * 0.006, size=len(plot_coords))
        jitter_note = "Small display jitter added because many samples share identical coordinates."

    plot_df = pd.DataFrame({"x": plot_coords[:, 0], "y": plot_coords[:, 1], "type": labels})
    present_order = [label for label in LABEL_ORDER if label in set(labels)]

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=180)
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="type",
        hue_order=present_order,
        palette={k: PALETTE[k] for k in present_order},
        s=12,
        linewidth=0,
        alpha=0.66,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if axis_percentile is not None:
        lo, hi = axis_percentile
        ax.set_xlim(np.percentile(plot_coords[:, 0], [lo, hi]))
        ax.set_ylim(np.percentile(plot_coords[:, 1], [lo, hi]))
    ax.margins(0.04)
    if jitter_note:
        ax.text(
            0.01,
            0.01,
            jitter_note,
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
            ha="left",
            va="bottom",
        )
    ax.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(PROCESSED_DIR / "pyrite_metadata.csv")
    values = pd.read_csv(PROCESSED_DIR / "pyrite_element_values.csv")
    observed = pd.read_csv(PROCESSED_DIR / "pyrite_observed_mask.csv")
    numeric = pd.read_csv(PROCESSED_DIR / "pyrite_numeric_mask.csv")

    labels = metadata["type"].fillna("<missing>").to_numpy()
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

    metrics = []
    cluster_rows = []
    k = len(np.unique(labels))
    for name, x in reps.items():
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        pca_coords = pca.fit_transform(x)
        pca_explained = pca.explained_variance_ratio_

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="euclidean",
            random_state=RANDOM_SEED,
        )
        umap_coords = reducer.fit_transform(x)

        cluster_labels = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit_predict(x)
        ari = adjusted_rand_score(labels, cluster_labels)
        nmi = normalized_mutual_info_score(labels, cluster_labels)

        pca_emb = pd.DataFrame(
            {
                "sample_id": metadata["sample_id"],
                "type": labels,
                "pc1": pca_coords[:, 0],
                "pc2": pca_coords[:, 1],
                "cluster": cluster_labels,
            }
        )
        umap_emb = pd.DataFrame(
            {
                "sample_id": metadata["sample_id"],
                "type": labels,
                "umap1": umap_coords[:, 0],
                "umap2": umap_coords[:, 1],
                "cluster": cluster_labels,
            }
        )
        safe_to_csv(pca_emb, EMBED_DIR / f"{name}_pca.csv")
        safe_to_csv(umap_emb, EMBED_DIR / f"{name}_umap.csv")

        plot_embedding(
            pca_coords,
            labels,
            title=f"{name} - PCA",
            x_label=f"PC1 ({pca_explained[0] * 100:.1f}%)",
            y_label=f"PC2 ({pca_explained[1] * 100:.1f}%)",
            output_stem=FIG_DIR / f"{name}_pca",
        )
        if name == "raw_robust_z":
            plot_embedding(
                pca_coords,
                labels,
                title=f"{name} - PCA central zoom",
                x_label=f"PC1 ({pca_explained[0] * 100:.1f}%)",
                y_label=f"PC2 ({pca_explained[1] * 100:.1f}%)",
                output_stem=FIG_DIR / f"{name}_pca_central_zoom",
                axis_percentile=(1, 99),
            )
        plot_embedding(
            umap_coords,
            labels,
            title=f"{name} - UMAP",
            x_label="UMAP1",
            y_label="UMAP2",
            output_stem=FIG_DIR / f"{name}_umap",
        )

        metrics.append(
            {
                "representation": name,
                "n_features": x.shape[1],
                "pc1_explained": pca_explained[0],
                "pc2_explained": pca_explained[1],
                "pc12_explained": pca_explained.sum(),
                "input_knn15_type_purity": sklearn_knn_label_purity(x, labels),
                "input_silhouette_type_sampled": sampled_silhouette(x, labels),
                "pca_knn15_type_purity": sklearn_knn_label_purity(pca_coords, labels),
                "pca_silhouette_type_sampled": sampled_silhouette(pca_coords, labels),
                "pca_trustworthiness": trustworthiness(x, pca_coords, n_neighbors=KNN_K),
                "umap_knn15_type_purity": sklearn_knn_label_purity(umap_coords, labels),
                "umap_silhouette_type_sampled": sampled_silhouette(umap_coords, labels),
                "umap_trustworthiness": trustworthiness(x, umap_coords, n_neighbors=KNN_K),
                "kmeans_k": k,
                "kmeans_ari_vs_type": ari,
                "kmeans_nmi_vs_type": nmi,
            }
        )
        for sample_id, cluster in zip(metadata["sample_id"], cluster_labels):
            cluster_rows.append({"representation": name, "sample_id": sample_id, "cluster": int(cluster)})

    metrics_df = pd.DataFrame(metrics).sort_values("umap_knn15_type_purity", ascending=False)
    metrics_path = safe_to_csv(metrics_df, RESULT_DIR / "representation_metrics.csv")
    clusters_path = safe_to_csv(pd.DataFrame(cluster_rows), RESULT_DIR / "representation_clusters.csv")

    selected_elements_path = safe_to_csv(
        pd.DataFrame(
        {
            "element": selected_elements,
            "numeric_coverage": numeric_coverage[selected_elements].to_numpy(),
        }
        ),
        RESULT_DIR / "selected_elements.csv",
    )

    lines = [
        "# Pyrite Representation Benchmark",
        "",
        "## Scope",
        "",
        f"- Samples: `{len(metadata)}`",
        f"- Grouping label: `type`",
        f"- Selected elements: `{len(selected_elements)}` with numeric coverage >= `{COVERAGE_THRESHOLD:.0%}`",
        f"- Elements: {', '.join(f'`{e}`' for e in selected_elements)}",
        "- PCA was computed with `sklearn.decomposition.PCA`.",
        f"- UMAP was computed with `n_neighbors={UMAP_N_NEIGHBORS}`, `min_dist={UMAP_MIN_DIST}`.",
        "",
        "## Representation Metrics",
        "",
        "| representation | features | PC1+PC2 | input kNN15 | PCA kNN15 | UMAP kNN15 | UMAP trust | kmeans ARI | kmeans NMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| `{row['representation']}` | {int(row['n_features'])} | "
            f"{row['pc12_explained']:.3f} | {row['input_knn15_type_purity']:.3f} | "
            f"{row['pca_knn15_type_purity']:.3f} | {row['umap_knn15_type_purity']:.3f} | "
            f"{row['umap_trustworthiness']:.3f} | {row['kmeans_ari_vs_type']:.3f} | "
            f"{row['kmeans_nmi_vs_type']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## First Reading",
            "",
            "- `input kNN15` asks whether local neighborhoods in the full representation space preserve `type` labels.",
            "- `PCA kNN15` and `UMAP kNN15` ask the same question after 2D projection.",
            "- `trustworthiness` measures how well the 2D embedding preserves local neighborhoods from the full representation.",
            "- `kmeans ARI/NMI` asks whether unsupervised clusters resemble the known `type` labels.",
            "- The observation-mask representation is included because missingness and censoring can themselves create structure.",
            "",
            "## Outputs",
            "",
            f"- `{metrics_path.name}`: benchmark metrics",
            f"- `{selected_elements_path.name}`: elements used in this first pass",
            f"- `{clusters_path.name}`: sample clusters in the full representation space",
            "- `embeddings/*_pca.csv`: sample-level PCA coordinates and clusters",
            "- `embeddings/*_umap.csv`: sample-level UMAP coordinates and clusters",
            "- `figures/*_pca.png` and `figures/*_pca.svg`: PCA scatter plots colored by `type`",
            "- `figures/*_umap.png` and `figures/*_umap.svg`: UMAP scatter plots colored by `type`",
            "- `figures/raw_robust_z_pca_central_zoom.*`: percentile zoom for the raw PCA plot dominated by extreme values",
            "",
        ]
    )
    (RESULT_DIR / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
