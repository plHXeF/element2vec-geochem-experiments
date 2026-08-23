from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 2000


def finite_or_nan(value: float | None) -> float:
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def as_array(frame: pd.DataFrame) -> np.ndarray:
    x = frame.to_numpy(dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def knn_label_purity(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(labels) <= k:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(x)
    idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


def safe_silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return float(
            silhouette_score(
                x,
                labels,
                metric="euclidean",
                sample_size=min(SILHOUETTE_SAMPLE, len(labels)),
                random_state=RANDOM_STATE,
            )
        )
    except Exception:
        return float("nan")


def safe_davies_bouldin(x: np.ndarray, clusters: np.ndarray) -> float:
    if len(np.unique(clusters)) < 2:
        return float("nan")
    try:
        return finite_or_nan(davies_bouldin_score(x, clusters))
    except Exception:
        return float("nan")


def safe_calinski_harabasz(x: np.ndarray, clusters: np.ndarray) -> float:
    if len(np.unique(clusters)) < 2:
        return float("nan")
    try:
        return finite_or_nan(calinski_harabasz_score(x, clusters))
    except Exception:
        return float("nan")


def kmeans_labels(x: np.ndarray, n_clusters: int) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, n_init=20, random_state=RANDOM_STATE).fit_predict(x)


def gmm_labels(x: np.ndarray, n_clusters: int) -> np.ndarray:
    model = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    return model.fit_predict(x)


def hdbscan_metrics(x: np.ndarray, labels: np.ndarray) -> dict:
    try:
        import hdbscan  # type: ignore
    except Exception as exc:
        return {
            "hdbscan_cluster_count": float("nan"),
            "hdbscan_ari": float("nan"),
            "hdbscan_nmi": float("nan"),
            "hdbscan_noise_ratio": float("nan"),
            "hdbscan_note": f"skipped: {type(exc).__name__}: {exc}",
        }
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=15)
        clusters = clusterer.fit_predict(x)
        non_noise = clusters != -1
        cluster_count = len(set(clusters[non_noise]))
        noise_ratio = float((clusters == -1).mean())
        if cluster_count >= 2:
            ari = adjusted_rand_score(labels, clusters)
            nmi = normalized_mutual_info_score(labels, clusters)
        else:
            ari = float("nan")
            nmi = float("nan")
        return {
            "hdbscan_cluster_count": int(cluster_count),
            "hdbscan_ari": finite_or_nan(ari),
            "hdbscan_nmi": finite_or_nan(nmi),
            "hdbscan_noise_ratio": finite_or_nan(noise_ratio),
            "hdbscan_note": "ok",
        }
    except Exception as exc:
        return {
            "hdbscan_cluster_count": float("nan"),
            "hdbscan_ari": float("nan"),
            "hdbscan_nmi": float("nan"),
            "hdbscan_noise_ratio": float("nan"),
            "hdbscan_note": f"skipped: {type(exc).__name__}: {exc}",
        }


def unsupervised_metrics_for(
    representation: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
) -> dict:
    x = as_array(frame)
    n_clusters = len(np.unique(labels))
    km = kmeans_labels(x, n_clusters)
    gm = gmm_labels(x, n_clusters)
    hdb = hdbscan_metrics(x, labels)
    row = {
        "representation": representation,
        "n_features": x.shape[1],
        "input_knn15": knn_label_purity(x, labels, 15),
        "input_knn30": knn_label_purity(x, labels, 30),
        "kmeans_ari": finite_or_nan(adjusted_rand_score(labels, km)),
        "kmeans_nmi": finite_or_nan(normalized_mutual_info_score(labels, km)),
        "gmm_ari": finite_or_nan(adjusted_rand_score(labels, gm)),
        "gmm_nmi": finite_or_nan(normalized_mutual_info_score(labels, gm)),
        "silhouette": safe_silhouette(x, labels),
        "davies_bouldin": safe_davies_bouldin(x, km),
        "calinski_harabasz": safe_calinski_harabasz(x, km),
    }
    row.update(hdb)
    return row


def run_unsupervised_metrics(reps: dict[str, pd.DataFrame], labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, frame in reps.items():
        print(f"[unsupervised] {name}")
        rows.append(unsupervised_metrics_for(name, frame, labels))
    return pd.DataFrame(rows)


def pca_sweep_for(
    representation: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    max_components: int = 12,
) -> pd.DataFrame:
    x = as_array(frame)
    dims = range(1, min(max_components, x.shape[1]) + 1)
    pca = PCA(n_components=max(dims), random_state=RANDOM_STATE)
    latent_all = pca.fit_transform(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_clusters = len(np.unique(labels))
    rows = []
    for dim in dims:
        latent = latent_all[:, :dim]
        km = kmeans_labels(latent, n_clusters)
        gm = gmm_labels(latent, n_clusters)
        rows.append(
            {
                "representation": representation,
                "dim": dim,
                "n_features": x.shape[1],
                "cumulative_variance": finite_or_nan(cumulative[dim - 1]),
                "pca_knn15": knn_label_purity(latent, labels, 15),
                "pca_knn30": knn_label_purity(latent, labels, 30),
                "kmeans_ari": finite_or_nan(adjusted_rand_score(labels, km)),
                "kmeans_nmi": finite_or_nan(normalized_mutual_info_score(labels, km)),
                "gmm_ari": finite_or_nan(adjusted_rand_score(labels, gm)),
                "gmm_nmi": finite_or_nan(normalized_mutual_info_score(labels, gm)),
            }
        )
    return pd.DataFrame(rows)


def run_pca_sweep(
    reps: dict[str, pd.DataFrame],
    labels: np.ndarray,
    max_components: int = 12,
) -> pd.DataFrame:
    parts = []
    for name, frame in reps.items():
        print(f"[pca sweep] {name}")
        parts.append(pca_sweep_for(name, frame, labels, max_components=max_components))
    return pd.concat(parts, ignore_index=True)


def run_umap3d(
    reps: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    labels: np.ndarray,
    out_dir: Path,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import umap  # type: ignore
    except Exception as exc:
        notes = pd.DataFrame(
            [
                {
                    "representation": name,
                    "trustworthiness": float("nan"),
                    "umap_knn15": float("nan"),
                    "umap_knn30": float("nan"),
                    "note": f"skipped: {type(exc).__name__}: {exc}",
                }
                for name in reps
            ]
        )
        return pd.DataFrame(), notes

    coords_parts = []
    metric_rows = []
    for name, frame in reps.items():
        print(f"[umap3d] {name}")
        x = as_array(frame)
        if x.shape[1] < 3:
            metric_rows.append(
                {
                    "representation": name,
                    "trustworthiness": float("nan"),
                    "umap_knn15": float("nan"),
                    "umap_knn30": float("nan"),
                    "note": "skipped: n_features < 3",
                }
            )
            continue
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=random_state,
        )
        coords = reducer.fit_transform(x)
        trust = trustworthiness(x, coords, n_neighbors=15)
        metric_rows.append(
            {
                "representation": name,
                "trustworthiness": finite_or_nan(trust),
                "umap_knn15": knn_label_purity(coords, labels, 15),
                "umap_knn30": knn_label_purity(coords, labels, 30),
                "note": "ok",
            }
        )
        coords_parts.append(
            pd.DataFrame(
                {
                    "sample_id": metadata["sample_id"].astype(str).to_numpy(),
                    "type": labels,
                    "representation": name,
                    "UMAP1": coords[:, 0],
                    "UMAP2": coords[:, 1],
                    "UMAP3": coords[:, 2],
                }
            )
        )
    coords_df = pd.concat(coords_parts, ignore_index=True) if coords_parts else pd.DataFrame()
    metrics_df = pd.DataFrame(metric_rows)
    return coords_df, metrics_df
