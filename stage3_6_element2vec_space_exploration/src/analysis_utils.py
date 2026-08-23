from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/element2vec-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def cosine_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    xn = row_normalize(x)
    yn = xn if y is None else row_normalize(y)
    return xn @ yn.T


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 4 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return float("nan"), float("nan")
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def permutation_pvalue(observed: float, null_values: np.ndarray, two_sided: bool = False) -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not len(null_values) or not np.isfinite(observed):
        return float("nan")
    if two_sided:
        hits = np.sum(np.abs(null_values) >= abs(observed))
    else:
        hits = np.sum(null_values >= observed)
    return float((hits + 1) / (len(null_values) + 1))


def knn_indices(x: np.ndarray, k: int = 15, metric: str = "cosine") -> np.ndarray:
    n_neighbors = min(k + 1, len(x))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(x)
    indices = model.kneighbors(return_distance=False)
    return indices[:, 1:]


def knn_purity(x: np.ndarray, labels: np.ndarray, k: int = 15, metric: str = "cosine") -> float:
    labels = np.asarray(labels)
    idx = knn_indices(x, k=k, metric=metric)
    return float(np.mean(labels[idx] == labels[:, None]))


def continuous_neighbor_smoothness(x: np.ndarray, values: np.ndarray, k: int = 15) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < k + 2:
        return float("nan")
    xv = x[valid]
    yv = values[valid]
    idx = knn_indices(xv, k=k)
    neighbor_mean = np.mean(yv[idx], axis=1)
    return safe_spearman(yv, neighbor_mean)[0]


def cv_ridge_r2(x: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 42) -> float:
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if valid.sum() < max(20, folds * 4):
        return float("nan")
    xv, yv = x[valid], y[valid]
    cv = KFold(n_splits=min(folds, len(yv)), shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    pred = cross_val_predict(model, xv, yv, cv=cv, n_jobs=1)
    return float(r2_score(yv, pred))


def cv_balanced_accuracy(x: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 42) -> float:
    labels = np.asarray(labels).astype(str)
    valid = pd.Series(labels).notna().to_numpy() & (labels != "") & (labels != "nan")
    xv, lv = x[valid], labels[valid]
    if len(np.unique(lv)) < 2:
        return float("nan")
    encoder = LabelEncoder()
    y = encoder.fit_transform(lv)
    counts = np.bincount(y)
    n_splits = min(folds, int(counts.min()))
    if n_splits < 2:
        return float("nan")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, xv, y, cv=cv, n_jobs=1)
    return float(balanced_accuracy_score(y, pred))


def pca_frame(x: np.ndarray, n_components: int = 3, prefix: str = "PC") -> tuple[PCA, pd.DataFrame]:
    n = min(n_components, x.shape[0], x.shape[1])
    pca = PCA(n_components=n, random_state=42)
    coords = pca.fit_transform(StandardScaler().fit_transform(x))
    return pca, pd.DataFrame(coords, columns=[f"{prefix}{i+1}" for i in range(n)])


def plot_scatter(
    coords: pd.DataFrame,
    color: pd.Series | np.ndarray,
    path: Path,
    title: str,
    categorical: bool = False,
    dpi: int = 180,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if categorical:
        labels = pd.Series(color).fillna("[MISSING]").astype(str)
        classes = labels.value_counts().head(12).index
        for cls in classes:
            mask = labels.eq(cls).to_numpy()
            ax.scatter(coords.loc[mask, coords.columns[0]], coords.loc[mask, coords.columns[1]], s=8, alpha=0.45, label=cls)
        ax.legend(fontsize=7, markerscale=2, loc="best")
    else:
        values = pd.to_numeric(pd.Series(color), errors="coerce").to_numpy()
        scatter = ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], c=values, s=7, alpha=0.5, cmap="viridis")
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    ax.set_xlabel(coords.columns[0])
    ax.set_ylabel(coords.columns[1])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, path: Path, title: str, dpi: int = 180, cmap: str = "coolwarm") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max(7, min(18, 0.55 * len(matrix.columns) + 3))
    height = max(5, min(18, 0.45 * len(matrix.index) + 2))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def effect_status(effect: float, pvalue: float | None = None) -> str:
    if not np.isfinite(effect):
        return "Negative"
    magnitude = abs(effect)
    significant = pvalue is None or (np.isfinite(pvalue) and pvalue <= 0.05)
    if magnitude >= 0.30 and significant:
        return "Strong positive"
    if magnitude >= 0.15 and significant:
        return "Moderate positive"
    if magnitude >= 0.05:
        return "Weak/ambiguous"
    return "Negative"


def group_distance_effect(x: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    labels = np.asarray(labels).astype(str)
    dist = squareform(pdist(row_normalize(x), metric="cosine"))
    same = labels[:, None] == labels[None, :]
    tri = np.triu(np.ones_like(same, dtype=bool), k=1)
    within = dist[same & tri]
    between = dist[(~same) & tri]
    if not len(within) or not len(between):
        return float("nan"), float("nan"), float("nan")
    pooled = math.sqrt((within.var() + between.var()) / 2.0)
    effect = (between.mean() - within.mean()) / pooled if pooled > 0 else float("nan")
    return float(effect), float(within.mean()), float(between.mean())


def centroid_by_label(x: np.ndarray, labels: pd.Series, min_count: int = 5) -> tuple[np.ndarray, list[str], np.ndarray]:
    labels = labels.fillna("[MISSING]").astype(str).reset_index(drop=True)
    rows, names, counts = [], [], []
    for name, indices in labels.groupby(labels).groups.items():
        idx = np.asarray(list(indices), dtype=int)
        if len(idx) >= min_count:
            rows.append(x[idx].mean(axis=0))
            names.append(name)
            counts.append(len(idx))
    return np.stack(rows), names, np.asarray(counts)
