from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..utils.io import ensure_output_dirs, load_config, output_dir, resolve_path


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("latent_")]


def knn_purity(x: np.ndarray, y: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(x))).fit(x)
    _, idx = nn.kneighbors(x)
    idx = idx[:, 1:]
    return float(np.mean([(y[neighbors] == y[i]).mean() for i, neighbors in enumerate(idx)]))


def classify(x: np.ndarray, y: np.ndarray) -> dict:
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return {"accuracy": np.nan, "macro_f1": np.nan}
    n_splits = int(min(5, counts.min()))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, x, y, cv=cv)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def make_projection(x: np.ndarray, out_path: Path) -> None:
    try:
        import umap

        coords = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(x)
        method = "umap3"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=min(3, x.shape[1]), random_state=42).fit_transform(x)
        method = "pca3"
    cols = {f"{method}_{i}": coords[:, i] for i in range(coords.shape[1])}
    pd.DataFrame(cols).to_csv(out_path, index=False, encoding="utf-8-sig")


def benchmark(config: dict, latent_path: str | Path | None = None) -> pd.DataFrame:
    ensure_output_dirs(config)
    out = output_dir(config)
    path = resolve_path(latent_path) if latent_path else out / "latents" / "sample_cls_latent.csv"
    df = pd.read_csv(path)
    feat_cols = _feature_columns(df)
    x = df[feat_cols].to_numpy(dtype=float)
    rows = []
    for label in ["rock_group", "rock_origin", "rock_type"]:
        valid = df[label].notna() & df[label].astype(str).str.len().gt(0)
        if valid.sum() < 10:
            continue
        y_text = df.loc[valid, label].astype(str).to_numpy()
        y = LabelEncoder().fit_transform(y_text)
        xv = x[valid.to_numpy()]
        cls = classify(xv, y)
        kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init="auto").fit(xv)
        rows.append(
            {
                "latent_file": str(path),
                "label": label,
                "n_samples": int(len(y)),
                "n_classes": int(len(np.unique(y))),
                "classification_accuracy": cls["accuracy"],
                "classification_macro_f1": cls["macro_f1"],
                "knn_purity_15": knn_purity(xv, y, 15),
                "knn_purity_30": knn_purity(xv, y, 30),
                "kmeans_ari": float(adjusted_rand_score(y, kmeans.labels_)),
                "kmeans_nmi": float(normalized_mutual_info_score(y, kmeans.labels_)),
            }
        )
    summary = pd.DataFrame(rows)
    out_path = out / "metrics" / "v2_benchmark_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    make_projection(x, out / "metrics" / "v2_latent_3d_projection.csv")
    print(f"[benchmark_latents] wrote {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/v2_base.yaml")
    parser.add_argument("--latent", default=None)
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    benchmark(load_config(args.config), args.latent)


if __name__ == "__main__":
    main()
