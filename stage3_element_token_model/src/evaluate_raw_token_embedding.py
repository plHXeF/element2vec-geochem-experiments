from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import OUTPUT_DIR, STAGE_DIR, load_config, markdown_table, save_dataframe, write_json


OUT_DIR = OUTPUT_DIR / "raw_embedding_benchmark"
FIG_DIR = OUT_DIR / "figures"
METRIC_DIR = OUT_DIR / "metrics"
SPACE_NAME = "sample_raw_element_token_flat_476"


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"sample_id", "deposit_type"}]


def make_xgb_or_fallback(config: dict):
    try:
        from xgboost import XGBClassifier  # type: ignore

        return (
            XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                n_estimators=int(config.get("xgb_n_estimators", 150)),
                max_depth=int(config.get("xgb_max_depth", 3)),
                learning_rate=float(config.get("xgb_learning_rate", 0.05)),
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=int(config.get("random_seed", 42)),
                n_jobs=4,
                verbosity=0,
            ),
            "xgboost",
            "ok",
        )
    except Exception as exc:
        return (
            HistGradientBoostingClassifier(max_iter=200, random_state=int(config.get("random_seed", 42))),
            "hist_gradient_boosting_fallback",
            f"xgboost unavailable: {type(exc).__name__}: {exc}",
        )


def cv_metrics(model_factory, x: np.ndarray, y: np.ndarray, config: dict, prefix: str) -> tuple[dict, list[dict], str, str]:
    skf = StratifiedKFold(n_splits=int(config.get("cv_splits", 5)), shuffle=True, random_state=int(config.get("random_seed", 42)))
    rows = []
    model_name = ""
    note = ""
    for fold, (train_idx, test_idx) in enumerate(skf.split(x, y), start=1):
        model, model_name, note = model_factory()
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])
        rows.append(
            {
                "fold": fold,
                f"{prefix}_accuracy": accuracy_score(y[test_idx], pred),
                f"{prefix}_macro_f1": f1_score(y[test_idx], pred, average="macro"),
                f"{prefix}_weighted_f1": f1_score(y[test_idx], pred, average="weighted"),
            }
        )
    df = pd.DataFrame(rows)
    summary = {}
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        col = f"{prefix}_{metric}"
        summary[f"{prefix}_{metric}_mean"] = float(df[col].mean())
        summary[f"{prefix}_{metric}_std"] = float(df[col].std(ddof=1))
    return summary, rows, model_name, note


def knn_purity(x_scaled: np.ndarray, labels: np.ndarray, k: int = 15) -> tuple[float, pd.DataFrame]:
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(x_scaled)
    idx = nn.kneighbors(x_scaled, return_distance=False)[:, 1:]
    purity = (labels[idx] == labels[:, None]).mean(axis=1)
    by_class = pd.DataFrame({"deposit_type": labels, "purity": purity}).groupby("deposit_type", as_index=False)["purity"].mean()
    return float(purity.mean()), by_class


def cluster_metrics(x_scaled: np.ndarray, y: np.ndarray, n_classes: int, seed: int) -> dict:
    clusters = KMeans(n_clusters=n_classes, n_init=20, random_state=seed).fit_predict(x_scaled)
    try:
        sil = silhouette_score(x_scaled, clusters, sample_size=min(2000, len(y)), random_state=seed)
    except Exception:
        sil = float("nan")
    return {
        "kmeans_ari": float(adjusted_rand_score(y, clusters)),
        "kmeans_nmi": float(normalized_mutual_info_score(y, clusters)),
        "kmeans_homogeneity": float(homogeneity_score(y, clusters)),
        "kmeans_completeness": float(completeness_score(y, clusters)),
        "silhouette_score": float(sil),
    }


def compute_3d(x_scaled: np.ndarray, labels: np.ndarray, sample_id: np.ndarray, name: str, config: dict) -> tuple[pd.DataFrame, str]:
    seed = int(config.get("random_seed", 42))
    try:
        import umap  # type: ignore

        n_neighbors = min(int(config.get("umap_n_neighbors", 30)), max(2, len(labels) - 1))
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=float(config.get("umap_min_dist", 0.1)),
            random_state=seed,
        )
        coords = reducer.fit_transform(x_scaled)
        method = "umap"
    except Exception as exc:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=min(3, x_scaled.shape[1]), random_state=seed).fit_transform(x_scaled)
        if coords.shape[1] < 3:
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), mode="constant")
        method = f"pca_fallback: {type(exc).__name__}: {exc}"

    out = pd.DataFrame(
        {
            "sample_id": sample_id.astype(str),
            "deposit_type": labels.astype(str),
            "space_name": name,
            "dim1": coords[:, 0],
            "dim2": coords[:, 1],
            "dim3": coords[:, 2],
        }
    )
    return out, method


def save_3d_plots(coords: pd.DataFrame, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.2, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    for label, group in coords.groupby("deposit_type", sort=False):
        ax.scatter(group["dim1"], group["dim2"], group["dim3"], s=12, alpha=0.8, label=label)
    ax.set_title(name)
    ax.set_xlabel("dim1")
    ax.set_ylabel("dim2")
    ax.set_zlabel("dim3")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / f"{name}_umap3d.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}_umap3d.svg", bbox_inches="tight")
    plt.close(fig)

    try:
        import plotly.express as px  # type: ignore

        plot = px.scatter_3d(coords, x="dim1", y="dim2", z="dim3", color="deposit_type", hover_data=["sample_id"], title=name)
        plot.write_html(FIG_DIR / f"{name}_umap3d.html")
    except Exception:
        traces = []
        for label, group in coords.groupby("deposit_type", sort=False):
            traces.append(
                {
                    "type": "scatter3d",
                    "mode": "markers",
                    "name": str(label),
                    "x": group["dim1"].round(6).tolist(),
                    "y": group["dim2"].round(6).tolist(),
                    "z": group["dim3"].round(6).tolist(),
                    "text": group["sample_id"].astype(str).tolist(),
                    "marker": {"size": 4, "opacity": 0.8},
                }
            )
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{name}</title></head>
<body><div id="plot" style="width:100%;height:92vh;"></div>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>Plotly.newPlot('plot', {json.dumps(traces)}, {{title: {json.dumps(name)}}});</script>
</body></html>"""
        (FIG_DIR / f"{name}_umap3d.html").write_text(html, encoding="utf-8")


def evaluate_raw_token(config: dict) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "embeddings" / "sample_raw_element_token_flat.csv"
    df = pd.read_csv(raw_path)
    cols = feature_cols(df)
    x = df[cols].to_numpy(dtype=np.float32)
    labels = df["deposit_type"].fillna("<missing>").astype(str).to_numpy()
    sample_id = df["sample_id"].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(labels)
    x_scaled = StandardScaler().fit_transform(x)
    notes = []

    xgb_summary, xgb_folds, xgb_model, xgb_note = cv_metrics(lambda: make_xgb_or_fallback(config), x, y, config, "xgb")
    if xgb_note != "ok":
        notes.append(xgb_note)

    def knn_factory():
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(config.get("knn_k", 15)))), "knn15", "ok"

    knn_summary, knn_folds, _, _ = cv_metrics(knn_factory, x, y, config, "knn15")
    purity, by_class = knn_purity(x_scaled, labels, k=int(config.get("knn_k", 15)))
    clusters = cluster_metrics(x_scaled, y, len(le.classes_), int(config.get("random_seed", 42)))
    coords, method = compute_3d(x_scaled, labels, sample_id, SPACE_NAME, config)
    if method != "umap":
        notes.append(method)
    save_dataframe(coords, FIG_DIR / f"{SPACE_NAME}_umap3d_coordinates.csv")
    save_3d_plots(coords, SPACE_NAME)

    row = {
        "space_name": SPACE_NAME,
        "n_features": int(x.shape[1]),
        **xgb_summary,
        **knn_summary,
        "knn15_neighborhood_purity": purity,
        **clusters,
        "notes": "; ".join(notes) if notes else "ok",
    }
    folds = []
    for r in xgb_folds:
        folds.append({"space_name": SPACE_NAME, "model": xgb_model, **r})
    for r in knn_folds:
        folds.append({"space_name": SPACE_NAME, "model": "knn15", **r})

    summary = pd.DataFrame([row])
    save_dataframe(summary, METRIC_DIR / "raw_element_token_benchmark_summary.csv")
    save_dataframe(pd.DataFrame(folds), METRIC_DIR / "raw_element_token_cv_fold_metrics.csv")
    by_class.insert(0, "space_name", SPACE_NAME)
    save_dataframe(by_class, METRIC_DIR / "raw_element_token_knn15_purity_by_class.csv")
    write_json(summary.to_dict(orient="records"), METRIC_DIR / "raw_element_token_benchmark_summary.json")
    return summary


def deposit_embedding_umap(config: dict) -> None:
    path = OUT_DIR / "embeddings" / "deposit_type_embedding_nomask.csv"
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c != "deposit_type"]
    x = df[cols].to_numpy(dtype=np.float32)
    labels = df["deposit_type"].astype(str).to_numpy()
    x_scaled = StandardScaler().fit_transform(x)
    coords, method = compute_3d(x_scaled, labels, labels, "deposit_type_embedding_nomask", config)
    save_dataframe(coords, FIG_DIR / "deposit_type_embedding_nomask_umap3d_coordinates.csv")
    save_3d_plots(coords, "deposit_type_embedding_nomask")


def write_report(summary: pd.DataFrame) -> None:
    audit_path = OUT_DIR / "audit" / "embedding_learning_audit.md"
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.exists() else "Audit file not found."
    report = f"""# Raw Element Token Embedding Benchmark

## What Was Added

This benchmark evaluates the unprojected sample-level token matrix:

`[x_i, visible_mask_i, learned_element_embedding_i]`

for each of the 14 elements. The resulting feature dimension is:

`14 * (1 + 1 + 32) = 476`

This is different from the previous `sample_input_token_flat.csv`, which used the projected Transformer input token:

`14 * d_model = 14 * 64 = 896`

## Benchmark Summary

{markdown_table(summary)}

## Outputs

- `embeddings/sample_raw_element_token_flat.csv`
- `embeddings/deposit_type_embedding_nomask.csv`
- `metrics/raw_element_token_benchmark_summary.csv`
- `figures/sample_raw_element_token_flat_476_umap3d.html`
- `figures/deposit_type_embedding_nomask_umap3d.html`

## Embedding Learning Audit

{audit_text}
"""
    (OUT_DIR / "raw_embedding_benchmark_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    config = load_config(args.config)
    summary = evaluate_raw_token(config)
    deposit_embedding_umap(config)
    write_report(summary)
    print(f"[raw token benchmark] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
