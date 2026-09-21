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

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

from utils import OUTPUT_DIR, STAGE_DIR, ensure_output_dirs, load_config, markdown_table, save_dataframe, write_json
from visualize import make_3d_visualization


SPACE_ORDER = [
    "previous_baseline_space",
    "log_robust_space",
    "log_robust_pca12_space",
    "sample_input_token_flat_pca12_space",
    "sample_encoded_token_flat_pca12_space",
    "transformer_latent_label_masked_space",
]


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in {"sample_id", "deposit_type"}]


def _make_xgb_or_fallback(config: dict):
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


def _cv_metrics(model_factory, x: np.ndarray, y: np.ndarray, labels: np.ndarray, config: dict, prefix: str) -> tuple[dict, list[dict], str, str]:
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


def _knn_purity(x_scaled: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, pd.DataFrame]:
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(x_scaled)
    idx = nn.kneighbors(x_scaled, return_distance=False)[:, 1:]
    same = labels[idx] == labels[:, None]
    purity_by_sample = same.mean(axis=1)
    by_class = pd.DataFrame({"deposit_type": labels, "purity": purity_by_sample}).groupby("deposit_type", as_index=False)["purity"].mean()
    return float(purity_by_sample.mean()), by_class


def _cluster_metrics(x_scaled: np.ndarray, y: np.ndarray, n_classes: int, config: dict) -> dict:
    clusters = KMeans(n_clusters=n_classes, n_init=20, random_state=int(config.get("random_seed", 42))).fit_predict(x_scaled)
    if len(np.unique(clusters)) > 1:
        try:
            sil = silhouette_score(
                x_scaled,
                clusters,
                sample_size=min(2000, len(clusters)),
                random_state=int(config.get("random_seed", 42)),
            )
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")
    return {
        "kmeans_ari": float(adjusted_rand_score(y, clusters)),
        "kmeans_nmi": float(normalized_mutual_info_score(y, clusters)),
        "kmeans_homogeneity": float(homogeneity_score(y, clusters)),
        "kmeans_completeness": float(completeness_score(y, clusters)),
        "silhouette_score": float(sil),
    }


def evaluate_space(space_name: str, path: Path, config: dict) -> tuple[dict, list[dict], pd.DataFrame]:
    df = pd.read_csv(path)
    cols = _feature_cols(df)
    x = df[cols].to_numpy(dtype=float)
    labels_text = df["deposit_type"].fillna("<missing>").astype(str).to_numpy()
    sample_id = df["sample_id"].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(labels_text)
    x_scaled = StandardScaler().fit_transform(x)
    notes = []

    def xgb_factory():
        return _make_xgb_or_fallback(config)

    xgb_summary, xgb_rows, model_name, xgb_note = _cv_metrics(xgb_factory, x, y, labels_text, config, "xgb")
    if xgb_note != "ok":
        notes.append(xgb_note)

    def knn_factory():
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=int(config.get("knn_k", 15)))), "knn15", "ok"

    knn_summary, knn_rows, _, _ = _cv_metrics(knn_factory, x, y, labels_text, config, "knn15")
    purity, by_class = _knn_purity(x_scaled, labels_text, int(config.get("knn_k", 15)))
    by_class.insert(0, "space_name", space_name)
    clusters = _cluster_metrics(x_scaled, y, len(le.classes_), config)
    vis = make_3d_visualization(space_name, x_scaled, sample_id, labels_text, config)
    if vis["visualization_note"] != "ok":
        notes.append(vis["visualization_note"])

    row = {
        "space_name": space_name,
        "n_features": int(x.shape[1]),
        **xgb_summary,
        **knn_summary,
        "knn15_neighborhood_purity": purity,
        **clusters,
        "notes": "; ".join(notes) if notes else "ok",
    }
    fold_rows = []
    for r in xgb_rows:
        fold_rows.append({"space_name": space_name, "model": model_name, **r})
    for r in knn_rows:
        fold_rows.append({"space_name": space_name, "model": "knn15", **r})
    return row, fold_rows, by_class


def write_report(summary: pd.DataFrame, config: dict) -> None:
    ranked = summary.sort_values("knn15_neighborhood_purity", ascending=False)
    clf_rank = summary.sort_values("xgb_macro_f1_mean", ascending=False)
    report = f"""# Stage 3 Element Token Transformer Report

## Scope

This stage trains a pyrite geochemistry element-token Transformer on the full processed dataset. It is an exploratory benchmark and sanity check, not a strict generalization estimate.

Main benchmark latent is `label_masked + mean_element`, so the deposit type token is forced to `[MASK_DEPOSIT]` during latent extraction.

## Spaces Evaluated

{markdown_table(summary[["space_name", "n_features", "knn15_neighborhood_purity", "xgb_macro_f1_mean", "knn15_macro_f1_mean", "kmeans_ari", "kmeans_nmi", "notes"]])}

## Best Local Neighborhood Structure

{markdown_table(ranked[["space_name", "knn15_neighborhood_purity", "kmeans_ari", "kmeans_nmi", "notes"]], max_rows=10)}

## Classification Sanity Check

The classification metrics use Stratified 5-fold CV. Because the Transformer was trained on all samples, these scores are not a clean generalization claim. They are only used to compare representation spaces.

{markdown_table(clf_rank[["space_name", "xgb_accuracy_mean", "xgb_macro_f1_mean", "xgb_weighted_f1_mean", "knn15_accuracy_mean", "knn15_macro_f1_mean"]], max_rows=10)}

## Outputs

- `outputs/embeddings/element_embedding.csv`: learned element identity embeddings.
- `outputs/embeddings/deposit_embedding.csv`: learned deposit type embeddings plus `[MASK_DEPOSIT]`.
- `outputs/embeddings/sample_input_token_flat.csv`: sample-specific projected element token matrix, flattened.
- `outputs/embeddings/sample_encoded_token_flat.csv`: sample-specific Transformer-encoded element token matrix, flattened.
- `outputs/latents/transformer_latent_label_masked_mean_element.csv`: main label-masked sample latent.
- `outputs/metrics/all_space_benchmark_summary.csv`: main summary table.
- `outputs/figures/*_3d.html`: 3D visualizations. If Plotly package was unavailable, HTML uses a CDN fallback.

## Notes

- If `xgboost` is unavailable in the active environment, evaluation falls back to `HistGradientBoostingClassifier`.
- If `umap-learn` is unavailable, 3D visualization falls back to PCA-3D and records this in `notes`.
- Do not use `label_visible` latent as the main result, because it can leak the target label through the deposit token.
"""
    (OUTPUT_DIR / "stage3_report.md").write_text(report, encoding="utf-8")


def evaluate_spaces(config: dict) -> pd.DataFrame:
    ensure_output_dirs()
    rows = []
    folds = []
    purity_parts = []
    for space_name in SPACE_ORDER:
        path = OUTPUT_DIR / "spaces" / f"{space_name}.csv"
        if not path.exists():
            if space_name == "previous_baseline_space":
                rows.append({"space_name": space_name, "n_features": 0, "notes": "previous baseline not found"})
                continue
            raise FileNotFoundError(path)
        print(f"[evaluate] {space_name}")
        row, fold_rows, by_class = evaluate_space(space_name, path, config)
        rows.append(row)
        folds.extend(fold_rows)
        purity_parts.append(by_class)

    summary = pd.DataFrame(rows)
    save_dataframe(summary, OUTPUT_DIR / "metrics" / "all_space_benchmark_summary.csv")
    write_json(summary.to_dict(orient="records"), OUTPUT_DIR / "metrics" / "all_space_benchmark_summary.json")
    save_dataframe(pd.DataFrame(folds), OUTPUT_DIR / "metrics" / "cv_fold_metrics.csv")
    if purity_parts:
        save_dataframe(pd.concat(purity_parts, ignore_index=True), OUTPUT_DIR / "metrics" / "knn15_neighborhood_purity_by_class.csv")
    write_report(summary, config)
    print(f"[evaluate] wrote {OUTPUT_DIR / 'metrics' / 'all_space_benchmark_summary.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    evaluate_spaces(load_config(args.config))


if __name__ == "__main__":
    main()
