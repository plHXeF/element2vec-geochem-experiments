from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from metrics import run_pca_sweep, run_umap3d, run_unsupervised_metrics  # noqa: E402
from representations import build_representations, save_representations  # noqa: E402
from visualization import (  # noqa: E402
    plot_feature_importance,
    plot_pca_sweep,
    plot_umap_projections,
    plot_unsupervised_summary,
    save_plotly_umap,
)
from xgb_analysis import feature_importance, run_classification  # noqa: E402


PROCESSED_DIR = WORKSPACE / "pyrite_benchmark" / "data" / "processed"
OUT_DIR = ROOT / "outputs"


def load_config() -> dict:
    with open(ROOT / "configs" / "stage2_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def summarize_top(df: pd.DataFrame, metric: str, n: int = 6) -> str:
    if df.empty or metric not in df:
        return "- not available\n"
    rows = []
    for _, row in df.sort_values(metric, ascending=False).head(n).iterrows():
        rows.append(f"- {row['representation']}: {metric}={row[metric]:.4f}")
    return "\n".join(rows) + "\n"


def markdown_table(obj, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if isinstance(obj, pd.Series):
        df = obj.rename("count").reset_index()
        df.columns = ["label", "count"]
    else:
        df = obj.copy()
    if columns is not None:
        df = df[columns]
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "not available\n"
    df = df.astype(object).where(pd.notna(df), "")
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[c]).replace("|", "\\|") for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def table_to_md(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "not available\n"
    return markdown_table(df, columns=columns, max_rows=max_rows)


def build_report(
    config: dict,
    bundle,
    unsup: pd.DataFrame,
    pca: pd.DataFrame,
    umap_metrics: pd.DataFrame,
    clf: pd.DataFrame,
    per_class: pd.DataFrame,
    perm: pd.DataFrame,
    gain: pd.DataFrame,
    shap_summary: pd.DataFrame,
    classification_notes: dict,
    importance_notes: list[str],
    plotly_notes: list[str],
) -> str:
    labels = bundle.metadata[config["target_column"]].fillna("<missing>").astype(str)
    counts = labels.value_counts()
    reps_desc = bundle.qc[["representation", "n_features", "raw_missing_rate", "final_missing_rate"]]
    best_clf = clf[clf["cv_scheme"] == "stratified_kfold"].sort_values("macro_f1", ascending=False)
    pca_plateau = []
    for rep, group in pca.groupby("representation", sort=False):
        group = group.sort_values("dim")
        values = group["pca_knn15"].to_numpy()
        dims = group["dim"].to_numpy()
        plateau = "not found"
        for i in range(len(values) - 3):
            gains = values[i + 1 : i + 4] - values[i : i + 3]
            if (gains < 0.005).all():
                plateau = str(int(dims[i]))
                break
        pca_plateau.append({"representation": rep, "pca_knn15_plateau_dim": plateau})
    plateau_df = pd.DataFrame(pca_plateau)

    top_perm_sections = []
    if not perm.empty:
        for rep, group in perm.groupby("representation", sort=False):
            top = group.sort_values("importance_mean", ascending=False).head(6)
            feats = ", ".join(f"{r.feature} ({r.importance_mean:.3f})" for r in top.itertuples())
            top_perm_sections.append(f"- {rep}: {feats}")

    fusion_candidates = []
    if not perm.empty:
        geo = perm[perm["representation"].isin(["geo_ratio_12", "geo_balance_12", "geo_info_12"])]
        for rep, group in geo.groupby("representation", sort=False):
            top = group.sort_values("importance_mean", ascending=False).head(4)["feature"].tolist()
            fusion_candidates.extend(top)
    fusion_candidates = list(dict.fromkeys(fusion_candidates))[:12]

    hdb_notes = unsup[["representation", "hdbscan_note"]].drop_duplicates() if "hdbscan_note" in unsup else pd.DataFrame()

    return f"""# Stage 2 Geo Representation XGB Report

## 1. Data input and reuse

Processed data directory: `{PROCESSED_DIR}`

Files reused:

- `pyrite_metadata.csv`
- `pyrite_element_values.csv`
- `pyrite_observed_mask.csv`
- `pyrite_numeric_mask.csv`

The raw Excel file was not re-cleaned. This stage uses the cleaned Stage 1
processed tables, because they already contain normalized labels and numeric
element matrices.

Samples: {len(bundle.metadata)}

Target label: `{config["target_column"]}`

Elements ({len(bundle.selected_elements)}): {", ".join(bundle.selected_elements)}

Class counts:

{markdown_table(counts)}

## 2. Representation definitions

Representation QC:

{markdown_table(reps_desc)}

All log transforms use element-specific eps values equal to half of the minimum
positive concentration for that element, with `1e-9` as fallback. Constructed
features are median-imputed and robust-scaled before benchmarking. Rank baseline
uses percentile ranks and standard scaling, following Stage 1.

Feature formulas and group definitions are saved to:
`outputs/representations/feature_definitions.csv`

## 3. Unsupervised input-space benchmark

Main table: `outputs/unsupervised/unsupervised_metrics.csv`

Best input kNN15:

{summarize_top(unsup, "input_knn15")}
Best input kNN30:

{summarize_top(unsup, "input_knn30")}
KMeans/GMM ARI and NMI remain diagnostic rather than final truth, because
deposit-type labels can overlap geochemically and class sizes are imbalanced.
Silhouette is computed using the true `type` label; Davies-Bouldin and
Calinski-Harabasz are computed on KMeans clusters.

HDBSCAN notes:

{markdown_table(hdb_notes) if not hdb_notes.empty else "not available"}

## 4. PCA dimension sweep

PCA is used only as a linear diagnostic. Cumulative explained variance is not
equivalent to geologic structure quality.

Main table: `outputs/pca_sweep/pca_dimension_sweep_metrics.csv`

Approximate kNN15 plateau dimensions:

{markdown_table(plateau_df)}

## 5. UMAP-3D visualization

Coordinates: `outputs/umap3d/umap3d_coordinates.csv`

Metrics: `outputs/umap3d/umap3d_metrics.csv`

UMAP trustworthiness and neighborhood purity:

{table_to_md(umap_metrics, ["representation", "trustworthiness", "umap_knn15", "umap_knn30", "note"], max_rows=20)}
Interactive Plotly status:

{chr(10).join("- " + note for note in plotly_notes)}

UMAP-3D coordinates are not ternary data. They are unconstrained latent
coordinates, so no ternary plot is produced for them.

## 6. XGBoost classification

Metrics: `outputs/xgb/xgb_classification_metrics.csv`

Per-class F1: `outputs/xgb/per_class_f1.csv`

Confusion matrices in long form: `outputs/xgb/confusion_matrix_long.csv`

Stratified CV ranking by macro-F1:

{table_to_md(best_clf, ["representation", "model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "note"], max_rows=20)}
Class filtering:

- minimum class count: {classification_notes.get("min_count")}
- samples used: {classification_notes.get("n_samples_used")}
- group column tried: {classification_notes.get("group_column")}

GroupKFold by citation/source is stricter than random StratifiedKFold and is
included when a usable metadata column is available. It is a first guard against
literature-source leakage, but later stages should add deposit- or district-level
grouping if those metadata become available.

## 7. Feature importance

Permutation importance: `outputs/xgb/feature_importance_permutation.csv`

XGBoost gain: `outputs/xgb/feature_importance_xgb_gain.csv`

SHAP summary: `outputs/xgb/shap_values_summary.csv` when available.

Top permutation features:

{chr(10).join(top_perm_sections) if top_perm_sections else "not available"}

Importance notes:

{chr(10).join("- " + note for note in importance_notes) if importance_notes else "- no warnings"}

Geologic reading should focus on stable signals that are both important in the
supervised task and reasonable under unsupervised structure. Features that mostly
separate source/citation groups in GroupKFold should be treated cautiously.

## 8. Next-stage `geo_fusion_12` suggestion

Candidate features selected from Stage 2 importance and unsupervised behavior:

{chr(10).join("- " + feat for feat in fusion_candidates) if fusion_candidates else "- not enough importance output"}

This is only a candidate list. It should be validated in a new experiment stage
without overwriting this stage.
"""


def main() -> None:
    config = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[stage2] building representations")
    bundle = build_representations(
        PROCESSED_DIR,
        selected_elements=config.get("selected_elements"),
    )
    save_representations(bundle, OUT_DIR)

    labels = bundle.metadata[config["target_column"]].fillna("<missing>").astype(str).to_numpy()

    print("[stage2] unsupervised benchmark")
    unsup = run_unsupervised_metrics(bundle.representations, labels)
    write_table(unsup, OUT_DIR / "unsupervised" / "unsupervised_metrics.csv")
    write_table(unsup, OUT_DIR / "unsupervised_metrics.csv")
    plot_unsupervised_summary(unsup, OUT_DIR / "figures")

    print("[stage2] PCA dimension sweep")
    pca = run_pca_sweep(
        bundle.representations,
        labels,
        max_components=int(config.get("pca_max_components", 12)),
    )
    write_table(pca, OUT_DIR / "pca_sweep" / "pca_dimension_sweep_metrics.csv")
    write_table(pca, OUT_DIR / "pca_dimension_sweep_metrics.csv")
    plot_pca_sweep(pca, OUT_DIR / "pca_sweep")

    print("[stage2] UMAP-3D")
    umap_cfg = config.get("umap", {})
    umap_coords, umap_metrics = run_umap3d(
        bundle.representations,
        bundle.metadata,
        labels,
        OUT_DIR / "umap3d",
        n_neighbors=int(umap_cfg.get("n_neighbors", 30)),
        min_dist=float(umap_cfg.get("min_dist", 0.1)),
        random_state=int(config.get("random_state", 42)),
    )
    write_table(umap_coords, OUT_DIR / "umap3d" / "umap3d_coordinates.csv")
    write_table(umap_metrics, OUT_DIR / "umap3d" / "umap3d_metrics.csv")
    plot_umap_projections(umap_coords, OUT_DIR / "umap3d")
    plotly_notes = save_plotly_umap(umap_coords, OUT_DIR / "umap3d")

    print("[stage2] classification")
    clf, per_class, confusion, classification_notes = run_classification(
        bundle.representations,
        bundle.metadata,
        config["target_column"],
        int(config.get("class_min_count", 30)),
        config.get("classification", {}),
        OUT_DIR / "xgb",
    )
    write_table(clf, OUT_DIR / "xgb" / "xgb_classification_metrics.csv")
    write_table(clf, OUT_DIR / "xgb_classification_metrics.csv")
    write_table(per_class, OUT_DIR / "xgb" / "per_class_f1.csv")
    write_table(confusion, OUT_DIR / "xgb" / "confusion_matrix_long.csv")

    print("[stage2] feature importance")
    perm, gain, shap_summary, importance_notes = feature_importance(
        bundle.representations,
        bundle.metadata,
        config["target_column"],
        int(config.get("class_min_count", 30)),
        config.get("classification", {}),
        OUT_DIR / "xgb",
    )
    write_table(perm, OUT_DIR / "xgb" / "feature_importance_permutation.csv")
    write_table(gain, OUT_DIR / "xgb" / "feature_importance_xgb_gain.csv")
    write_table(shap_summary, OUT_DIR / "xgb" / "shap_values_summary.csv")
    plot_feature_importance(perm, OUT_DIR / "xgb")

    print("[stage2] writing report")
    report = build_report(
        config,
        bundle,
        unsup,
        pca,
        umap_metrics,
        clf,
        per_class,
        perm,
        gain,
        shap_summary,
        classification_notes,
        importance_notes,
        plotly_notes,
    )
    (OUT_DIR / "stage2_report.md").write_text(report, encoding="utf-8")
    print(f"[stage2] done: {OUT_DIR / 'stage2_report.md'}")


if __name__ == "__main__":
    main()
