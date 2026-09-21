from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")

from geochem_workspace import configured_seed
RANDOM_SEED = configured_seed(42)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from benchmark_representations import (
    COVERAGE_THRESHOLD,
    PROCESSED_DIR,
    RESULT_DIR,
    log10_transform,
    rank_uniform,
    robust_scale,
    standard_scale,
    winsorize,
)


DIAG_DIR = RESULT_DIR / "diagnostics"


def main() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(PROCESSED_DIR / "pyrite_metadata.csv")
    values = pd.read_csv(PROCESSED_DIR / "pyrite_element_values.csv")
    observed = pd.read_csv(PROCESSED_DIR / "pyrite_observed_mask.csv")
    numeric = pd.read_csv(PROCESSED_DIR / "pyrite_numeric_mask.csv")

    element_cols = [c for c in values.columns if c != "sample_id"]
    numeric_coverage = numeric[element_cols].mean().sort_values(ascending=False)
    selected = numeric_coverage[numeric_coverage >= COVERAGE_THRESHOLD].index.tolist()

    value_array = values[selected].to_numpy(dtype=float)
    mask_array = observed[selected].to_numpy(dtype=float)
    labels = metadata["type"].fillna("<missing>")

    # ------------------------------------------------------------------
    # Observed-mask diagnostics
    # ------------------------------------------------------------------
    mask_frame = observed[["sample_id", *selected]].copy()
    pattern_strings = observed[selected].astype(str).agg("".join, axis=1)
    pattern_counts = pattern_strings.value_counts().rename_axis("mask_pattern").reset_index(name="count")
    pattern_counts["coverage_count"] = pattern_counts["mask_pattern"].str.count("1")
    pattern_counts["coverage_rate"] = pattern_counts["coverage_count"] / len(selected)
    pattern_counts.to_csv(DIAG_DIR / "observed_mask_patterns.csv", index=False, encoding="utf-8-sig")

    top_pattern_rows = []
    for pattern in pattern_counts.head(30)["mask_pattern"]:
        idx = pattern_strings == pattern
        subset = metadata.loc[idx]
        type_counts = subset["type"].fillna("<missing>").value_counts().head(6)
        citation_counts = subset["citation"].fillna("<missing>").value_counts().head(6)
        top_pattern_rows.append(
            {
                "mask_pattern": pattern,
                "count": int(idx.sum()),
                "coverage_count": int(pattern.count("1")),
                "top_types": "; ".join(f"{k}:{v}" for k, v in type_counts.items()),
                "top_citations": "; ".join(f"{k}:{v}" for k, v in citation_counts.items()),
            }
        )
    pd.DataFrame(top_pattern_rows).to_csv(
        DIAG_DIR / "observed_mask_top_patterns_by_label.csv", index=False, encoding="utf-8-sig"
    )

    mask_pca = PCA(n_components=4, random_state=RANDOM_SEED).fit(standard_scale(mask_array))
    mask_loadings = pd.DataFrame(
        mask_pca.components_.T,
        index=selected,
        columns=[f"pc{i}" for i in range(1, 5)],
    )
    mask_loadings.insert(0, "element", mask_loadings.index)
    mask_loadings.reset_index(drop=True).to_csv(DIAG_DIR / "observed_mask_pca_loadings.csv", index=False)

    # ------------------------------------------------------------------
    # Raw robust PCA diagnostics
    # ------------------------------------------------------------------
    raw = robust_scale(value_array)
    raw_pca = PCA(n_components=4, random_state=RANDOM_SEED).fit(raw)
    raw_scores = raw_pca.transform(raw)
    raw_loadings = pd.DataFrame(
        raw_pca.components_.T,
        index=selected,
        columns=[f"pc{i}" for i in range(1, 5)],
    )
    raw_loadings["pc1_abs"] = raw_loadings["pc1"].abs()
    raw_loadings["pc2_abs"] = raw_loadings["pc2"].abs()
    raw_loadings.insert(0, "element", raw_loadings.index)
    raw_loadings.reset_index(drop=True).sort_values("pc1_abs", ascending=False).to_csv(
        DIAG_DIR / "raw_robust_pca_loadings.csv", index=False, encoding="utf-8-sig"
    )

    score_df = metadata[
        [
            "sample_id",
            "citation",
            "sample_name",
            "type",
            "ore_stage",
            "stage",
            "hostrock_type",
            "subtype",
        ]
    ].copy()
    score_df["pc1"] = raw_scores[:, 0]
    score_df["pc2"] = raw_scores[:, 1]
    score_df["abs_pc1"] = score_df["pc1"].abs()
    score_df["abs_pc2"] = score_df["pc2"].abs()
    for element in selected:
        score_df[element] = values[element]
    score_df.sort_values("abs_pc1", ascending=False).head(80).to_csv(
        DIAG_DIR / "raw_robust_extreme_pc1_samples.csv", index=False, encoding="utf-8-sig"
    )
    score_df.sort_values("abs_pc2", ascending=False).head(80).to_csv(
        DIAG_DIR / "raw_robust_extreme_pc2_samples.csv", index=False, encoding="utf-8-sig"
    )

    element_distribution = []
    for element in selected:
        series = values[element].dropna()
        element_distribution.append(
            {
                "element": element,
                "count": int(series.shape[0]),
                "median": series.median(),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "p99": series.quantile(0.99),
                "max": series.max(),
                "max_over_median": series.max() / series.median() if series.median() else np.nan,
                "max_over_iqr": series.max() / (series.quantile(0.75) - series.quantile(0.25))
                if (series.quantile(0.75) - series.quantile(0.25))
                else np.nan,
            }
        )
    pd.DataFrame(element_distribution).sort_values("max_over_iqr", ascending=False).to_csv(
        DIAG_DIR / "raw_element_tail_diagnostics.csv", index=False, encoding="utf-8-sig"
    )

    lines = [
        "# Representation Diagnostics",
        "",
        "## Observed Mask",
        "",
        f"- Selected elements: {', '.join(selected)}",
        f"- Unique observed/missing patterns: `{pattern_counts.shape[0]}` across `{len(metadata)}` samples.",
        f"- Top pattern count: `{int(pattern_counts.iloc[0]['count'])}`.",
        "- `observed_mask_z` contains no concentration values; it only encodes reported/censored vs absent cells.",
        "",
        "## Raw Robust PCA",
        "",
        f"- Raw robust PCA explained variance ratio PC1-PC4: {', '.join(f'{v:.3f}' for v in raw_pca.explained_variance_ratio_)}.",
        "- This representation is robust-centered but not tail-clipped; very large concentrations can still dominate PCA.",
        "",
        "## Diagnostic Files",
        "",
        "- `observed_mask_patterns.csv`",
        "- `observed_mask_top_patterns_by_label.csv`",
        "- `observed_mask_pca_loadings.csv`",
        "- `raw_robust_pca_loadings.csv`",
        "- `raw_robust_extreme_pc1_samples.csv`",
        "- `raw_robust_extreme_pc2_samples.csv`",
        "- `raw_element_tail_diagnostics.csv`",
        "",
    ]
    (DIAG_DIR / "diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
