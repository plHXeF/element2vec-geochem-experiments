from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "pyrite.xlsx"
OUT_DIR = ROOT / "outputs"

KNOWN_METADATA_COLUMNS = [
    "Unnamed: 0",
    "Citation",
    "Sample_name",
    "Type",
    "Ore_stage",
    "Stage",
    "Hostclass",
    "Hostrock_type",
    "Subtype",
]

LABEL_COLUMNS = [
    "Citation",
    "Type",
    "Ore_stage",
    "Stage",
    "Hostclass",
    "Hostrock_type",
    "Subtype",
]

CENSORED_RE = re.compile(r"^\s*(?:<|<=|≤|＜|Јј)\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")
ABOVE_RANGE_RE = re.compile(r"^\s*(?:>|>=|≥)\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")


def parse_geochem_value(value):
    """Return numeric value, state, and raw detection limit if present."""
    if pd.isna(value):
        return np.nan, "missing", np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value), "measured", np.nan
    text = str(value).strip()
    if not text:
        return np.nan, "missing", np.nan
    if text.lower() in {"bdl", "b.d.l.", "below detection", "below detection limit"}:
        return np.nan, "below_detection_no_limit", np.nan
    match = CENSORED_RE.match(text)
    if match:
        limit = float(match.group(1))
        return limit / 2.0, "below_detection", limit
    match = ABOVE_RANGE_RE.match(text)
    if match:
        limit = float(match.group(1))
        return limit, "above_range", limit
    try:
        return float(text), "measured", np.nan
    except ValueError:
        return np.nan, "non_numeric", np.nan


def pct(n: float, d: float) -> float:
    if not d:
        return 0.0
    return 100.0 * n / d


def top_counts(df: pd.DataFrame, column: str, limit: int = 12) -> list[tuple[str, int]]:
    counts = (
        df[column]
        .fillna("<missing>")
        .astype(str)
        .str.strip()
        .replace("", "<blank>")
        .value_counts(dropna=False)
        .head(limit)
    )
    return [(str(k), int(v)) for k, v in counts.items()]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(RAW_FILE)
    df = pd.read_excel(RAW_FILE, sheet_name=xls.sheet_names[0])
    n_rows, n_cols = df.shape

    metadata_cols = [c for c in KNOWN_METADATA_COLUMNS if c in df.columns]
    element_cols = [c for c in df.columns if c not in metadata_cols]

    numeric_df = pd.DataFrame(index=df.index)
    state_df = pd.DataFrame(index=df.index)
    detection_limit_df = pd.DataFrame(index=df.index)
    for col in element_cols:
        parsed = df[col].map(parse_geochem_value)
        numeric_df[col] = parsed.map(lambda x: x[0])
        state_df[col] = parsed.map(lambda x: x[1])
        detection_limit_df[col] = parsed.map(lambda x: x[2])

    column_rows = []
    for col in df.columns:
        role = "metadata" if col in metadata_cols else "element"
        raw_non_null = int(df[col].notna().sum())
        column_rows.append(
            {
                "column": col,
                "role": role,
                "raw_dtype": str(df[col].dtype),
                "non_null_count": raw_non_null,
                "missing_count": int(n_rows - raw_non_null),
                "missing_rate": round(1 - raw_non_null / n_rows, 6),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )
    columns_summary = pd.DataFrame(column_rows)
    columns_summary.to_csv(OUT_DIR / "pyrite_columns.csv", index=False, encoding="utf-8-sig")

    missing_rows = []
    for col in df.columns:
        missing_rows.append(
            {
                "column": col,
                "role": "metadata" if col in metadata_cols else "element",
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": round(float(df[col].isna().mean()), 6),
                "non_null_count": int(df[col].notna().sum()),
            }
        )
    missingness = pd.DataFrame(missing_rows).sort_values(
        ["role", "missing_rate", "column"], ascending=[True, False, True]
    )
    missingness.to_csv(OUT_DIR / "pyrite_missingness.csv", index=False, encoding="utf-8-sig")

    element_rows = []
    for col in element_cols:
        states = state_df[col].value_counts(dropna=False).to_dict()
        values = numeric_df[col].dropna()
        element_rows.append(
            {
                "element": col,
                "coverage_count": int(values.shape[0]),
                "coverage_rate": round(float(values.shape[0] / n_rows), 6),
                "missing_count": int(states.get("missing", 0)),
                "below_detection_count": int(states.get("below_detection", 0)),
                "below_detection_no_limit_count": int(states.get("below_detection_no_limit", 0)),
                "above_range_count": int(states.get("above_range", 0)),
                "non_numeric_count": int(states.get("non_numeric", 0)),
                "min": values.min() if not values.empty else math.nan,
                "p25": values.quantile(0.25) if not values.empty else math.nan,
                "median": values.median() if not values.empty else math.nan,
                "p75": values.quantile(0.75) if not values.empty else math.nan,
                "max": values.max() if not values.empty else math.nan,
            }
        )
    element_summary = pd.DataFrame(element_rows).sort_values(
        ["coverage_count", "element"], ascending=[False, True]
    )
    element_summary.to_csv(OUT_DIR / "pyrite_element_summary.csv", index=False, encoding="utf-8-sig")

    label_rows = []
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            continue
        for value, count in top_counts(df, col, limit=30):
            label_rows.append({"column": col, "value": value, "count": count})
    label_summary = pd.DataFrame(label_rows)
    label_summary.to_csv(OUT_DIR / "pyrite_label_summary.csv", index=False, encoding="utf-8-sig")

    preview_cols = [c for c in ["Sample_name", "Citation", "Type", "Hostrock_type", "Subtype"] if c in df.columns]
    preview = pd.concat([df[preview_cols], numeric_df[element_cols]], axis=1).head(50)
    preview.to_csv(OUT_DIR / "pyrite_numeric_preview.csv", index=False, encoding="utf-8-sig")

    high_coverage = element_summary[element_summary["coverage_rate"] >= 0.5]
    medium_coverage = element_summary[
        (element_summary["coverage_rate"] >= 0.2) & (element_summary["coverage_rate"] < 0.5)
    ]
    censored_total = int((state_df == "below_detection").sum().sum())
    bdl_total = int((state_df == "below_detection_no_limit").sum().sum())
    above_range_total = int((state_df == "above_range").sum().sum())
    non_numeric_total = int((state_df == "non_numeric").sum().sum())

    lines = []
    lines.append("# Pyrite Data Audit")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Source file: `{RAW_FILE.relative_to(ROOT)}`")
    lines.append(f"- Sheets: {', '.join(xls.sheet_names)}")
    lines.append(f"- Active sheet: `{xls.sheet_names[0]}`")
    lines.append(f"- Shape: `{n_rows}` rows x `{n_cols}` columns")
    lines.append(f"- Metadata columns: `{len(metadata_cols)}`")
    lines.append(f"- Element columns: `{len(element_cols)}`")
    lines.append(f"- Below-detection cells parsed from `<value>` or `<=value`: `{censored_total}`")
    lines.append(f"- `bdl` cells without explicit detection limit: `{bdl_total}`")
    lines.append(f"- Above-range cells parsed from `>value`: `{above_range_total}`")
    lines.append(f"- Non-numeric cells inside element columns: `{non_numeric_total}`")
    lines.append("")
    lines.append("## Metadata Columns")
    lines.append("")
    for col in metadata_cols:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append("## Element Columns")
    lines.append("")
    lines.append(", ".join(f"`{c}`" for c in element_cols))
    lines.append("")
    lines.append("## Label Columns: Top Values")
    lines.append("")
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            continue
        lines.append(f"### `{col}`")
        for value, count in top_counts(df, col, limit=12):
            lines.append(f"- `{value}`: {count}")
        lines.append("")
    lines.append("## Element Coverage")
    lines.append("")
    lines.append(f"- Elements with >=50% coverage: `{len(high_coverage)}`")
    lines.append(f"- Elements with 20%-50% coverage: `{len(medium_coverage)}`")
    lines.append("")
    lines.append("### Highest Coverage Elements")
    lines.append("")
    for _, row in element_summary.head(20).iterrows():
        lines.append(
            f"- `{row['element']}`: {int(row['coverage_count'])}/{n_rows} "
            f"({pct(row['coverage_count'], n_rows):.1f}%), "
            f"below detection: {int(row['below_detection_count'])}, "
            f"bdl: {int(row['below_detection_no_limit_count'])}"
        )
    lines.append("")
    lines.append("### Lowest Coverage Elements")
    lines.append("")
    for _, row in element_summary.tail(15).iterrows():
        lines.append(
            f"- `{row['element']}`: {int(row['coverage_count'])}/{n_rows} "
            f"({pct(row['coverage_count'], n_rows):.1f}%)"
        )
    lines.append("")
    lines.append("## Initial Benchmark Implications")
    lines.append("")
    lines.append("- Treat `<value>` / `<=value` and `bdl` as censored observations, not ordinary missing data.")
    lines.append("- First benchmark should compare both value-based representations and an observation-mask representation.")
    lines.append("- Coverage is uneven across elements, so each representation should record which elements were actually used.")
    lines.append("- `Type`, `Hostrock_type`, and `Subtype` look like the first candidate grouping variables for PCA/UMAP coloring and cluster purity.")
    lines.append("")
    lines.append("## Generated Files")
    lines.append("")
    lines.append("- `pyrite_columns.csv`: column roles and basic metadata")
    lines.append("- `pyrite_missingness.csv`: missingness by column")
    lines.append("- `pyrite_element_summary.csv`: element coverage, censoring, and simple statistics")
    lines.append("- `pyrite_label_summary.csv`: top label/category counts")
    lines.append("- `pyrite_numeric_preview.csv`: first 50 rows after numeric parsing")
    lines.append("")

    (OUT_DIR / "pyrite_data_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
