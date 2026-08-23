from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "pyrite.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

METADATA_COLUMNS = [
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

LABEL_COLUMNS = ["Type", "Ore_stage", "Stage", "Hostclass", "Hostrock_type", "Subtype"]

CENSORED_RE = re.compile(r"^\s*(?:<|<=|≤|＜|Јј)\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")
ABOVE_RANGE_RE = re.compile(r"^\s*(?:>|>=|≥)\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$")

TYPE_MAP = {
    "orogenic": "orogenic",
    "sedimentary": "sedimentary",
    "porphyry": "porphyry",
    "carlin": "carlin",
    "ni-cu/pge": "ni-cu-pge",
    "vms": "vms",
}


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).replace("\u00a0", " ").strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label(value) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return text.lower()


def normalize_type(value) -> str | None:
    label = normalize_label(value)
    if label is None:
        return None
    return TYPE_MAP.get(label, label)


def parse_geochem_value(value):
    """
    Parse element cells into a benchmark-ready numeric value plus observation state.

    Explicit lower-than-detection values are represented as half the detection
    limit. Plain "bdl" is kept as missing numeric value but retained in state.
    """
    if pd.isna(value):
        return np.nan, "missing", np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value), "measured", np.nan

    text = clean_text(value)
    if text is None:
        return np.nan, "missing", np.nan

    lowered = text.lower()
    if lowered in {"bdl", "b.d.l.", "below detection", "below detection limit"}:
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


def build_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in LABEL_COLUMNS:
        if col not in df.columns:
            continue
        if col == "Type":
            normalizer = normalize_type
        else:
            normalizer = normalize_label
        counts = df[col].fillna("<missing>").astype(str).map(lambda x: clean_text(x) or "<missing>").value_counts()
        for raw_value, count in counts.items():
            raw_for_norm = None if raw_value == "<missing>" else raw_value
            rows.append(
                {
                    "column": col,
                    "raw_value": raw_value,
                    "standard_value": normalizer(raw_for_norm) or "",
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(RAW_FILE, sheet_name="Sheet1")
    n_rows = len(df)
    metadata_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    element_cols = [c for c in df.columns if c not in metadata_cols]

    sample_ids = pd.Series([f"PYR{i + 1:06d}" for i in range(n_rows)], name="sample_id")

    metadata = pd.DataFrame({"sample_id": sample_ids})
    metadata["source_row"] = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else np.arange(n_rows)
    metadata["citation"] = df["Citation"].map(clean_text) if "Citation" in df.columns else None
    metadata["sample_name"] = df["Sample_name"].map(clean_text) if "Sample_name" in df.columns else None
    metadata["type_raw"] = df["Type"].map(clean_text) if "Type" in df.columns else None
    metadata["type"] = df["Type"].map(normalize_type) if "Type" in df.columns else None
    metadata["ore_stage_raw"] = df["Ore_stage"].map(clean_text) if "Ore_stage" in df.columns else None
    metadata["ore_stage"] = df["Ore_stage"].map(normalize_label) if "Ore_stage" in df.columns else None
    metadata["stage_raw"] = df["Stage"].map(clean_text) if "Stage" in df.columns else None
    metadata["stage"] = df["Stage"].map(normalize_label) if "Stage" in df.columns else None
    metadata["hostclass_raw"] = df["Hostclass"].map(clean_text) if "Hostclass" in df.columns else None
    metadata["hostclass"] = df["Hostclass"].map(normalize_label) if "Hostclass" in df.columns else None
    metadata["hostrock_type_raw"] = df["Hostrock_type"].map(clean_text) if "Hostrock_type" in df.columns else None
    metadata["hostrock_type"] = df["Hostrock_type"].map(normalize_label) if "Hostrock_type" in df.columns else None
    metadata["subtype_raw"] = df["Subtype"].map(clean_text) if "Subtype" in df.columns else None
    metadata["subtype"] = df["Subtype"].map(normalize_label) if "Subtype" in df.columns else None

    values = pd.DataFrame({"sample_id": sample_ids})
    states = pd.DataFrame({"sample_id": sample_ids})
    detection_limits = pd.DataFrame({"sample_id": sample_ids})

    for col in element_cols:
        parsed = df[col].map(parse_geochem_value)
        values[col] = parsed.map(lambda x: x[0])
        states[col] = parsed.map(lambda x: x[1])
        detection_limits[col] = parsed.map(lambda x: x[2])

    observed_mask = states.copy()
    for col in element_cols:
        observed_mask[col] = states[col].isin(
            ["measured", "below_detection", "below_detection_no_limit", "above_range"]
        ).astype(int)

    numeric_mask = states.copy()
    for col in element_cols:
        numeric_mask[col] = values[col].notna().astype(int)

    label_mapping = build_label_mapping(df)

    metadata.to_csv(PROCESSED_DIR / "pyrite_metadata.csv", index=False, encoding="utf-8-sig")
    values.to_csv(PROCESSED_DIR / "pyrite_element_values.csv", index=False, encoding="utf-8-sig")
    states.to_csv(PROCESSED_DIR / "pyrite_element_states.csv", index=False, encoding="utf-8-sig")
    detection_limits.to_csv(PROCESSED_DIR / "pyrite_detection_limits.csv", index=False, encoding="utf-8-sig")
    observed_mask.to_csv(PROCESSED_DIR / "pyrite_observed_mask.csv", index=False, encoding="utf-8-sig")
    numeric_mask.to_csv(PROCESSED_DIR / "pyrite_numeric_mask.csv", index=False, encoding="utf-8-sig")
    label_mapping.to_csv(PROCESSED_DIR / "pyrite_label_mapping.csv", index=False, encoding="utf-8-sig")

    type_counts = metadata["type"].fillna("<missing>").value_counts()
    state_counts = states[element_cols].stack().value_counts()
    element_numeric_coverage = numeric_mask[element_cols].sum().sort_values(ascending=False)

    lines = [
        "# Pyrite Prepared Dataset",
        "",
        "## Outputs",
        "",
        "- `data/processed/pyrite_metadata.csv`: sample metadata with standardized labels",
        "- `data/processed/pyrite_element_values.csv`: numeric element matrix",
        "- `data/processed/pyrite_element_states.csv`: per-element observation state",
        "- `data/processed/pyrite_detection_limits.csv`: explicit detection or range limits when present",
        "- `data/processed/pyrite_observed_mask.csv`: 1 if a cell had any observed/censored value",
        "- `data/processed/pyrite_numeric_mask.csv`: 1 if a numeric value is available after parsing",
        "- `data/processed/pyrite_label_mapping.csv`: raw-to-standard label mapping",
        "",
        "## Parsing Rules",
        "",
        "- Numeric cells are kept as measured values.",
        "- `<value`, `<=value`, `≤value`, and full-width variants are stored as `value / 2` with state `below_detection`.",
        "- `bdl` is stored as missing numeric value with state `below_detection_no_limit`.",
        "- `>value` is stored as `value` with state `above_range`.",
        "- Missing cells are state `missing`.",
        "",
        "## Type Counts After Standardization",
        "",
    ]
    for label, count in type_counts.items():
        lines.append(f"- `{label}`: {int(count)}")
    lines.extend(["", "## Element State Counts", ""])
    for state, count in state_counts.items():
        lines.append(f"- `{state}`: {int(count)}")
    lines.extend(["", "## Highest Numeric Coverage Elements", ""])
    for element, count in element_numeric_coverage.head(15).items():
        lines.append(f"- `{element}`: {int(count)}/{n_rows} ({count / n_rows * 100:.1f}%)")
    lines.extend(["", "## Notes For Benchmark", ""])
    lines.append("- Use `type` as the first benchmark grouping label; it is now case-normalized.")
    lines.append("- Keep `stage` and `subtype` as auxiliary labels until their geological meaning is reviewed.")
    lines.append("- Compare value-only representations with mask-aware representations because censoring is substantial.")
    lines.append("")

    (OUTPUT_DIR / "pyrite_prepare_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
