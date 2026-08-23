from __future__ import annotations

import numpy as np
import pandas as pd


def clean_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values


def fit_transform_stats(
    series: pd.Series,
    field_name: str,
    db_column: str,
    log_transform: bool,
    eps: float,
    min_iqr: float,
) -> dict:
    raw = clean_numeric(series)
    valid_raw = raw.dropna()
    transformed = raw.copy()
    if log_transform:
        transformed = transformed.where(transformed > 0)
        transformed = np.log10(transformed + eps)
    valid = transformed.dropna()
    q25 = float(valid.quantile(0.25)) if len(valid) else 0.0
    q75 = float(valid.quantile(0.75)) if len(valid) else 1.0
    iqr = max(float(q75 - q25), min_iqr)
    median = float(valid.median()) if len(valid) else 0.0
    return {
        "field": field_name,
        "db_column": db_column,
        "median": median,
        "iqr": iqr,
        "log_transform": bool(log_transform),
        "eps": float(eps),
        "coverage": float(len(valid) / len(series)) if len(series) else 0.0,
        "valid_count": int(len(valid)),
        "raw_valid_count": int(len(valid_raw)),
    }


def apply_value_transform(values: pd.Series, stats: dict, clip_z: float | None = None) -> pd.Series:
    x = clean_numeric(values)
    if bool(stats["log_transform"]):
        x = x.where(x > 0)
        x = np.log10(x + float(stats["eps"]))
    z = (x - float(stats["median"])) / float(stats["iqr"])
    if clip_z is not None:
        z = z.clip(-float(clip_z), float(clip_z))
    return z
