from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    OUTPUT_DIR,
    config_snapshot,
    ensure_output_dirs,
    load_config,
    metadata_columns,
    resolve_path,
    robust_scale,
    save_dataframe,
    selected_elements,
    set_seed,
)


def preprocess(config: dict) -> dict:
    ensure_output_dirs()
    set_seed(int(config.get("random_seed", 42)))
    config_snapshot(config)

    processed_dir = resolve_path(config.get("processed_dir", "../pyrite_benchmark/data/processed"))
    sample_col, label_col = metadata_columns(config)
    elements = selected_elements(config)
    eps = float(config.get("eps", 1e-9))

    metadata = pd.read_csv(processed_dir / "pyrite_metadata.csv")
    values = pd.read_csv(processed_dir / "pyrite_element_values.csv")
    numeric_mask_path = processed_dir / "pyrite_numeric_mask.csv"
    if numeric_mask_path.exists():
        numeric_mask = pd.read_csv(numeric_mask_path)
    else:
        numeric_mask = values.copy()
        for col in elements:
            numeric_mask[col] = values[col].notna().astype(int)

    missing_elements = [el for el in elements if el not in values.columns]
    if missing_elements:
        raise ValueError(f"Missing element columns: {missing_elements}")
    if sample_col not in metadata.columns or sample_col not in values.columns:
        raise ValueError(f"sample_id column {sample_col!r} not found")
    if label_col not in metadata.columns:
        raise ValueError(f"label column {label_col!r} not found in metadata")

    values = values[[sample_col] + elements].copy()
    numeric_mask = numeric_mask[[sample_col] + elements].copy()
    metadata = metadata.copy()
    values[sample_col] = values[sample_col].astype(str)
    numeric_mask[sample_col] = numeric_mask[sample_col].astype(str)
    metadata[sample_col] = metadata[sample_col].astype(str)

    metadata = metadata.set_index(sample_col).loc[values[sample_col]].reset_index()
    sample_ids = values[sample_col].astype(str).to_numpy()
    deposit_type = metadata[label_col].fillna("<missing>").astype(str).to_numpy()
    classes = np.array(sorted(pd.unique(deposit_type)), dtype=object)
    class_to_id = {name: i for i, name in enumerate(classes)}
    label_ids = np.array([class_to_id[x] for x in deposit_type], dtype=np.int64)

    raw = values[elements].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    visible = numeric_mask[elements].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    visible = (visible > 0).astype(np.float32)

    raw_for_log = np.where(np.isfinite(raw), np.maximum(raw, 0.0), np.nan)
    log_values = np.log10(raw_for_log + eps)
    log_values[~np.isfinite(raw)] = np.nan
    log_robust, med, iqr = robust_scale(log_values)

    pre_dir = OUTPUT_DIR / "preprocessed"
    np.savez_compressed(
        pre_dir / "pyrite_token_data.npz",
        sample_id=sample_ids,
        deposit_type=deposit_type,
        label_id=label_ids,
        element_values=log_robust.astype(np.float32),
        visible_mask=visible.astype(np.float32),
        raw_values=raw.astype(np.float32),
        elements=np.array(elements, dtype=object),
        deposit_classes=classes,
        log_median=med,
        log_iqr=iqr,
    )

    save_dataframe(
        pd.DataFrame({"element": elements, "log_median": med, "log_iqr": iqr}),
        pre_dir / "log_robust_scaler.csv",
    )
    save_dataframe(
        pd.DataFrame({"deposit_type": classes, "label_id": np.arange(len(classes))}),
        pre_dir / "deposit_label_mapping.csv",
    )
    save_dataframe(
        pd.DataFrame({"sample_id": sample_ids, "deposit_type": deposit_type, "label_id": label_ids}),
        pre_dir / "sample_metadata.csv",
    )
    log_space = pd.DataFrame(log_robust, columns=elements)
    log_space.insert(0, "deposit_type", deposit_type)
    log_space.insert(0, "sample_id", sample_ids)
    save_dataframe(log_space, pre_dir / "log_robust_space.csv")

    summary = {
        "n_samples": int(len(sample_ids)),
        "n_elements": int(len(elements)),
        "n_deposit_types": int(len(classes)),
        "processed_dir": str(processed_dir),
        "elements": elements,
        "deposit_counts": pd.Series(deposit_type).value_counts().to_dict(),
        "numeric_coverage": {el: float(visible[:, i].mean()) for i, el in enumerate(elements)},
    }
    pd.Series(summary, dtype=object).to_json(pre_dir / "preprocess_summary.json", force_ascii=False, indent=2)
    print(f"[preprocess] samples={summary['n_samples']} elements={summary['n_elements']} classes={summary['n_deposit_types']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pyrite_token_transformer.yaml"))
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    preprocess(load_config(args.config))


if __name__ == "__main__":
    main()
