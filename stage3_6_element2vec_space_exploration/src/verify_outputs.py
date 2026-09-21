from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .common import ensure_directories, load_json, load_yaml, resolve_path
from .registry import REGISTRY_COLUMNS


REQUIRED_AUDIT = ["model_audit.md", "data_audit.md", "available_representations.json"]
REQUIRED_REPORTS = [
    "element2vec_exploration_report.md",
    "experiment_summary.csv",
    "promising_directions.md",
    "negative_results.md",
    "data_and_leakage_risks.md",
    "figure_index.md",
    "reproducibility.md",
]
REQUIRED_EXPERIMENT_FILES = ["config.json", "metrics.json", "samples.csv", "observations.md", "errors.log"]
ALLOWED_STATUSES = {
    "Strong positive",
    "Moderate positive",
    "Weak/ambiguous",
    "Negative",
    "Failed technically",
    "Invalid due to leakage/confounding",
}


def verify(config: dict) -> dict:
    paths = ensure_directories(config)
    errors: list[str] = []
    for name in REQUIRED_AUDIT:
        if not (paths["audit"] / name).is_file():
            errors.append(f"missing audit/{name}")
    for name in REQUIRED_REPORTS:
        if not (paths["reports"] / name).is_file():
            errors.append(f"missing reports/{name}")

    registry_path = paths["results"] / "experiment_registry.csv"
    if not registry_path.exists():
        errors.append("missing results/experiment_registry.csv")
        raise AssertionError("\n".join(errors))
    registry = pd.read_csv(registry_path, low_memory=False)
    missing_columns = [column for column in REGISTRY_COLUMNS if column not in registry]
    if missing_columns:
        errors.append(f"registry missing columns: {missing_columns}")
    if registry["experiment_id"].duplicated().any():
        errors.append("registry contains duplicate experiment_id")
    if not set(registry["status"]).issubset(ALLOWED_STATUSES):
        errors.append(f"unexpected statuses: {sorted(set(registry['status']) - ALLOWED_STATUSES)}")
    expected_families = set("ABCDEFGHIJKX")
    missing_families = expected_families - set(registry["family"].astype(str))
    if missing_families:
        errors.append(f"experiment families not represented: {sorted(missing_families)}")
    required_nonempty = [
        "experiment_name", "family", "representation_level", "dataset", "input_fields",
        "target_geological_relation", "control_group", "quantitative_metrics", "status",
        "conclusion_summary", "risks_and_confounders", "hypothesis", "metric_primary",
        "stability", "main_observation", "possible_confounder", "recommended_next_step",
    ]
    for column in required_nonempty:
        bad = registry[column].isna() | registry[column].astype(str).str.strip().eq("")
        if bad.any():
            errors.append(f"registry column {column} empty for {registry.loc[bad, 'experiment_id'].tolist()}")

    figure_count = 0
    for _, row in registry.iterrows():
        exp_dir = paths["results"] / row["experiment_id"]
        if not exp_dir.is_dir():
            errors.append(f"missing experiment directory {row['experiment_id']}")
            continue
        for name in REQUIRED_EXPERIMENT_FILES:
            if not (exp_dir / name).is_file():
                errors.append(f"{row['experiment_id']} missing {name}")
        figures_dir = exp_dir / "figures"
        if not figures_dir.is_dir() or not any(figures_dir.iterdir()):
            errors.append(f"{row['experiment_id']} has no figure")
        for raw in str(row["figure_paths"]).split(";"):
            if not raw or raw == "nan":
                continue
            figure_count += 1
            path = resolve_path(raw)
            if not path.is_file():
                errors.append(f"{row['experiment_id']} figure path missing: {raw}")
        if row["status"] == "Failed technically":
            error_log = exp_dir / "errors.log"
            if not error_log.exists() or not error_log.read_text(encoding="utf-8").strip():
                errors.append(f"{row['experiment_id']} failed technically but errors.log is empty")

    summary = pd.read_csv(paths["reports"] / "experiment_summary.csv", low_memory=False)
    if len(summary) != len(registry):
        errors.append(f"experiment_summary rows={len(summary)} registry rows={len(registry)}")
    if set(summary["experiment_id"]) != set(registry["experiment_id"]):
        errors.append("experiment_summary ids do not match registry")

    manifest = load_json(paths["cache"] / "cache_manifest.json")
    metadata = pd.read_csv(paths["cache"] / "sample_metadata.csv", usecols=["sample_id"])
    sample_hash = hashlib.sha256(",".join(metadata["sample_id"].astype(str)).encode("utf-8")).hexdigest()
    if sample_hash != manifest["sample_id_sha256"]:
        errors.append("sample id hash does not match cache manifest")
    if len(metadata) != int(manifest["sample_count"]):
        errors.append("sample count does not match cache manifest")

    result = {
        "experiments": len(registry),
        "families": sorted(set(registry["family"].astype(str))),
        "status_counts": registry["status"].value_counts().to_dict(),
        "registered_figure_paths": figure_count,
        "cache_samples": len(metadata),
        "errors": errors,
    }
    if errors:
        raise AssertionError("Output verification failed:\n- " + "\n- ".join(errors))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/exploration.yaml")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    result = verify(load_yaml(args.config))
    print("[verify] PASS")
    print(result)


if __name__ == "__main__":
    main()
