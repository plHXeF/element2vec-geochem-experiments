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
from typing import Any

import pandas as pd

from .common import load_yaml, resolve_path


def verify(sweep_config: dict[str, Any]) -> dict[str, Any]:
    settings = sweep_config["pca_sweep"]
    base_config = load_yaml(sweep_config["base_config"])
    root = resolve_path(settings["output_root"])
    base_registry = pd.read_csv(resolve_path(base_config["project"]["results_dir"]) / "experiment_registry.csv")
    expected_ids = set(base_registry["experiment_id"].astype(str))
    expected_dimensions = [int(value) for value in settings["dimensions"]]
    errors: list[str] = []
    total_rows = 0
    failed_by_dimension: dict[int, list[str]] = {}
    required_artifacts = ["config.json", "metrics.json", "samples.csv", "observations.md", "errors.log"]

    for dimension in expected_dimensions:
        results = root / "results" / f"{dimension}d"
        registry_path = results / "experiment_registry.csv"
        if not registry_path.exists():
            errors.append(f"missing registry for {dimension}D")
            continue
        registry = pd.read_csv(registry_path)
        total_rows += len(registry)
        ids = set(registry["experiment_id"].astype(str))
        if ids != expected_ids:
            errors.append(f"{dimension}D experiment IDs differ: missing={sorted(expected_ids-ids)}, extra={sorted(ids-expected_ids)}")
        if registry["experiment_id"].duplicated().any():
            errors.append(f"{dimension}D registry has duplicate IDs")
        failed = registry.loc[registry["status"].eq("Failed technically"), "experiment_id"].astype(str).tolist()
        failed_by_dimension[dimension] = failed
        for experiment_id in expected_ids:
            out = results / experiment_id
            for name in required_artifacts:
                if not (out / name).exists():
                    errors.append(f"{dimension}D/{experiment_id} missing {name}")
            config_path = out / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                recorded = config.get("config", {}).get("pca_dimension")
                if recorded != dimension:
                    errors.append(f"{dimension}D/{experiment_id} records dimension={recorded}")
            if experiment_id in failed and (out / "errors.log").exists() and not (out / "errors.log").read_text(encoding="utf-8").strip():
                errors.append(f"{dimension}D/{experiment_id} failed without errors.log evidence")

    expected_failures = {"D3b_nonlinear_axis_failure", "I0_strict_cipw_unavailable"}
    for dimension, failed in failed_by_dimension.items():
        if set(failed) != expected_failures:
            errors.append(f"{dimension}D unexpected technical failures: {failed}")

    reports = root / "reports"
    required_reports = [
        "pca_dimension_sweep_report.md",
        "experiment_dimension_metrics.csv",
        "experiment_dimension_analysis.csv",
        "all_scalar_metrics_long.csv",
        "scalar_metric_dimension_analysis.csv",
        "experiment_all_metric_summary.csv",
        "dimension_summary.csv",
        "family_dimension_summary.csv",
        "figure_index.csv",
        "coverage_audit.json",
    ]
    for name in required_reports:
        path = reports / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing/empty report artifact: {name}")
    coverage_path = reports / "coverage_audit.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        if not coverage.get("coverage_ok") or coverage.get("total_rows") != len(expected_ids) * len(expected_dimensions):
            errors.append(f"coverage audit failed: {coverage}")
    metrics_path = reports / "experiment_dimension_metrics.csv"
    if metrics_path.exists() and len(pd.read_csv(metrics_path)) != len(expected_ids) * len(expected_dimensions):
        errors.append("experiment_dimension_metrics row count mismatch")
    scalar_path = reports / "all_scalar_metrics_long.csv"
    if scalar_path.exists():
        scalar = pd.read_csv(scalar_path)
        scalar_dimensions = set(pd.to_numeric(scalar["pca_dimension"], errors="coerce").dropna().astype(int))
        if scalar_dimensions != set(expected_dimensions):
            errors.append("all_scalar_metrics_long does not cover all dimensions")
    figure_index_path = reports / "figure_index.csv"
    if figure_index_path.exists():
        figures = pd.read_csv(figure_index_path)
        for value in figures["figure_path"].dropna().astype(str):
            if not Path(value).exists():
                errors.append(f"registered curve figure missing: {value}")

    basis_manifest = root / "basis" / "basis_manifest.csv"
    if not basis_manifest.exists() or len(pd.read_csv(basis_manifest)) < 10:
        errors.append("PCA basis manifest is missing or incomplete")

    summary = {
        "experiments_per_dimension": len(expected_ids),
        "dimensions": expected_dimensions,
        "total_registry_rows": total_rows,
        "expected_technical_failures": sorted(expected_failures),
        "basis_count": len(pd.read_csv(basis_manifest)) if basis_manifest.exists() else 0,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))
    print("[pca-verify] PASS")
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml",
    )
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    verify(load_yaml(args.config))


if __name__ == "__main__":
    main()
