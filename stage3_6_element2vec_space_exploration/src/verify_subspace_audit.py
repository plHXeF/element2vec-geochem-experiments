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
from PIL import Image

from .common import load_yaml, resolve_path


REQUIRED_REPORTS = [
    "subspace_exploration_report.md", "experiment_dimension_matrix.csv", "dimension_summary.csv",
    "low_rank_recoveries.md", "visualization_only_results.md", "confounded_results.md",
    "negative_results.md", "pc_semantic_attribution.csv", "figure_index.md", "reproducibility.md",
]
REQUIRED_COLUMNS = """experiment_id family representation_level native_dimension projection_method
target_dimension pca_explained_variance sample_count random_seed primary_metric metric_value
full_metric_value delta_vs_full random_projection_mean delta_vs_random_projection untrained_metric
geology_metric source_metric missingness_metric bootstrap_ci_low bootstrap_ci_high stability
visual_pattern quantitative_support status main_observation possible_confounder recommended_next_step""".split()


def run(config: dict[str, Any]) -> dict[str, Any]:
    root = resolve_path(config["subspace_audit"]["output_root"])
    checks = []
    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for rel in [
        "audit/subspace_representation_audit.md", "audit/subspace_available_dimensions.json",
        "audit/subspace_experiment_compatibility.csv",
    ]:
        path = root / rel
        check(rel, path.exists() and path.stat().st_size > 0, str(path))
    for name in REQUIRED_REPORTS:
        path = root / "reports" / name
        check(f"report:{name}", path.exists() and path.stat().st_size > 0, str(path))

    compat = pd.read_csv(root / "audit" / "subspace_experiment_compatibility.csv")
    check("compatibility_50", len(compat) == 50 and compat["experiment_id"].nunique() == 50, f"rows={len(compat)}")
    matrix = pd.read_csv(root / "reports" / "experiment_dimension_matrix.csv", low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in matrix]
    check("matrix_columns", not missing, f"missing={missing}")
    check("matrix_experiment_coverage", matrix.loc[matrix["family"] != "S", "experiment_id"].nunique() == 50,
          f"experiments={matrix.loc[matrix['family'] != 'S', 'experiment_id'].nunique()}")
    check("explicit_not_applicable", (matrix["projection_method"] == "Not applicable").sum() > 0,
          f"rows={(matrix['projection_method'] == 'Not applicable').sum()}")

    registry = pd.read_csv(root / "results" / "experiment_registry.csv", low_memory=False)
    check("replay_no_technical_failure", not registry["status"].eq("Failed technically").any(),
          f"failed={registry['status'].eq('Failed technically').sum()}")
    for dim in (64, 32, 3, 2):
        mask = registry["experiment_id"].str.contains(fr"__RP{dim}_seed\d+$", regex=True)
        counts = registry.loc[mask, "experiment_id"].str.extract(r"seed(\d+)$")[0].nunique()
        check(f"rp{dim}_ten_seeds", counts == 10, f"seeds={counts}, rows={mask.sum()}")

    dims = json.loads((root / "audit" / "subspace_available_dimensions.json").read_text(encoding="utf-8"))
    numeric = dims["static_trained_numeric"]["spaces"]
    check("no_rank_padding_numeric", not numeric["PCA64"]["applicable"] and numeric["PCA32"]["applicable"],
          f"PCA64={numeric['PCA64']}, PCA32={numeric['PCA32']}")

    figures = pd.read_csv(root / "reports" / "figure_index.csv")
    broken = []
    for value in figures["figure"].dropna().head(300):
        path = Path(value)
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            broken.append(f"{path}: {exc}")
    check("figure_integrity", not broken, f"indexed={len(figures)}, checked={min(len(figures),300)}, broken={len(broken)}")

    report = (root / "reports" / "subspace_exploration_report.md").read_text(encoding="utf-8")
    sections = [f"## {i}." for i in range(1, 13)]
    check("report_sections", all(section in report for section in sections), f"sections={sections}")
    result = {"passed": all(row["passed"] for row in checks), "checks": checks,
              "matrix_rows": len(matrix), "registry_rows": len(registry), "figure_rows": len(figures)}
    path = root / "reports" / "verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["passed"]:
        failed = [row for row in checks if not row["passed"]]
        raise SystemExit(f"Verification failed: {failed}")
    print(f"[verify-subspace] passed checks={len(checks)} matrix={len(matrix)} registry={len(registry)} figures={len(figures)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
