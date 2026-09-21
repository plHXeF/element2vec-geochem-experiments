from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .common import load_yaml, resolve_path, save_json, set_reproducible_seed
from .experiment_context import ExperimentContext
from .experiments_core import run_core_experiments
from .experiments_followup import run_followup_experiments
from .experiments_process import run_process_experiments
from .experiments_relations import run_relation_experiments
from .pca_sweep_context import (
    LowRankExperimentContext,
    PCABasisStore,
    PCADimensionRegistry,
    SharedInferenceCache,
    dimension_config,
)


def _target_ids(base_config: dict[str, Any]) -> list[str]:
    registry_path = resolve_path(base_config["project"]["results_dir"]) / "experiment_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"Base experiment registry is required before the PCA sweep: {registry_path}"
        )
    return pd.read_csv(registry_path, usecols=["experiment_id"])["experiment_id"].astype(str).tolist()


def _complete_ids(registry_path: Path) -> set[str]:
    if not registry_path.exists():
        return set()
    frame = pd.read_csv(registry_path, usecols=["experiment_id"])
    complete = set()
    for experiment_id in frame["experiment_id"].astype(str):
        out = registry_path.parent / experiment_id
        required = ["config.json", "metrics.json", "samples.csv", "observations.md", "errors.log"]
        if all((out / name).exists() for name in required):
            complete.add(experiment_id)
    return complete


def run(
    sweep_config: dict[str, Any],
    dimensions: list[int] | None = None,
    experiments: list[str] | None = None,
    force: bool = False,
) -> None:
    base_config = load_yaml(sweep_config["base_config"])
    settings = sweep_config["pca_sweep"]
    output_root = resolve_path(settings["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    requested_dimensions = [int(value) for value in (dimensions or settings["dimensions"])]
    all_ids = _target_ids(base_config)
    selected = set(experiments or all_ids)
    unknown = selected.difference(all_ids)
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")

    seed = int(base_config["sampling"]["seed"])
    set_reproducible_seed(seed, int(base_config["runtime"].get("num_threads", 8)))
    base = ExperimentContext(base_config)
    basis_store = PCABasisStore(
        output_root / "basis",
        seed=seed,
        max_components=int(settings.get("max_components", max(requested_dimensions))),
        max_fit_samples=int(settings.get("max_fit_samples", 12000)),
        iterated_power=int(settings.get("randomized_iterated_power", 3)),
    )
    inference_cache = SharedInferenceCache(base)
    run_rows = []
    overall_started = time.perf_counter()

    for dimension in requested_dimensions:
        config = dimension_config(base_config, output_root, dimension)
        results_dir = resolve_path(config["project"]["results_dir"])
        registry_path = results_dir / "experiment_registry.csv"
        completed = set() if force else _complete_ids(registry_path)
        pending = selected.difference(completed)
        print(f"[pca-sweep] dimension={dimension} pending={len(pending)} completed={len(completed & selected)}")
        if not pending:
            continue
        context = LowRankExperimentContext(base, basis_store, inference_cache, dimension, config)
        registry = PCADimensionRegistry(results_dir, dimension)
        started = time.perf_counter()
        run_core_experiments(context, registry, only=pending)
        run_relation_experiments(context, registry, only=pending)
        run_process_experiments(context, registry, only=pending)
        run_followup_experiments(context, registry, only=pending)
        elapsed = time.perf_counter() - started
        rows = len(pd.read_csv(registry.registry_path)) if registry.registry_path.exists() else 0
        run_rows.append(
            {
                "dimension": dimension,
                "experiments_requested": len(pending),
                "registry_rows": rows,
                "elapsed_seconds": elapsed,
            }
        )
        save_json(
            {
                "dimension": dimension,
                "elapsed_seconds": elapsed,
                "basis_summary": context.basis_summary(),
                "results_dir": str(results_dir),
            },
            results_dir / "dimension_manifest.json",
        )
        context.release()
        print(f"[pca-sweep] dimension={dimension} finished in {elapsed:.1f}s rows={rows}")

    basis_store.write_manifest()
    prior_rows = []
    manifest_path = output_root / "run_manifest.csv"
    if manifest_path.exists():
        prior_rows = pd.read_csv(manifest_path).to_dict("records")
    combined = pd.DataFrame(prior_rows + run_rows)
    if not combined.empty:
        combined = combined.sort_values(["dimension", "elapsed_seconds"]).drop_duplicates("dimension", keep="last")
        combined.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    save_json(
        {
            "dimensions": requested_dimensions,
            "experiments": sorted(selected),
            "force": force,
            "elapsed_seconds_this_invocation": time.perf_counter() - overall_started,
            "output_root": str(output_root),
        },
        output_root / "last_run.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute Stage 3.6 experiments in nested PCA subspaces.")
    parser.add_argument(
        "--config",
        default="stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml",
    )
    parser.add_argument("--dimension", action="append", type=int, default=None)
    parser.add_argument("--experiment", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    run(load_yaml(args.config), dimensions=args.dimension, experiments=args.experiment, force=args.force)


if __name__ == "__main__":
    main()
