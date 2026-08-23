from __future__ import annotations

import argparse
import time

from .common import load_yaml, set_reproducible_seed
from .experiment_context import ExperimentContext
from .experiments_core import run_core_experiments
from .experiments_relations import run_relation_experiments
from .experiments_process import run_process_experiments
from .experiments_followup import run_followup_experiments
from .registry import ExperimentRegistry


def run(config: dict, batch: str = "all", experiments: list[str] | None = None) -> None:
    started = time.perf_counter()
    set_reproducible_seed(
        int(config["sampling"]["seed"]),
        int(config["runtime"].get("num_threads", 8)),
    )
    context = ExperimentContext(config)
    registry = ExperimentRegistry(config["project"]["results_dir"])
    only = set(experiments) if experiments else None
    if batch in {"all", "core"}:
        run_core_experiments(context, registry, only=only)
    if batch in {"all", "relations"}:
        run_relation_experiments(context, registry, only=only)
    if batch in {"all", "process"}:
        run_process_experiments(context, registry, only=only)
    if batch in {"all", "followup"}:
        run_followup_experiments(context, registry, only=only)
    if batch not in {"all", "core", "relations", "process", "followup"}:
        raise ValueError(f"Unknown experiment batch: {batch}")
    print(f"[experiments] batch={batch} finished in {time.perf_counter() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/exploration.yaml")
    parser.add_argument("--batch", default="all", choices=["all", "core", "relations", "process", "followup"])
    parser.add_argument("--experiment", action="append", default=None, help="Run only the named experiment id; may be repeated.")
    args = parser.parse_args()
    run(load_yaml(args.config), batch=args.batch, experiments=args.experiment)


if __name__ == "__main__":
    main()
