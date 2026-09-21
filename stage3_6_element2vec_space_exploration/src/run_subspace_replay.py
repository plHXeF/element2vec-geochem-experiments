from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import load_yaml, resolve_path, set_reproducible_seed
from .experiment_context import ExperimentContext
from .experiments_core import run_core_experiments
from .experiments_followup import run_followup_experiments
from .experiments_process import run_process_experiments
from .experiments_relations import run_relation_experiments
from .pca_sweep_context import SharedInferenceCache
from .subspace_context import CROSS_SPACE_IDS, SubspaceExperimentContext, SubspaceRegistry
from .subspace_projection import ProjectionManager, ProjectionSpec


INVARIANT = {
    "B4_detection_limit_unavailable", "D3b_nonlinear_axis_failure", "G3_protolith_metamorphic",
    "I0_strict_cipw_unavailable", "K4_checkpoint_sensitivity_unavailable", "K5_umap_parameter_unavailable",
}


def _pools(ctx: ExperimentContext, root: Path) -> tuple[np.ndarray, np.ndarray]:
    fit_ids = pd.read_csv(root / "projections" / "fitting_pools" / "sample_fit_pool.csv")["sample_id"]
    token_pool = pd.read_csv(root / "projections" / "fitting_pools" / "token_fit_pool.csv")
    id_to_idx = pd.Series(np.arange(len(ctx.meta)), index=ctx.meta["sample_id"]).to_dict()
    fit_idx = np.asarray([id_to_idx[value] for value in fit_ids], dtype=int)
    token_key = pd.MultiIndex.from_frame(token_pool[["sample_id", "field"]])
    all_key = pd.MultiIndex.from_frame(ctx.token_meta[["sample_id", "field"]])
    # Duplicate sample/field pairs can exist; select exact deterministic prefix by
    # reproducing the audit's seeded candidate sampling when key matching is ambiguous.
    fit_set = set(fit_ids.tolist())
    candidates = np.flatnonzero(ctx.token_meta["sample_id"].isin(fit_set).to_numpy())
    rng = np.random.default_rng(ctx.seed + 17)
    token_idx = np.sort(rng.choice(candidates, size=len(token_pool), replace=False))
    return fit_idx, token_idx


def _run_batches(ctx: SubspaceExperimentContext, registry: SubspaceRegistry, only: set[str]) -> None:
    run_core_experiments(ctx, registry, only=only)
    run_relation_experiments(ctx, registry, only=only)
    run_process_experiments(ctx, registry, only=only)
    run_followup_experiments(ctx, registry, only=only)


def run(config: dict[str, Any], phase: str = "main") -> None:
    base_config = load_yaml(config["base_config"])
    settings = config["subspace_audit"]
    root = resolve_path(settings["output_root"])
    results_dir = root / "results"
    set_reproducible_seed(int(settings["seed"]), int(base_config["runtime"].get("num_threads", 8)))
    base = ExperimentContext(base_config)
    fit_idx, token_idx = _pools(base, root)
    manager = ProjectionManager(root / "projections", seed=int(settings["seed"]), dtype=settings["numerical_dtype"])
    inference_cache = SharedInferenceCache(base)
    original = pd.read_csv(resolve_path(base_config["project"]["results_dir"]) / "experiment_registry.csv")
    all_ids = set(original["experiment_id"].astype(str)) - INVARIANT
    if phase == "main":
        specs = [ProjectionSpec("PCA64", "PCA", 64), ProjectionSpec("PCA32", "PCA", 32),
                 ProjectionSpec("PCA3", "PCA", 3), ProjectionSpec("PCA2", "PCA", 2)]
    elif phase == "random":
        selected = set(settings["random_projection_experiment_ids"]) & all_ids
        all_ids = selected
        specs = [
            ProjectionSpec(f"RP{dim}_seed{seed:02d}", "RP", dim, seed=int(seed))
            for dim in settings["random_projection_dimensions"]
            for seed in settings["random_projection_seeds"]
        ]
    elif phase == "aux":
        specs = [
            ProjectionSpec("PCA64_WHITEN", "PCA", 64, whiten=True),
            ProjectionSpec("PCA32_WHITEN", "PCA", 32, whiten=True),
        ] + [ProjectionSpec(f"DROP_PC{count}", "DROP_PC", drop_count=count) for count in settings["drop_pc_counts"]]
        all_ids = all_ids & {
            "C1_same_value_context", "C2_element_role_proxies", "C3_contextual_neighbors",
            "D1_rock_continuity", "D2_local_geological_neighbors", "D3_global_axes",
            "F1_lithology_perturbation", "F2_single_element_jacobian", "F3_combined_interventions",
            "G1_label_neighbors", "G2_volcanic_plutonic", "G4_magmatic_sequences",
            "J1_beyond_traditional_indicators", "J2_grouped_increment", "K1_representation_baselines",
            "X1_context_raw_control", "X2_role_raw_control", "X4_random_missingness_followup",
        }
    else:
        raise ValueError(phase)

    compatibility = pd.read_csv(root / "audit" / "subspace_experiment_compatibility.csv")
    for spec in specs:
        complete = set()
        reg_path = results_dir / "experiment_registry.csv"
        if reg_path.exists():
            complete = set(pd.read_csv(reg_path)["experiment_id"].astype(str))
        target = {eid for eid in all_ids if f"{eid}__{spec.name}" not in complete}
        if spec.method == "PCA" and not spec.whiten:
            col = spec.name
            allowed = set(compatibility.loc[compatibility[col] == "applicable", "experiment_id"].astype(str))
            target &= allowed
        if not target:
            print(f"[subspace-replay] {spec.name}: complete")
            continue
        print(f"[subspace-replay] {spec.name}: {len(target)} experiments")
        ordinary = target - CROSS_SPACE_IDS
        cross = target & CROSS_SPACE_IDS
        if ordinary:
            ctx = SubspaceExperimentContext(base, manager, inference_cache, spec, fit_idx, token_idx, cross_space=False)
            _run_batches(ctx, SubspaceRegistry(results_dir, spec), ordinary)
            ctx.release()
        if cross:
            ctx = SubspaceExperimentContext(base, manager, inference_cache, spec, fit_idx, token_idx, cross_space=True)
            _run_batches(ctx, SubspaceRegistry(results_dir, spec), cross)
            ctx.release()
    manager.write_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml")
    parser.add_argument("--phase", choices=["main", "random", "aux"], default="main")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    run(load_yaml(args.config), args.phase)


if __name__ == "__main__":
    main()
