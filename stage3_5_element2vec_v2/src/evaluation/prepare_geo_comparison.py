"""Freeze the prior audit's exact paired IDs/pools and validate new coordinates."""

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args, run_dir, asset
if __name__ == "__main__":
    _require_run_dir()
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import WholeRockTokenDataset, load_vocab_bundle
from ..data.geochemical_coordinates import transform_numeric_frame, coordinate_spec, coordinate_fingerprint
from ..utils.io import load_config, output_dir, resolve_path, save_json


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(config, baseline_audit):
    destination = output_dir(config)/"audit"
    if "value_rope_geo_mask15" not in destination.resolve().parts and run_dir() is None:
        raise ValueError("Use a separate value_rope_geo_mask15 output directory")
    if destination.exists():
        raise FileExistsError("Paired audit already exists; do not overwrite its frozen inputs")
    baseline_audit = resolve_path(baseline_audit)
    old = json.loads((baseline_audit/"protocol.json").read_text())
    if sha(baseline_audit/"metadata.csv") != old["sample_hash"]:
        raise ValueError("Baseline metadata differs from its frozen protocol")
    meta = pd.read_csv(baseline_audit/"metadata.csv", dtype={"sample_id": str}, low_memory=False)
    if len(meta) != 12000 or meta.sample_id.duplicated().any():
        raise ValueError("Expected the original 12,000 unique sample IDs")
    with np.load(baseline_audit/"global_inputs.npz") as cached:
        fit, evaluation = cached["fit"].copy(), cached["evaluation"].copy()
    all_positions = np.concatenate([fit, evaluation])
    if len(fit) != 8000 or len(evaluation) != 4000 or not np.array_equal(np.sort(all_positions), np.arange(12000)):
        raise ValueError("Baseline fitting/evaluation pools are not the expected disjoint 8000/4000 partition")
    vocab = load_vocab_bundle(config)
    dataset = WholeRockTokenDataset(config, vocab, mode="extract")
    indexed = dataset.df.copy()
    indexed.index = indexed[dataset.sample_col].astype(str)
    if indexed.index.duplicated().any() or not meta.sample_id.isin(indexed.index).all():
        raise ValueError("Paired sample IDs missing or duplicated in the current database")
    selected = indexed.loc[meta.sample_id]
    expected, paired_qc = transform_numeric_frame(meta, config, vocab)
    cols = [f["db_column"] for f in config["data"]["numeric_fields"]]
    np.testing.assert_allclose(selected[cols].to_numpy(float), expected[cols].to_numpy(float),
                               atol=1e-9, rtol=1e-9, equal_nan=True,
                               err_msg="Current DB coordinates differ from baseline raw sample data")
    # Aggregate empirical mask density without exposing per-sample predictions.
    dataset.mode = "train"
    count, observed = 0, 0
    for i in range(min(4096, len(dataset))):
        item = dataset[(i, "numeric_value", None, 42+i)]
        count += len(item["numeric_value_targets_multi"])
        observed += sum(t["kind"] == 0 for t in item["tokens"])
    destination.mkdir(parents=True)
    meta[["sample_id"]].to_csv(destination/"paired_sample_ids.csv", index=False)
    np.savez(destination/"paired_pools.npz", fit=fit, evaluation=evaluation)
    protocol = {"experiment": "rope_geo_mask15", "status": "awaiting_user_training",
                "seed": old["seed"], "random_initialization_seeds": old["random_initialization_seeds"],
                "global_samples": len(meta), "fit": len(fit), "evaluation": len(evaluation),
                "baseline_protocol_sha256": sha(baseline_audit/"protocol.json"),
                "baseline_metadata_sha256": old["sample_hash"], "baseline_checkpoints": old["checkpoints"],
                "paired_ids_sha256": sha(destination/"paired_sample_ids.csv"),
                "paired_pools_sha256": sha(destination/"paired_pools.npz"),
                "coordinate_spec": coordinate_spec(config), "coordinate_fingerprint": coordinate_fingerprint(config),
                "full_data_coordinate_qc": dataset.coordinate_qc, "paired_coordinate_qc": paired_qc,
                "mask_check": {"numeric_task_samples": min(4096, len(dataset)),
                               "masked_tokens": count, "observed_tokens": observed,
                               "fraction": count/max(observed, 1)},
                "comparison_rule": "Identical physical perturbations transformed per model; coordinate-native tests reported separately."}
    save_json(protocol, destination/"protocol.json")
    print(json.dumps({"samples": len(dataset), "paired_samples": len(meta),
                      "closure_fraction": dataset.coordinate_qc["closure_fraction"],
                      "mask_fraction": count/max(observed, 1), "output": str(destination)}, indent=2))
    return protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml")
    parser.add_argument("--sqlite-path", required=True)
    parser.add_argument("--baseline-audit", default=str(asset("rope_z_run") / "audit"))
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    config = load_config(args.config)
    config["data"]["sqlite_path"] = args.sqlite_path
    prepare(config, args.baseline_audit)


if __name__ == "__main__":
    main()
