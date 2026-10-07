"""All mutations require a registered mutable run; training is an explicit verb."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import geochem_workspace as ws
from .prepare import PreparedData, prepare_database, save_json, digest_file
from .training import build, train, device_for, seed_all, signature
from .batching import make_batch, to_device
from .losses import per_sample_loss
from .config import configuration
from .model import DIMENSIONS
from .schema import FIELDS, EXACT, dictionary


def verify_data(data):
    n = len(data); original = 0; nonempty = 0; invalid_finite = 0
    from .schema import RAW_TAGS, INDEX
    columns = [INDEX[x] for x in RAW_TAGS]
    for start in range(0, n, 8192):
        v = data.values[start:start+8192]; s = data.states[start:start+8192]
        nonempty += int((s != 0).any(1).sum())
        original += int((s[:, columns] != 0).sum())
        # Trace entries that were absent and then derived must not count as raw.
        original -= int(np.isin(s[:, columns], [3, 5]).sum())
        invalid_finite += int((np.isin(s, [1, 2, 3, 4, 5]) & ~np.isfinite(v)).sum())
    assert nonempty == n and original == data.coverage["original_observations"] and invalid_finite == 0
    assert len(set(map(int, data.sample_ids))) == n
    pool = __import__("pandas").read_csv(data.root/"cohort.csv")
    assert pool.groupby("component").split.nunique().max() == 1
    assert np.all(np.isfinite(data.weights)) and np.all(data.weights > 0)
    return {"passed": True, "rows": n, "original_observations": original,
            "derived_exact": data.coverage["derived_exact"], "lower_bounds": data.coverage["lower_bound_observations"],
            "unique_ids": n, "sample_level_outputs_private": True,
            "prepared_fingerprint": data.fingerprint}


def matrix(data):
    rows = []
    for arm in ("fixed", "native"):
        for d in DIMENSIONS:
            c = configuration(arm, d)
            with torch.device("meta"):
                model = build(c, data)
            rows.append({"arm": arm, "embedding_dim": d, "token_dim": model.d_model,
                         "attention_heads": model.layers[0].self_attn.num_heads, **model.parameter_counts(),
                         "epochs": 50, "steps": math.ceil(len(data)/1024)*50, "sample_presentations": len(data)*50})
    return rows


def preflight(config, data, root):
    device = device_for(config["training"]["device"])
    seed_all(config["training"]["seed"])
    model = build(config, data).to(device)
    size = min(config["training"]["micro_batch"], len(data))
    # Use largest real panels, not only sparse rows which underestimate memory.
    with __import__("sqlite3").connect(data.root/"metadata.sqlite") as conn:
        ids = [r[0] for r in conn.execute("SELECT row_index FROM samples ORDER BY observed_count DESC,row_index LIMIT ?", (size,))]
    before = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    elapsed = []
    try:
        for task in (0, 2):
            model.zero_grad(set_to_none=True)
            batch = to_device(make_batch(data, ids, model, forced_task=task), device)
            if device.type == "mps": torch.mps.synchronize()
            start = time.monotonic()
            outputs = model(batch)
            loss, _ = per_sample_loss(outputs, batch, torch.ones(len(FIELDS), device=device))
            loss.mean().backward()
            if device.type == "mps": torch.mps.synchronize()
            elapsed.append(time.monotonic()-start)
            if not torch.isfinite(loss).all(): raise FloatingPointError("Preflight loss not finite")
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Preflight gradient not finite")
        assert all(torch.equal(v.cpu(), before[k]) for k,v in model.state_dict().items())
        estimate = model.parameter_counts()["total"]*16  # weights, gradients, two Adam moments
        report = {"passed": True, "optimizer_updates": 0, "weights_unchanged": True,
                  "device": str(device), "micro_batch": size, "largest_raw_panel": int(batch["mask"].sum(1).max()),
                  "seconds_per_microbatch": elapsed[-1], "estimated_epoch_seconds": elapsed[-1]*len(data)/size,
                  "parameter_state_bytes_estimate": estimate, "parameter_counts": model.parameter_counts(),
                  "prepared_fingerprint": data.fingerprint,
                  "config_signature": signature(config), "finite_gradients": True,
                  "note": "forward/backward feasibility only; optimizer/memory/IO costs make runtime an estimate"}
        if device.type == "mps": report["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        save_json(root/"preflight.json", report)
        return report
    finally:
        del model, before
        if device.type == "mps": torch.mps.empty_cache()


def verify_training(root, data, config):
    report = json.loads((root/"training_complete.json").read_text())
    from .training import signature
    assert report["config_signature"] == signature(config)
    assert report["prepared_fingerprint"] == data.fingerprint
    assert report["presentations"] == len(data)*config["training"]["epochs"]
    assert report["step"] == math.ceil(len(data)/config["training"]["effective_batch"])*config["training"]["epochs"]
    for epoch in range(1, config["training"]["epochs"]+1):
        coverage = json.loads((root/"logs"/f"coverage-epoch-{epoch:03d}.json").read_text())
        assert coverage["unique_rows"] == len(data) and coverage["min_visits"] == coverage["max_visits"] == 1
    ck = torch.load(root/"checkpoints"/"latest.pt", map_location="cpu", weights_only=False)
    assert ck["cursor"]["presentations"] == report["presentations"]
    assert all(torch.isfinite(v).all() for v in ck["model_state"].values() if v.is_floating_point())
    return {"passed": True, "rows": len(data), "epochs": config["training"]["epochs"], "presentations": report["presentations"],
            "checkpoint_sha256": digest_file(root/"checkpoints"/"latest.pt")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify-data", "matrix", "preflight", "train", "verify-training", "extract", "evaluate"])
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/e010/prepare.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Extraction device only")
    args = parser.parse_args()
    root = ws.require_run_dir(); ws.guard_args(args)
    cfg = ws.load_config_file(args.config)
    if args.command == "prepare":
        report = prepare_database(ws.resolve_path(cfg["source_asset"]), root/"prepared_data", cfg.get("expected_rows"), censor_rules=cfg.get("censor_rules", []))
        print(json.dumps({k: report[k] for k in ("rows", "original_observations", "derived_exact", "lower_bound_observations")}))
        return
    prepared = cfg.get("prepared_asset")
    data = PreparedData(ws.resolve_path(prepared) if prepared else root/"prepared_data", rehash=True)
    if cfg.get("expected_rows") is not None and len(data) != cfg["expected_rows"]:
        raise ValueError("Prepared row count differs from the registered configuration")
    if args.command == "verify-data":
        report = verify_data(data); save_json(root/"data_verification.json", report)
    elif args.command == "matrix":
        report = matrix(data); save_json(root/"parameter_matrix.json", report)
    elif args.command == "preflight": report = preflight(cfg, data, root)
    elif args.command == "train":
        proof = json.loads((root/"preflight.json").read_text())
        if not proof["passed"] or proof["prepared_fingerprint"] != data.fingerprint or proof.get("config_signature") != signature(cfg):
            raise ValueError("Matching preflight is required before training")
        report = train(cfg, data, root, args.resume)
    elif args.command == "verify-training":
        report = verify_training(root, data, cfg); save_json(root/"training_verification.json", report)
    elif args.command == "extract":
        if not args.checkpoint: raise ValueError("--checkpoint is required")
        from .evaluation import extract
        extract(ws.resolve_path(args.checkpoint), data, root, args.device)
        report = {"extracted": True}
    else:
        from .evaluation import evaluate
        report = {"metrics": len(evaluate(data, root, args.device))}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
