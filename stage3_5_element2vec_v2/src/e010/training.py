"""Explicit training entry point with full-pass accounting and strict resume."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .model import E010Model
from .prepare import save_json, digest_file
from .batching import make_batch, to_device
from .losses import per_sample_loss
from .schema import FIELDS, LABELS


def signature(config):
    # Device and physical asset location do not define the scientific model.
    c = copy.deepcopy(config)
    c.pop("prepared_asset", None); c.pop("source_asset", None)
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def device_for(name):
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available. Refusing to silently run the formal scan on CPU")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(name)


def active_tags(data):
    return [data.coverage["by_field"][f]["missing"] < len(data) for f in FIELDS]


def build(config, data):
    return E010Model(config, {k: len(data.label_vocab[k]) for k in LABELS}, active_tags(data))


def optimizer_for(model, config):
    # Unobserved entries retain their exact initialization. Decay must not move
    # unused rows of a dense embedding parameter.
    embeddings, other = [], []
    for name, p in model.named_parameters():
        (embeddings if "embedding" in name else other).append(p)
    return torch.optim.AdamW([{"params": embeddings, "weight_decay": 0.},
                             {"params": other, "weight_decay": config["training"]["weight_decay"]}],
                            lr=config["training"]["learning_rate"])


def rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "mps": state["device"] = torch.mps.get_rng_state()
    if device.type == "cuda": state["device"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, device):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if device.type == "mps": torch.mps.set_rng_state(state["device"])
    if device.type == "cuda": torch.cuda.set_rng_state_all(state["device"])


def save_checkpoint(path, model, config, data, optimizer=None, cursor=None, device=None):
    payload = {"e010_version": 1, "config": config, "config_signature": signature(config),
               "prepared_fingerprint": data.fingerprint, "dictionary": json.loads((data.root/"dictionary.json").read_text()),
               "statistics": data.stats, "label_vocab": data.label_vocab, "active_tags": active_tags(data),
               "censor_rules": getattr(data, "manifest", {}).get("censor_rules", []),
               "model_state": model.state_dict(), "cursor": cursor, "parameter_counts": model.parameter_counts(),
               "torch_version": torch.__version__, "device": str(device)}
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), rng=rng_state(device))
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save(payload, tmp); tmp.replace(p)


def load_model(checkpoint, device="cpu"):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck.get("e010_version") != 1 or signature(ck["config"]) != ck["config_signature"]:
        raise ValueError("Invalid E010 checkpoint/version/signature")
    from .schema import dictionary
    if ck["dictionary"] != dictionary():
        raise ValueError("Checkpoint dictionary incompatible with implementation")
    model = E010Model(ck["config"], {k: len(ck["label_vocab"][k]) for k in LABELS}, ck["active_tags"])
    model.load_state_dict(ck["model_state"], strict=True)
    return model.to(device), ck


@torch.no_grad()
def fixed_probe(model, data, config, field_weights, device):
    """A fixed, label-visible pretraining task diagnostic, never validation."""
    import pandas as pd
    ids = pd.read_csv(data.root/"cohort.csv").row_index.to_numpy(np.int64)
    ids = np.random.default_rng(4201).permutation(ids)[:config["training"]["probe_size"]]
    model.eval(); totals = {k: 0. for k in ("loss", "numeric", "tag", "label")}
    denominator = float(np.sum(data.weights[ids], dtype=np.float64))
    for start in range(0, len(ids), config["training"]["micro_batch"]):
        b = to_device(make_batch(data,ids[start:start+config["training"]["micro_batch"]],model,seed=4201,epoch=0),device)
        losses, parts = per_sample_loss(model(b),b,field_weights,config.get("loss","student_t"))
        totals["loss"] += float((losses*b["weights"]).sum().cpu())/denominator
        for k,v in parts.items(): totals[k] += float((v*b["weights"]).sum().cpu())/denominator
    if not all(math.isfinite(v) for v in totals.values()): raise FloatingPointError("Non-finite fixed probe")
    return {"rows":len(ids),"fixed_mask_seed":4201,"boundary":"pretraining-exposed diagnostic; not validation",**totals}


def train(config, data, root, resume=False):
    root = Path(root); device = device_for(config["training"]["device"])
    cfg = config["training"]; seed = int(cfg["seed"])
    seed_all(seed); model = build(config, data).to(device)
    optimizer = optimizer_for(model, config)
    n = len(data); effective = int(cfg["effective_batch"]); micro = int(cfg["micro_batch"])
    if effective < 1 or micro < 1 or micro > effective:
        raise ValueError("Invalid micro/effective batch size")
    epochs = int(cfg["epochs"]); total_steps = math.ceil(n/effective)*epochs
    ckpt = root / "checkpoints"; ckpt.mkdir(exist_ok=True)
    log_path = root / "logs" / "training.jsonl"; log_path.parent.mkdir(exist_ok=True)
    cursor = {"epoch": 0, "offset": 0, "step": 0, "presentations": 0}
    if resume:
        saved = torch.load(ckpt / "latest.pt", map_location="cpu", weights_only=False)
        if saved["config_signature"] != signature(config) or saved["prepared_fingerprint"] != data.fingerprint:
            raise ValueError("Resume configuration or input fingerprint changed")
        if saved["device"] != str(device) or saved["torch_version"] != torch.__version__:
            raise ValueError("Strict resume requires the same device backend and PyTorch version")
        model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer"])
        cursor = saved["cursor"]; restore_rng(saved["rng"], device)
        if log_path.exists():
            # Entries after the last durable checkpoint came from an interrupted
            # attempt. Preserve them separately before truncating the replay log.
            lines = log_path.read_text().splitlines()
            keep = [x for x in lines if json.loads(x)["step"] <= cursor["step"]]
            if len(keep) != len(lines):
                (root/"logs"/f"interrupted-{time.time_ns()}.jsonl").write_text("\n".join(lines)+"\n")
                log_path.write_text("\n".join(keep)+("\n" if keep else ""))
    elif (ckpt / "latest.pt").exists() or log_path.exists():
        raise FileExistsError("Training already exists; use validated resume or a new run")
    else:
        save_checkpoint(ckpt/"initial.pt", model, config, data, device=device)
    counts = np.array([s["n_exact"] for s in data.stats], float)
    median = float(np.median(counts[counts > 0])) if (counts > 0).any() else 1.
    fw = np.clip(np.sqrt(median/np.maximum(counts, 1)), .25, 4)
    field_weights = torch.tensor(fw, dtype=torch.float32, device=device)
    save_json(root / "training_contract.json", {"rows_per_epoch": n, "epochs": epochs, "steps": total_steps,
              "presentations": n*epochs, "parameter_counts": model.parameter_counts(),
              "config_signature": signature(config), "prepared_fingerprint": data.fingerprint,
              "evaluation_boundary": "full-data and label pretraining exposure; no external-generalization claim"})
    for epoch in range(cursor["epoch"], epochs):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 8701])).permutation(n)
        start_at = cursor["offset"] if epoch == cursor["epoch"] else 0
        visited = np.zeros(n, np.uint8); visited[order[:start_at]] = 1
        for start in range(start_at, n, effective):
            start_time = time.monotonic()
            selection = order[start:start+effective]
            denominator = float(np.sum(data.weights[selection], dtype=np.float64))
            optimizer.zero_grad(set_to_none=True); model.train()
            totals = {k: 0. for k in ("loss", "numeric", "tag", "label")}
            targets = 0; informative = 0
            step = cursor["step"] + 1
            warm = min(1., step/max(1, cfg["warmup_steps"]))
            progress = max(0., (step-cfg["warmup_steps"])/max(1, total_steps-cfg["warmup_steps"]))
            lr = warm*(cfg["min_lr"] + .5*(cfg["learning_rate"]-cfg["min_lr"])*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups: group["lr"] = lr
            for offset in range(0, len(selection), micro):
                ids = selection[offset:offset+micro]
                batch = to_device(make_batch(data, ids, model, seed=seed, epoch=epoch), device)
                out = model(batch)
                losses, parts = per_sample_loss(out, batch, field_weights, config.get("loss", "student_t"))
                loss = (losses*batch["weights"]).sum()/denominator
                if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss; no optimizer update made")
                loss.backward()
                totals["loss"] += float(loss.detach().cpu())
                for k, v in parts.items(): totals[k] += float((v*batch["weights"]).sum().detach().cpu())/denominator
                targets += int(batch["numeric_target"].sum().cpu()); informative += int((batch["task"] >= 0).sum().cpu())
                visited[ids] += 1
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            end = min(n, start+effective)
            cursor = {"epoch": epoch+1 if end == n else epoch, "offset": 0 if end == n else end,
                      "step": step, "presentations": epoch*n+end}
            row = {"step": step, "epoch": epoch+end/n, "samples": len(selection), "presentations": cursor["presentations"],
                   "targets": targets, "informative_rows": informative, "lr": lr, "gradient_norm": float(norm.cpu()),
                   "seconds": time.monotonic()-start_time, **totals}
            with log_path.open("a") as f: f.write(json.dumps(row)+"\n")
            if step % cfg["log_every"] == 0 or end == n: print(json.dumps(row), flush=True)
            if cfg.get("probe_size",0) and (step % cfg.get("probe_every",500) == 0 or end == n):
                probe = fixed_probe(model,data,config,field_weights,device)
                save_json(root/"logs"/f"probe-step-{step:06d}.json", {"step":step,**probe})
            if end == n:
                if not np.all(visited == 1): raise AssertionError("Epoch coverage mismatch")
                save_json(root / "logs" / f"coverage-epoch-{epoch+1:03d}.json",
                          {"epoch": epoch+1, "unique_rows": int(np.count_nonzero(visited)), "min_visits": int(visited.min()),
                           "max_visits": int(visited.max()), "sample_id_sha256": digest_file(data.root/"sample_ids.npy")})
                if epoch+1 in set(cfg["retain_epochs"]) | {epochs}:
                    save_checkpoint(ckpt/f"epoch-{epoch+1:03d}.pt", model, config, data, cursor=cursor, device=device)
            if step % cfg["checkpoint_every"] == 0 or end == n:
                save_checkpoint(ckpt/"latest.pt", model, config, data, optimizer, cursor, device)
        if not np.all(visited == 1): raise AssertionError("Epoch coverage mismatch")
    if cursor["presentations"] != n*epochs: raise AssertionError("Incomplete full-data exposure")
    save_json(root/"training_complete.json", {"passed": True, **cursor, "config_signature": signature(config),
                                             "prepared_fingerprint": data.fingerprint})
    return cursor
