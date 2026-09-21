from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args, run_dir, asset
if __name__ == "__main__":
    _require_run_dir()

import argparse
import copy
import math
import random
import gc
from collections import deque
from pathlib import Path

import pandas as pd
import torch

from ..data.build_vocab import build_vocab
from ..data.collate import collate_token_batch
from ..data.geochemical_coordinates import coordinate_spec, coordinate_fingerprint, GEO_VERSION
from ..utils.io import save_json
from ..data.dataset import (
    ID_TO_TASK,
    TASK_TO_ID,
    WholeRockTokenDataset,
    get_pure_label_probabilities,
    get_task_probabilities,
    load_vocab_bundle,
)
from ..models.element2vec_v2 import build_model
from ..training.losses import compute_losses
from ..training.scheduler import build_warmup_cosine_scheduler
from ..utils.io import ensure_output_dirs, load_config, output_dir, vocab_dir
from ..utils.logging import CsvLogger
from ..utils.seed import set_seed


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if torch.is_tensor(v) else v
    return moved


def device_memory_stats(device: torch.device) -> dict[str, float]:
    stats: dict[str, float] = {}
    if device.type == "mps" and hasattr(torch, "mps"):
        if hasattr(torch.mps, "current_allocated_memory"):
            stats["mps_allocated_mb"] = float(torch.mps.current_allocated_memory()) / (1024**2)
        if hasattr(torch.mps, "driver_allocated_memory"):
            stats["mps_driver_allocated_mb"] = float(torch.mps.driver_allocated_memory()) / (1024**2)
        if hasattr(torch.mps, "recommended_max_memory"):
            stats["mps_recommended_max_mb"] = float(torch.mps.recommended_max_memory()) / (1024**2)
    elif device.type == "cuda":
        stats["cuda_allocated_mb"] = float(torch.cuda.memory_allocated(device)) / (1024**2)
        stats["cuda_reserved_mb"] = float(torch.cuda.memory_reserved(device)) / (1024**2)
        stats["cuda_max_allocated_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024**2)
    return stats


def release_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def export_embedding_table(weight: torch.Tensor, names: list[str], path: Path, name_col: str) -> None:
    arr = weight.detach().cpu().numpy()
    rows = []
    for i, name in enumerate(names):
        row = {name_col: name, "index": i}
        row.update({f"dim_{j}": arr[i, j] for j in range(arr.shape[1])})
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def export_model_embeddings(model, vocab, out: Path, prefix: str) -> None:
    numeric_names = vocab.numeric_vocab["numeric_tags"] + [vocab.numeric_vocab["mask_numeric_tag"]]
    export_embedding_table(
        model.numeric_tag_embedding.weight,
        numeric_names,
        out / "embeddings" / f"{prefix}_numeric_tag_embeddings.csv",
        "numeric_tag",
    )
    export_embedding_table(
        model.pure_label_embedding.weight,
        vocab.pure_vocab["pure_tokens"],
        out / "embeddings" / f"{prefix}_pure_label_embeddings.csv",
        "pure_label_token",
    )


def save_checkpoints(model, optimizer, scheduler, config, vocab, step: int, out: Path) -> None:
    out_ckpt = out / "checkpoints"
    out_ckpt.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "token_projector"):
        torch.save(model.token_projector.state_dict(), out_ckpt / "token_projector.pt")
    torch.save(model.transformer_encoder.state_dict(), out_ckpt / "transformer_encoder.pt")
    torch.save(model.decoders.state_dict(), out_ckpt / "decoders.pt")
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "coordinate_spec": coordinate_spec(config) if "value_transform" in config["data"] else None,
            "coordinate_fingerprint": coordinate_fingerprint(config) if "numeric_fields" in config["data"] else None,
            "numeric_vocab": vocab.numeric_vocab,
            "pure_vocab": vocab.pure_vocab,
            "value_transform_stats": vocab.transform_stats.to_dict(orient="records"),
        },
        out_ckpt / "full_model.pt",
    )


def resolve_training_steps(config: dict, dataset_size: int, max_steps_override: int | None = None) -> tuple[int, int, int | None]:
    micro_batch_size = int(config["training"]["batch_size"])
    grad_accum = max(int(config["training"].get("gradient_accumulation_steps", 1)), 1)
    micro_batches_per_epoch = math.ceil(dataset_size / micro_batch_size)
    steps_per_epoch = micro_batches_per_epoch // grad_accum
    if steps_per_epoch < 1:
        raise ValueError(
            f"Dataset yields {micro_batches_per_epoch} micro-batch(es), fewer than "
            f"gradient_accumulation_steps={grad_accum}; increase --max-samples or lower --batch-size."
        )
    target_epochs = config["training"].get("epochs", config["training"].get("target_epochs"))
    if max_steps_override is not None:
        return int(max_steps_override), steps_per_epoch, int(target_epochs) if target_epochs else None
    configured_steps = config["training"].get("max_steps")
    if configured_steps is not None:
        return int(configured_steps), steps_per_epoch, int(target_epochs) if target_epochs else None
    if target_epochs is not None:
        return int(math.ceil(float(target_epochs) * steps_per_epoch)), steps_per_epoch, int(target_epochs)
    raise ValueError("Set training.max_steps, training.target_epochs, or pass --max-steps.")


def resolve_save_every(config: dict, steps_per_epoch: int) -> int:
    value = config["training"].get("save_every", "epoch")
    if isinstance(value, str) and value.lower() == "epoch":
        return steps_per_epoch
    return int(value)


def resolve_log_every(config: dict) -> int:
    if "logging" in config and "log_every_steps" in config["logging"]:
        return int(config["logging"]["log_every_steps"])
    return int(config["training"].get("log_every", 20))


def resolve_batch_log_every(config: dict) -> int:
    return max(int(config.get("logging", {}).get("batch_log_every", 0) or 0), 0)


def make_task_plan(config: dict, batch_size: int, rng: random.Random) -> list[tuple[str, str | None]]:
    task_probs = get_task_probabilities(config)
    pure_probs = get_pure_label_probabilities(config)
    tasks = list(task_probs)
    raw_counts = {task: task_probs[task] * batch_size for task in tasks}
    counts = {task: int(math.floor(raw_counts[task])) for task in tasks}
    remainder = batch_size - sum(counts.values())
    ranked = sorted(tasks, key=lambda task: raw_counts[task] - counts[task], reverse=True)
    for task in ranked[:remainder]:
        counts[task] += 1
    plan: list[tuple[str, str | None]] = []
    for task in tasks:
        for _ in range(counts[task]):
            pure_field = None
            if task == "mask_pure_label":
                pure_field = weighted_choice(list(pure_probs), pure_probs, rng)
            plan.append((task, pure_field))
    rng.shuffle(plan)
    return plan


def weighted_choice(items: list[str], weights: dict[str, float], rng: random.Random) -> str:
    if not items:
        return ""
    vals = [max(float(weights.get(item, 0.0)), 0.0) for item in items]
    if sum(vals) <= 0:
        vals = [1.0] * len(items)
    return rng.choices(items, weights=vals, k=1)[0]


def iter_epoch_batches(dataset: WholeRockTokenDataset, config: dict, rng: random.Random):
    batch_size = int(config["training"]["batch_size"])
    balanced = bool(config.get("masking", {}).get("task_balanced_batch", False))
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        if balanced:
            plan = make_task_plan(config, len(chunk), rng)
            items = [dataset[(idx, task, pure_field, None)] for idx, (task, pure_field) in zip(chunk, plan)]
        else:
            items = [dataset[idx] for idx in chunk]
        yield collate_token_batch(items)


def build_fixed_probe_batches(dataset: WholeRockTokenDataset, config: dict) -> list[dict]:
    probe_cfg = config.get("probe", {})
    if not probe_cfg.get("enabled", False):
        return []
    size = min(int(probe_cfg.get("size", 8192)), len(dataset))
    batch_size = int(probe_cfg.get("batch_size", config["training"]["batch_size"]))
    rng = random.Random(int(config["training"]["seed"]) + 104729)
    if probe_cfg.get("fixed_samples", True):
        indices = rng.sample(range(len(dataset)), size) if size < len(dataset) else list(range(len(dataset)))
    else:
        indices = [rng.randrange(len(dataset)) for _ in range(size)]
    batches = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        plan = make_task_plan(config, len(chunk), rng) if probe_cfg.get("fixed_masks", True) else [(None, None)] * len(chunk)
        items = []
        for local_i, (idx, (task, pure_field)) in enumerate(zip(chunk, plan)):
            seed = int(config["training"]["seed"]) + 1000003 + start + local_i if probe_cfg.get("fixed_masks", True) else None
            items.append(dataset[(idx, task, pure_field, seed)])
        batches.append(collate_token_batch(items))
    return batches


def task_count_row(batch: dict) -> dict[str, int]:
    task_ids = batch["task_ids"]
    return {
        "mask_numeric_value_count": int((task_ids == TASK_TO_ID["mask_numeric_value"]).sum().detach().cpu()),
        "mask_numeric_tag_count": int((task_ids == TASK_TO_ID["mask_numeric_tag"]).sum().detach().cpu()),
        "mask_pure_label_count": int((task_ids == TASK_TO_ID["mask_pure_label"]).sum().detach().cpu()),
    }


def numeric_mask_counts(batch: dict) -> dict[str, int]:
    eligible = (batch["task_ids"] == 0)[:, None] & batch["numeric_token_mask"] & batch["attention_mask"]
    selected = batch.get("numeric_prediction_mask", batch["mask_value_flags"].bool())
    return {"numeric_masked_tokens": int((selected & eligible).sum().detach().cpu()),
            "numeric_eligible_tokens": int(eligible.sum().detach().cpu())}


def average_rows(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if vals:
            out[key] = float(sum(vals) / len(vals))
    return out


class EmaTracker:
    def __init__(self, window: int):
        self.window = max(int(window), 1)
        self.values: dict[str, deque] = {}

    def update(self, row: dict, keys: list[str]) -> dict:
        out = {}
        for key in keys:
            self.values.setdefault(key, deque(maxlen=self.window)).append(float(row[key]))
            out[f"{key}_ema{self.window}"] = sum(self.values[key]) / len(self.values[key])
        return out


@torch.no_grad()
def evaluate_probe(model, probe_batches: list[dict], device: torch.device, pure_fields: list[str], config: dict) -> dict:
    model.eval()
    totals: dict[str, float] = {}
    total_samples = 0
    numeric_total, numeric_count = 0.0, 0
    multi_value = False
    for raw_batch in probe_batches:
        batch = move_batch_to_device(raw_batch, device)
        outputs = model(batch)
        losses = compute_losses(outputs, batch, pure_fields, config["training"]["loss_weights"])
        batch_size = int(batch["attention_mask"].shape[0])
        total_samples += batch_size
        if "numeric_prediction_mask" in batch:
            multi_value = True
            count = numeric_mask_counts(batch)["numeric_masked_tokens"]
            numeric_total += float(losses["numeric_value_loss"].detach().cpu()) * count
            numeric_count += count
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
        del batch, outputs, losses
    if total_samples == 0:
        return {}
    result = {f"probe_{key}": value / total_samples for key, value in totals.items()}
    if multi_value:
        old_numeric = result["probe_numeric_value_loss"]
        result["probe_numeric_value_loss"] = numeric_total / max(numeric_count, 1)
        result["probe_total_loss"] += float(config["training"]["loss_weights"].get("numeric_value", 1.0)) * (result["probe_numeric_value_loss"] - old_numeric)
        result["probe_numeric_masked_tokens"] = numeric_count
    return result


def train(config: dict, max_steps_override: int | None = None, max_samples: int | None = None) -> dict:
    if config["data"].get("value_transform", {}).get("name") == GEO_VERSION:
        destination = output_dir(config).resolve()
        if "value_rope_geo_mask15" not in destination.parts and run_dir() is None:
            raise ValueError("Geo experiment outputs must be inside a separate value_rope_geo_mask15 directory")
        if (destination / "checkpoints/full_model.pt").exists() or (destination / "logs/train_log.csv").exists():
            raise FileExistsError("Geo training output already exists; choose a fresh --outputs-dir to preserve the run")
    ensure_output_dirs(config)
    out = output_dir(config)
    set_seed(int(config["training"]["seed"]))
    vocab_root = vocab_dir(config)
    required = [vocab_root / "numeric_tag_vocab.json", vocab_root / "pure_label_vocab.json", vocab_root / "value_transform_stats.csv"]
    if not all(p.exists() for p in required):
        if config.get("data", {}).get("vocab_dir"):
            missing = ", ".join(str(p) for p in required if not p.exists())
            raise FileNotFoundError(f"Frozen vocabulary is incomplete; missing: {missing}")
        build_vocab(config)
    vocab = load_vocab_bundle(config)
    dataset = WholeRockTokenDataset(config, vocab, mode="train", max_samples=max_samples)
    if config["data"].get("value_transform", {}).get("name") == GEO_VERSION:
        save_json({"config": config, "coordinate_qc": dataset.coordinate_qc,
                   "max_steps_override": max_steps_override, "max_samples_override": max_samples,
                   "validation": "full-data training; fixed training probe, no held-out validation"},
                  out / "logs/run_manifest.json")
    device = choose_device(str(config["training"].get("device", "auto")))
    model = build_model(config, vocab).to(device)
    export_model_embeddings(model, vocab, out, "initial")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    max_steps, steps_per_epoch, target_epochs = resolve_training_steps(config, len(dataset), max_steps_override)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        max_steps,
        int(config["training"]["warmup_steps"]),
        min_lr=float(config["training"].get("min_lr", 0.0)),
    )
    logger = CsvLogger(out / "logs" / "train_log.csv")
    probe_logger = CsvLogger(out / "logs" / "probe_log.csv")
    pure_fields = config["data"]["pure_label_fields"]
    log_every = resolve_log_every(config)
    batch_log_every = resolve_batch_log_every(config)
    save_every = resolve_save_every(config, steps_per_epoch)
    grad_accum_steps = max(int(config["training"].get("gradient_accumulation_steps", 1)), 1)
    micro_batches_per_epoch = math.ceil(len(dataset) / int(config["training"]["batch_size"]))
    ema = EmaTracker(int(config.get("logging", {}).get("ema_window", 500)))
    rng = random.Random(int(config["training"]["seed"]))
    probe_batches = build_fixed_probe_batches(dataset, config)
    memory_cfg = config.get("memory", {})
    log_memory = bool(memory_cfg.get("log_memory", True))
    probe_every = int(config.get("probe", {}).get("eval_every_steps", 500))
    empty_cache_every = int(memory_cfg.get("empty_cache_every_steps", 0) or 0)
    step = 0
    print(
        f"[train_v2] device={device} samples={len(dataset)} batch_size={config['training']['batch_size']} "
        f"grad_accum={grad_accum_steps} micro_batches_per_epoch={micro_batches_per_epoch} "
        f"steps_per_epoch={steps_per_epoch} target_epochs={target_epochs} max_steps={max_steps}"
    )
    if probe_batches:
        probe_row = {"step": 0, "epoch": 0.0, **evaluate_probe(model, probe_batches, device, pure_fields, config)}
        if log_memory:
            probe_row.update(device_memory_stats(device))
        probe_logger.append(probe_row)
        print(f"[probe] step=0 loss={probe_row.get('probe_total_loss', float('nan')):.4f}")
        if memory_cfg.get("empty_cache_after_probe", True):
            release_device_cache(device)
    epoch_number = 0
    while step < max_steps:
        epoch_number += 1
        accum_rows: list[dict] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_number, raw_batch in enumerate(iter_epoch_batches(dataset, config, rng), start=1):
            if batch_log_every and (
                batch_number == 1
                or batch_number == micro_batches_per_epoch
                or batch_number % batch_log_every == 0
            ):
                counts = task_count_row(raw_batch)
                target_epoch_text = str(target_epochs) if target_epochs is not None else "?"
                print(
                    f"[batch] epoch={epoch_number}/{target_epoch_text} "
                    f"micro_batch={batch_number}/{micro_batches_per_epoch} "
                    f"optimizer_step={step}/{max_steps} size={raw_batch['attention_mask'].shape[0]} "
                    f"tasks(value/tag/label)="
                    f"{counts['mask_numeric_value_count']}/"
                    f"{counts['mask_numeric_tag_count']}/"
                    f"{counts['mask_pure_label_count']} "
                    f"numeric_masked/eligible={numeric_mask_counts(raw_batch)['numeric_masked_tokens']}/"
                    f"{numeric_mask_counts(raw_batch)['numeric_eligible_tokens']}"
                )
            batch = move_batch_to_device(raw_batch, device)
            model.train()
            outputs = model(batch)
            losses = compute_losses(outputs, batch, pure_fields, config["training"]["loss_weights"])
            (losses["total_loss"] / grad_accum_steps).backward()
            loss_row = {
                "total_loss": float(losses["total_loss"].detach().cpu()),
                "numeric_value_loss": float(losses["numeric_value_loss"].detach().cpu()),
                "numeric_tag_loss": float(losses["numeric_tag_loss"].detach().cpu()),
                "rock_group_loss": float(losses["rock_group_loss"].detach().cpu()),
                "rock_origin_loss": float(losses["rock_origin_loss"].detach().cpu()),
                "rock_type_loss": float(losses["rock_type_loss"].detach().cpu()),
                "batch_size": int(batch["attention_mask"].shape[0]),
                "num_tokens_mean": float(batch["attention_mask"].sum(dim=1).float().mean().detach().cpu()),
                "num_tokens_max": int(batch["attention_mask"].sum(dim=1).max().detach().cpu()),
                **task_count_row(batch),
                **numeric_mask_counts(batch),
            }
            accum_rows.append(loss_row)
            if len(accum_rows) < grad_accum_steps:
                continue
            step += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            avg = average_rows(accum_rows)
            row = {
                "step": step,
                "epoch": step / steps_per_epoch,
                "total_loss": avg["total_loss"],
                "numeric_value_loss": avg["numeric_value_loss"],
                "numeric_tag_loss": avg["numeric_tag_loss"],
                "rock_group_loss": avg["rock_group_loss"],
                "rock_origin_loss": avg["rock_origin_loss"],
                "rock_type_loss": avg["rock_type_loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": int(avg["batch_size"]),
                "effective_batch_size": int(avg["batch_size"] * grad_accum_steps),
                "num_tokens_mean": avg["num_tokens_mean"],
                "num_tokens_max": int(avg["num_tokens_max"]),
                "mask_task_type": "task_balanced" if config.get("masking", {}).get("task_balanced_batch", False) else ID_TO_TASK.get(int(batch["task_ids"][0].detach().cpu()), "none"),
                "mask_numeric_value_count": int(avg["mask_numeric_value_count"]),
                "mask_numeric_tag_count": int(avg["mask_numeric_tag_count"]),
                "mask_pure_label_count": int(avg["mask_pure_label_count"]),
            }
            if log_memory:
                row.update(device_memory_stats(device))
            row["numeric_masked_tokens"] = sum(r["numeric_masked_tokens"] for r in accum_rows)
            row["numeric_eligible_tokens"] = sum(r["numeric_eligible_tokens"] for r in accum_rows)
            row["numeric_mask_fraction"] = row["numeric_masked_tokens"] / max(row["numeric_eligible_tokens"], 1)
            row.update(
                ema.update(
                    row,
                    [
                        "total_loss",
                        "numeric_value_loss",
                        "numeric_tag_loss",
                        "rock_group_loss",
                        "rock_origin_loss",
                        "rock_type_loss",
                    ],
                )
            )
            logger.append(row)
            if step == 1 or step % log_every == 0:
                print(
                    f"[train_v2] step={step} epoch={row['epoch']:.2f} loss={row['total_loss']:.4f} "
                    f"ema={row.get('total_loss_ema500', row['total_loss']):.4f} "
                    f"num={row['numeric_value_loss']:.4f} tag={row['numeric_tag_loss']:.4f} "
                    f"rg={row['rock_group_loss']:.4f} ro={row['rock_origin_loss']:.4f} rt={row['rock_type_loss']:.4f} "
                    f"lr={row['learning_rate']:.3g}"
                )
            if probe_batches and (step % probe_every == 0):
                probe_row = {"step": step, "epoch": step / steps_per_epoch, **evaluate_probe(model, probe_batches, device, pure_fields, config)}
                if log_memory:
                    probe_row.update(device_memory_stats(device))
                probe_logger.append(probe_row)
                print(f"[probe] step={step} loss={probe_row.get('probe_total_loss', float('nan')):.4f}")
                if memory_cfg.get("empty_cache_after_probe", True):
                    release_device_cache(device)
            if step % save_every == 0:
                save_checkpoints(model, optimizer, scheduler, config, vocab, step, out)
                if memory_cfg.get("empty_cache_after_checkpoint", True):
                    release_device_cache(device)
            if empty_cache_every > 0 and step % empty_cache_every == 0:
                release_device_cache(device)
            del outputs, losses, batch
            accum_rows = []
            if step >= max_steps:
                break
    save_checkpoints(model, optimizer, scheduler, config, vocab, step, out)
    export_model_embeddings(model, vocab, out, "final")
    return {"steps": step, "steps_per_epoch": steps_per_epoch, "checkpoint": str(out / "checkpoints" / "full_model.pt")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/v2_base.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--probe-size", type=int, default=None)
    parser.add_argument("--probe-batch-size", type=int, default=None)
    parser.add_argument(
        "--batch-log-every",
        type=int,
        default=None,
        help="Print explicit batch progress every N micro-batches; 0 disables it.",
    )
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    config = copy.deepcopy(load_config(args.config))
    if args.sqlite_path is not None:
        config["data"]["sqlite_path"] = args.sqlite_path
    if args.outputs_dir is not None:
        config["project"]["outputs_dir"] = args.outputs_dir
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.probe_size is not None:
        config.setdefault("probe", {})["size"] = args.probe_size
    if args.probe_batch_size is not None:
        config.setdefault("probe", {})["batch_size"] = args.probe_batch_size
    if args.batch_log_every is not None:
        config.setdefault("logging", {})["batch_log_every"] = args.batch_log_every
    train(config, max_steps_override=args.max_steps, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
