from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import gc
import hashlib
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from stage3_5_element2vec_v2.src.data.collate import collate_token_batch
from stage3_5_element2vec_v2.src.data.dataset import load_vocab_bundle
from stage3_5_element2vec_v2.src.data.transforms import apply_value_transform
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
from stage3_5_element2vec_v2.src.training.train_v2 import choose_device, move_batch_to_device
from stage3_5_element2vec_v2.src.utils.io import load_config as load_source_config

from .common import (
    ensure_directories,
    load_yaml,
    resolve_path,
    save_json,
    set_reproducible_seed,
    stratified_sample_indices,
)


METADATA_COLUMNS = [
    "sample_id",
    "canonical_ref_id",
    "author",
    "data_source",
    "year",
    "latitude",
    "longitude",
    "rock_name",
    "protolith",
    "material",
    "rock_group",
    "rock_origin",
    "rock_facies",
    "rock_type",
    "age",
    "mg_number",
    "asi",
    "mali",
    "maficity",
    "cia",
    "wip",
    "spar",
    "qtzindex",
    "sr87_sr86",
    "epsilon_nd",
    "epsilon_hf",
    "pb206_pb204",
    "pb207_pb204",
    "pb208_pb204",
]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _query_sample_frame(config: dict, source_config: dict) -> pd.DataFrame:
    db_path = resolve_path(config["source"]["sqlite_path"])
    view = config["source"]["view_name"]
    seed = int(config["sampling"]["seed"])
    n = int(config["sampling"]["main_samples"])
    stratify = str(config["sampling"].get("stratify_by", "rock_type"))
    numeric_cols = [item["db_column"] for item in source_config["data"]["numeric_fields"]]
    with sqlite3.connect(db_path) as conn:
        strata = pd.read_sql_query(
            f"SELECT sample_id, {_quote(stratify)} FROM {_quote(view)} ORDER BY sample_id",
            conn,
        )
        selected_pos = stratified_sample_indices(strata, stratify, n=n, seed=seed)
        selected_ids = strata.iloc[selected_pos]["sample_id"].astype(int).tolist()
        columns = list(dict.fromkeys(METADATA_COLUMNS + numeric_cols))
        frames = []
        for start in range(0, len(selected_ids), 800):
            chunk = selected_ids[start : start + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                f"SELECT {', '.join(_quote(c) for c in columns)} FROM {_quote(view)} "
                f"WHERE sample_id IN ({placeholders})"
            )
            frames.append(pd.read_sql_query(query, conn, params=chunk))
    frame = pd.concat(frames, ignore_index=True).sort_values("sample_id").reset_index(drop=True)
    if len(frame) != len(set(selected_ids)):
        raise RuntimeError(f"Selected {len(set(selected_ids))} unique ids but queried {len(frame)} rows")
    return frame


def _add_transformed_values(frame: pd.DataFrame, source_config: dict, vocab) -> pd.DataFrame:
    frame = frame.copy()
    stats_by_col = {row["db_column"]: row.to_dict() for _, row in vocab.transform_stats.iterrows()}
    clip_z = source_config["data"]["value_transform"].get("clip_z")
    observed_cols = []
    for item in source_config["data"]["numeric_fields"]:
        col = item["db_column"]
        z_col = f"z__{item['name']}"
        frame[z_col] = apply_value_transform(frame[col], stats_by_col[col], clip_z=clip_z)
        observed_cols.append(z_col)
    frame["num_numeric_tokens"] = frame[observed_cols].notna().sum(axis=1).astype(int)
    frame["missing_fraction"] = 1.0 - frame["num_numeric_tokens"] / len(observed_cols)
    frame["total_alkali"] = frame["na2o"] + frame["k2o"]
    frame["log_sr_y"] = _safe_log_ratio(frame["sr_ppm"], frame["y_ppm"])
    frame["log_la_yb"] = _safe_log_ratio(frame["la_ppm"], frame["yb_ppm"])
    frame["log_th_sc"] = _safe_log_ratio(frame["th_ppm"], frame["sc_ppm"])
    frame["log_ce_pb"] = _safe_log_ratio(frame["ce_ppm"], frame["pb_ppm"])
    frame["log_nb_u"] = _safe_log_ratio(frame["nb_ppm"], frame["u_ppm"])
    return frame


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    valid = (a > 0) & (b > 0)
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out.loc[valid] = np.log10(a.loc[valid] / b.loc[valid])
    return out


def _make_item(row: pd.Series, source_config: dict, vocab, pure_mode: str) -> dict[str, Any]:
    numeric_tokens = []
    tag_to_idx = vocab.numeric_vocab["tag_to_idx"]
    for item in source_config["data"]["numeric_fields"]:
        value = row[f"z__{item['name']}"]
        if pd.notna(value):
            numeric_tokens.append(
                {
                    "kind": 0,
                    "name": item["name"],
                    "numeric_tag_id": int(tag_to_idx[item["name"]]),
                    "pure_label_id": 0,
                    "value": float(value),
                    "mask_tag": 0.0,
                    "mask_value": 0.0,
                }
            )
    pure_tokens = []
    labels: dict[str, str] = {}
    if pure_mode not in {"numeric_only", "none"}:
        for field in source_config["data"]["pure_label_fields"]:
            raw = row.get(field)
            if pd.isna(raw) or not str(raw).strip():
                continue
            value = str(raw).strip()
            labels[field] = value
            if pure_mode == "visible_labels":
                token = f"{field}:{value}"
                mask_tag = 0.0
            elif pure_mode == "masked_labels":
                token = vocab.pure_vocab["mask_tokens"][field]
                mask_tag = 1.0
            else:
                raise ValueError(f"Unknown pure_mode={pure_mode}")
            token_id = vocab.pure_vocab["token_to_idx"].get(token)
            if token_id is None:
                continue
            pure_tokens.append(
                {
                    "kind": 1,
                    "name": token,
                    "numeric_tag_id": 0,
                    "pure_label_id": int(token_id),
                    "value": 0.0,
                    "mask_tag": mask_tag,
                    "mask_value": 0.0,
                    "field": field,
                    "label": value,
                }
            )
    return {
        "sample_id": int(row["sample_id"]),
        "tokens": numeric_tokens + pure_tokens,
        "target": {
            "task": "none",
            "task_id": -1,
            "mask_position": 0,
            "numeric_value": 0.0,
            "numeric_tag_id": -100,
            "pure_field": "",
            "pure_class_id": -100,
        },
        "labels": labels,
    }


class TokenReservoir:
    def __init__(self, per_field: int, seed: int):
        self.per_field = int(per_field)
        self.rng = np.random.default_rng(seed)
        self.seen: dict[str, int] = defaultdict(int)
        self.records: dict[str, list[tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = defaultdict(list)

    def add(self, field: str, projected: np.ndarray, contextual: np.ndarray, metadata: dict[str, Any]) -> None:
        self.seen[field] += 1
        records = self.records[field]
        payload = (projected.astype(np.float32, copy=True), contextual.astype(np.float32, copy=True), metadata)
        if len(records) < self.per_field:
            records.append(payload)
            return
        replacement = int(self.rng.integers(0, self.seen[field]))
        if replacement < self.per_field:
            records[replacement] = payload

    def arrays(self) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        projected, contextual, metadata = [], [], []
        for field in sorted(self.records):
            for proj, ctx, meta in self.records[field]:
                projected.append(proj)
                contextual.append(ctx)
                metadata.append(meta)
        return np.stack(projected), np.stack(contextual), pd.DataFrame(metadata)


@torch.no_grad()
def _extract_mode(
    model,
    frame: pd.DataFrame,
    source_config: dict,
    vocab,
    device: torch.device,
    batch_size: int,
    pure_mode: str,
    reservoir: TokenReservoir | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs = {"projected_mean_numeric": [], "encoded_mean_numeric": [], "cls": []}
    fields = vocab.numeric_vocab["numeric_tags"]
    total_batches = int(np.ceil(len(frame) / batch_size))
    for batch_i, start in enumerate(range(0, len(frame), batch_size), start=1):
        chunk = frame.iloc[start : start + batch_size]
        items = [_make_item(row, source_config, vocab, pure_mode) for _, row in chunk.iterrows()]
        raw_batch = collate_token_batch(items)
        batch = move_batch_to_device(raw_batch, device)
        encoded = model.encode(batch)
        projected = encoded["projected_tokens"]
        contextual = encoded["encoded"][:, 1:, :]
        numeric_mask = batch["numeric_token_mask"] & batch["attention_mask"]
        denom = numeric_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        projected_mean = (projected * numeric_mask.unsqueeze(-1)).sum(dim=1) / denom
        contextual_mean = (contextual * numeric_mask.unsqueeze(-1)).sum(dim=1) / denom
        outputs["projected_mean_numeric"].append(projected_mean.cpu().numpy().astype(np.float32))
        outputs["encoded_mean_numeric"].append(contextual_mean.cpu().numpy().astype(np.float32))
        outputs["cls"].append(encoded["encoded"][:, 0, :].cpu().numpy().astype(np.float32))
        if reservoir is not None:
            proj_cpu = projected.cpu().numpy()
            ctx_cpu = contextual.cpu().numpy()
            num_ids = raw_batch["numeric_tag_ids"].numpy()
            num_mask = raw_batch["numeric_token_mask"].numpy()
            vals = raw_batch["values"].numpy()
            for i, sample_id in enumerate(raw_batch["sample_ids"]):
                for j in np.flatnonzero(num_mask[i]):
                    field = fields[int(num_ids[i, j])]
                    reservoir.add(
                        field,
                        proj_cpu[i, j],
                        ctx_cpu[i, j],
                        {
                            "sample_id": int(sample_id),
                            "field": field,
                            "normalized_value": float(vals[i, j]),
                            "rock_group": str(chunk.iloc[i].get("rock_group", "")),
                            "rock_origin": str(chunk.iloc[i].get("rock_origin", "")),
                            "rock_type": str(chunk.iloc[i].get("rock_type", "")),
                            "data_source": str(chunk.iloc[i].get("data_source", "")),
                            "canonical_ref_id": chunk.iloc[i].get("canonical_ref_id"),
                        },
                    )
        if batch_i == 1 or batch_i % 20 == 0 or batch_i == total_batches:
            print(f"[extract:{pure_mode}] batch {batch_i}/{total_batches}")
        del batch, encoded, projected, contextual, projected_mean, contextual_mean
    return {key: np.concatenate(parts, axis=0) for key, parts in outputs.items()}


def _save_static_embeddings(model, cache_dir: Path, prefix: str) -> None:
    np.savez_compressed(
        cache_dir / f"static_embeddings_{prefix}.npz",
        numeric=model.numeric_tag_embedding.weight.detach().cpu().numpy().astype(np.float32),
        pure=model.pure_label_embedding.weight.detach().cpu().numpy().astype(np.float32),
        cls=model.cls_token.detach().cpu().numpy().astype(np.float32),
    )


def _load_trained_model(config: dict, source_config: dict, vocab, device: torch.device):
    checkpoint_path = resolve_path(config["source"]["checkpoint"])
    model = build_model(source_config, vocab)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    step = int(checkpoint.get("step", -1))
    del checkpoint
    gc.collect()
    return model.to(device), step


def extract(config: dict) -> None:
    started = time.perf_counter()
    paths = ensure_directories(config)
    cache_dir = paths["cache"]
    seed = int(config["sampling"]["seed"])
    set_reproducible_seed(seed, int(config["runtime"].get("num_threads", 8)))
    source_config = load_source_config(config["source"]["model_config"])
    vocab = load_vocab_bundle(source_config)
    frame = _add_transformed_values(_query_sample_frame(config, source_config), source_config, vocab)
    frame.to_csv(cache_dir / "sample_metadata.csv", index=False, encoding="utf-8-sig")
    sample_hash = hashlib.sha256(",".join(frame["sample_id"].astype(str)).encode("utf-8")).hexdigest()
    device_name = str(config["runtime"].get("device", "auto"))
    device = choose_device(device_name)
    batch_size = int(config["sampling"]["batch_size"])
    print(f"[cache] samples={len(frame)} device={device} batch_size={batch_size}")

    # Recreate the training-time random initialization for a low-cost negative control.
    set_reproducible_seed(seed, int(config["runtime"].get("num_threads", 8)))
    random_model = build_model(source_config, vocab)
    _save_static_embeddings(random_model, cache_dir, "random")
    random_n = min(int(config["sampling"]["random_model_control_samples"]), len(frame))
    random_model = random_model.to(device)
    random_latents = _extract_mode(
        random_model,
        frame.iloc[:random_n],
        source_config,
        vocab,
        device,
        batch_size,
        "numeric_only",
    )
    np.savez_compressed(cache_dir / "latents_random_numeric_only.npz", **random_latents)
    del random_model, random_latents
    gc.collect()
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    trained_model, checkpoint_step = _load_trained_model(config, source_config, vocab, device)
    _save_static_embeddings(trained_model, cache_dir, "trained")
    reservoir = TokenReservoir(int(config["sampling"]["token_reservoir_per_field"]), seed + 17)
    primary = _extract_mode(
        trained_model,
        frame,
        source_config,
        vocab,
        device,
        batch_size,
        "numeric_only",
        reservoir=reservoir,
    )
    np.savez_compressed(cache_dir / "latents_numeric_only.npz", **primary)
    proj_tok, ctx_tok, token_meta = reservoir.arrays()
    np.savez_compressed(cache_dir / "token_reservoir_numeric_only.npz", projected=proj_tok, contextual=ctx_tok)
    token_meta.to_csv(cache_dir / "token_reservoir_metadata.csv", index=False, encoding="utf-8-sig")
    del primary, proj_tok, ctx_tok, reservoir

    for mode in config["analysis"].get("auxiliary_input_modes", []):
        mode_latents = _extract_mode(trained_model, frame, source_config, vocab, device, batch_size, mode)
        np.savez_compressed(cache_dir / f"latents_{mode}.npz", **mode_latents)
        del mode_latents

    visible_n = min(int(config["sampling"]["visible_label_control_samples"]), len(frame))
    visible = _extract_mode(
        trained_model,
        frame.iloc[:visible_n],
        source_config,
        vocab,
        device,
        batch_size,
        "visible_labels",
    )
    np.savez_compressed(cache_dir / "latents_visible_labels_control.npz", **visible)
    del visible, trained_model
    gc.collect()

    # Save stable names separately so NPZ arrays remain numeric-only and mmap-friendly.
    pd.DataFrame(
        {
            "numeric_index": range(len(vocab.numeric_vocab["numeric_tags"]) + 1),
            "numeric_tag": vocab.numeric_vocab["numeric_tags"] + [vocab.numeric_vocab["mask_numeric_tag"]],
        }
    ).to_csv(cache_dir / "numeric_embedding_names.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {"pure_index": range(len(vocab.pure_vocab["pure_tokens"])), "pure_label_token": vocab.pure_vocab["pure_tokens"]}
    ).to_csv(cache_dir / "pure_embedding_names.csv", index=False, encoding="utf-8-sig")

    checkpoint_path = resolve_path(config["source"]["checkpoint"])
    manifest = {
        "sample_count": len(frame),
        "sample_id_sha256": sample_hash,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "device": str(device),
        "batch_size": batch_size,
        "token_reservoir_rows": len(token_meta),
        "numeric_fields": vocab.numeric_vocab["numeric_tags"],
        "elapsed_seconds": time.perf_counter() - started,
        "files": sorted(path.name for path in cache_dir.iterdir() if path.is_file()),
    }
    save_json(manifest, cache_dir / "cache_manifest.json")
    print(f"[cache] complete in {manifest['elapsed_seconds']:.1f}s; wrote {cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/exploration.yaml")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    extract(load_yaml(args.config))


if __name__ == "__main__":
    main()
