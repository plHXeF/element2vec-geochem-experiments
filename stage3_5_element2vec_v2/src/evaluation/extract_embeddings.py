from __future__ import annotations

import argparse
import copy
import hashlib

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.collate import collate_token_batch
from ..data.geochemical_coordinates import coordinate_spec, coordinate_fingerprint, GEO_VERSION
from ..data.dataset import WholeRockTokenDataset, load_vocab_bundle
from ..models.element2vec_v2 import build_model
from ..training.train_v2 import choose_device, move_batch_to_device
from ..utils.io import ensure_output_dirs, load_config, output_dir, save_json


def validate_checkpoint_compatibility(config: dict, checkpoint: dict, vocab) -> None:
    saved_config = checkpoint.get("config", {})
    requested_coordinate = config.get("data", {}).get("value_transform", {}).get("name", "v2_robust_z")
    saved_coordinate = saved_config.get("data", {}).get("value_transform", {}).get("name", "v2_robust_z")
    if requested_coordinate != saved_coordinate:
        raise ValueError("Checkpoint coordinate version does not match the requested coordinate")
    if requested_coordinate == GEO_VERSION:
        expected = coordinate_fingerprint(config)
        if checkpoint.get("coordinate_fingerprint") != expected or coordinate_fingerprint(saved_config) != expected:
            raise ValueError("Checkpoint coordinate reference values, scales or closure rules do not match")
        if checkpoint.get("coordinate_spec") != coordinate_spec(config):
            raise ValueError("Checkpoint is missing or has inconsistent coordinate metadata")
    requested_architecture = str(config["model"].get("architecture", "mlp_fusion"))
    saved_architecture = str(checkpoint.get("config", {}).get("model", {}).get("architecture", "mlp_fusion"))
    if requested_architecture != saved_architecture:
        raise ValueError(
            f"Checkpoint architecture is {saved_architecture!r}, but config requests {requested_architecture!r}"
        )
    if checkpoint.get("numeric_vocab") != vocab.numeric_vocab:
        raise ValueError("Checkpoint numeric vocabulary does not match the configured frozen vocabulary")
    if checkpoint.get("pure_vocab") != vocab.pure_vocab:
        raise ValueError("Checkpoint pure-label vocabulary does not match the configured frozen vocabulary")
    saved_stats = checkpoint.get("value_transform_stats")
    if requested_architecture == "value_rope" and saved_stats is None:
        raise ValueError("value_rope checkpoint is missing its value transformation statistics")
    if saved_stats is not None:
        current = vocab.transform_stats.to_dict(orient="records")
        if saved_stats != current:
            raise ValueError("Checkpoint value transformation statistics do not match the configured frozen statistics")


def _latent_rows(sample_ids, labels, latents, prefix: str) -> list[dict]:
    arr = latents.detach().cpu().numpy()
    rows = []
    for i, sample_id in enumerate(sample_ids):
        row = {"sample_id": sample_id}
        for key in ["rock_group", "rock_origin", "rock_type"]:
            row[key] = labels[i].get(key, "")
        row.update({f"{prefix}_{j}": arr[i, j] for j in range(arr.shape[1])})
        rows.append(row)
    return rows


@torch.no_grad()
def extract(config: dict, checkpoint: str, max_samples: int | None = None,
            sample_ids_path: str | None = None, label_mode: str = "masked") -> dict:
    if label_mode not in {"masked", "chemistry_only"}:
        raise ValueError("Unknown extraction label mode")
    if sample_ids_path and max_samples is not None:
        raise ValueError("Use either exact sample IDs or --max-samples, not both")
    if config["data"].get("value_transform", {}).get("name") == GEO_VERSION:
        if "value_rope_geo_mask15" not in output_dir(config).resolve().parts:
            raise ValueError("Geo extraction outputs must be in a separate value_rope_geo_mask15 directory")
    ensure_output_dirs(config)
    out = output_dir(config)
    vocab = load_vocab_bundle(config)
    device = choose_device(str(config["training"].get("device", "auto")))
    model = build_model(config, vocab).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    validate_checkpoint_compatibility(config, ckpt, vocab)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = WholeRockTokenDataset(config, vocab, mode="extract", max_samples=max_samples, mask_pure_labels=True)
    if sample_ids_path:
        wanted = pd.read_csv(sample_ids_path, dtype={"sample_id": str})["sample_id"]
        if wanted.empty or wanted.duplicated().any():
            raise ValueError("Paired extraction requires nonempty, unique sample IDs")
        indexed = dataset.df.copy()
        indexed.index = indexed[dataset.sample_col].astype(str)
        if indexed.index.duplicated().any() or not wanted.isin(indexed.index).all():
            raise ValueError("Paired sample IDs are missing or ambiguous in the database")
        dataset.df = indexed.loc[wanted.tolist()].reset_index(drop=True)
    if label_mode == "chemistry_only":
        dataset.chemistry_only = True
    loader = DataLoader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_token_batch,
    )
    cls_rows, mean_rows = [], []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        enc = model.encode(batch)
        encoded = enc["encoded"]
        cls_latent = encoded[:, 0, :]
        token_hidden = encoded[:, 1:, :]
        numeric_mask = batch["numeric_token_mask"] & batch["attention_mask"]
        denom = numeric_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        mean_numeric = (token_hidden * numeric_mask.unsqueeze(-1)).sum(dim=1) / denom
        cls_rows.extend(_latent_rows(raw_batch["sample_ids"], raw_batch["labels"], cls_latent, "latent"))
        mean_rows.extend(_latent_rows(raw_batch["sample_ids"], raw_batch["labels"], mean_numeric, "latent"))
    prefix = "chemistry_only_" if label_mode == "chemistry_only" else ""
    cls_path = out / "latents" / f"{prefix}sample_cls_latent.csv"
    mean_path = out / "latents" / f"{prefix}sample_mean_numeric_latent.csv"
    pd.DataFrame(cls_rows).to_csv(cls_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(mean_rows).to_csv(mean_path, index=False, encoding="utf-8-sig")
    if config["data"].get("value_transform", {}).get("name") == GEO_VERSION:
        ordered_ids = "\n".join(dataset.df[dataset.sample_col].astype(str).tolist())
        save_json({"samples": len(dataset), "label_mode": label_mode,
                   "ordered_sample_ids_sha256": hashlib.sha256(ordered_ids.encode()).hexdigest(),
                   "coordinate_fingerprint": coordinate_fingerprint(config),
                   "checkpoint": str(checkpoint)},
                  out / "logs" / f"{prefix}extraction_manifest.json")
    print(f"[extract_embeddings] wrote {cls_path}")
    print(f"[extract_embeddings] wrote {mean_path}")
    return {"cls": str(cls_path), "mean_numeric": str(mean_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_5_element2vec_v2/configs/v2_base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sqlite-path", default=None)
    parser.add_argument("--outputs-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-ids", default=None, help="Private CSV containing exact ordered sample_id values.")
    parser.add_argument("--label-mode", choices=["masked", "chemistry_only"], default="masked")
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.sqlite_path is not None:
        config["data"]["sqlite_path"] = args.sqlite_path
    if args.outputs_dir is not None:
        config["project"]["outputs_dir"] = args.outputs_dir
    if args.device is not None:
        config["training"]["device"] = args.device
    extract(config, args.checkpoint, max_samples=args.max_samples,
            sample_ids_path=args.sample_ids, label_mode=args.label_mode)


if __name__ == "__main__":
    main()
