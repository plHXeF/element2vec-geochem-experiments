from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import build_model
from utils import OUTPUT_DIR, ensure_output_dirs, load_config, save_dataframe


def _load_model(config: dict):
    checkpoint = torch.load(OUTPUT_DIR / "checkpoints" / "token_transformer_last.pt", map_location="cpu", weights_only=False)
    elements = list(checkpoint["elements"])
    deposit_classes = list(checkpoint["deposit_classes"])
    model = build_model(config, len(elements), len(deposit_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, elements, deposit_classes


def _pool(encoded: torch.Tensor, n_elements: int, pooling: str) -> torch.Tensor:
    if pooling == "mean_element":
        return encoded[:, :n_elements, :].mean(dim=1)
    if pooling == "deposit":
        return encoded[:, n_elements, :]
    if pooling in {"mean_all", "all"}:
        return encoded.mean(dim=1)
    raise ValueError(f"Unknown pooling: {pooling}")


def extract_one(config: dict, mode: str, pooling: str) -> Path:
    ensure_output_dirs()
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    sample_id = data["sample_id"].astype(str)
    deposit_type = data["deposit_type"].astype(str)
    x = torch.tensor(data["element_values"], dtype=torch.float32)
    visible = torch.tensor(data["visible_mask"], dtype=torch.float32)
    y = torch.tensor(data["label_id"], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = _load_model(config)
    model = model.to(device)
    loader = DataLoader(TensorDataset(x, visible, y), batch_size=int(config.get("batch_size", 128)), shuffle=False)

    parts = []
    with torch.no_grad():
        for xb, vb, yb in loader:
            xb = xb.to(device)
            vb = vb.to(device)
            yb = yb.to(device)
            if mode == "label_masked":
                dep_ids = torch.full((xb.shape[0],), model.mask_deposit_index, dtype=torch.long, device=device)
                label_visible = torch.zeros(xb.shape[0], dtype=torch.float32, device=device)
            elif mode == "label_visible":
                dep_ids = yb
                label_visible = torch.ones(xb.shape[0], dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            _, encoded = model.encode(xb, vb, dep_ids, label_visible)
            latent = _pool(encoded, model.n_elements, pooling).detach().cpu().numpy()
            parts.append(latent)

    latent_all = np.vstack(parts)
    df = pd.DataFrame(latent_all, columns=[f"latent_{i}" for i in range(latent_all.shape[1])])
    df.insert(0, "deposit_type", deposit_type)
    df.insert(0, "sample_id", sample_id)
    path = OUTPUT_DIR / "latents" / f"transformer_latent_{mode}_{pooling}.csv"
    save_dataframe(df, path)
    print(f"[extract_latent] wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pyrite_token_transformer.yaml"))
    parser.add_argument("--mode", choices=["label_masked", "label_visible", "both"], default="label_masked")
    parser.add_argument("--pooling", choices=["mean_element", "deposit", "mean_all"], default="mean_element")
    args = parser.parse_args()
    config = load_config(args.config)
    modes = ["label_masked", "label_visible"] if args.mode == "both" else [args.mode]
    for mode in modes:
        extract_one(config, mode, args.pooling)


if __name__ == "__main__":
    main()
