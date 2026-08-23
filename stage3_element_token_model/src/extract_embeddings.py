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


def _embedding_frame(names: list[str], emb: np.ndarray, name_col: str) -> pd.DataFrame:
    df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    df.insert(0, name_col, names)
    return df


def extract_embeddings(config: dict) -> None:
    ensure_output_dirs()
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    sample_id = data["sample_id"].astype(str)
    deposit_type = data["deposit_type"].astype(str)
    x = torch.tensor(data["element_values"], dtype=torch.float32)
    visible = torch.tensor(data["visible_mask"], dtype=torch.float32)
    y = torch.tensor(data["label_id"], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, elements, deposit_classes = _load_model(config)
    model = model.to(device)

    elem_emb = model.element_embedding.weight.detach().cpu().numpy()
    dep_names = deposit_classes + ["[MASK_DEPOSIT]"]
    dep_emb = model.deposit_embedding.weight.detach().cpu().numpy()
    save_dataframe(_embedding_frame(elements, elem_emb, "element"), OUTPUT_DIR / "embeddings" / "element_embedding.csv")
    save_dataframe(_embedding_frame(dep_names, dep_emb, "deposit_type"), OUTPUT_DIR / "embeddings" / "deposit_embedding.csv")

    loader = DataLoader(TensorDataset(x, visible, y), batch_size=int(config.get("batch_size", 128)), shuffle=False)
    input_parts = []
    encoded_parts = []
    with torch.no_grad():
        for xb, vb, _ in loader:
            xb = xb.to(device)
            vb = vb.to(device)
            dep_ids = torch.full((xb.shape[0],), model.mask_deposit_index, dtype=torch.long, device=device)
            label_visible = torch.zeros(xb.shape[0], dtype=torch.float32, device=device)
            input_tokens, encoded = model.encode(xb, vb, dep_ids, label_visible)
            input_elem = input_tokens[:, : model.n_elements, :].reshape(xb.shape[0], -1).detach().cpu().numpy()
            encoded_elem = encoded[:, : model.n_elements, :].reshape(xb.shape[0], -1).detach().cpu().numpy()
            input_parts.append(input_elem)
            encoded_parts.append(encoded_elem)

    input_flat = np.vstack(input_parts)
    encoded_flat = np.vstack(encoded_parts)
    input_df = pd.DataFrame(input_flat, columns=[f"tok_{i}" for i in range(input_flat.shape[1])])
    input_df.insert(0, "deposit_type", deposit_type)
    input_df.insert(0, "sample_id", sample_id)
    encoded_df = pd.DataFrame(encoded_flat, columns=[f"tok_{i}" for i in range(encoded_flat.shape[1])])
    encoded_df.insert(0, "deposit_type", deposit_type)
    encoded_df.insert(0, "sample_id", sample_id)

    save_dataframe(input_df, OUTPUT_DIR / "embeddings" / "sample_input_token_flat.csv")
    save_dataframe(encoded_df, OUTPUT_DIR / "embeddings" / "sample_encoded_token_flat.csv")
    save_dataframe(encoded_df, OUTPUT_DIR / "embeddings" / "sample_token_matrix_flat.csv")
    print("[extract_embeddings] wrote element/deposit embeddings and sample token flats")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pyrite_token_transformer.yaml"))
    args = parser.parse_args()
    extract_embeddings(load_config(args.config))


if __name__ == "__main__":
    main()
