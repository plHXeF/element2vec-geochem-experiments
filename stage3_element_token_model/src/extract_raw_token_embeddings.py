from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import build_model
from utils import OUTPUT_DIR, STAGE_DIR, ensure_output_dirs, load_config, save_dataframe, set_seed


OUT_DIR = OUTPUT_DIR / "raw_embedding_benchmark"


def load_checkpoint_model(config: dict):
    checkpoint = torch.load(OUTPUT_DIR / "checkpoints" / "token_transformer_last.pt", map_location="cpu", weights_only=False)
    elements = list(checkpoint["elements"])
    deposit_classes = list(checkpoint["deposit_classes"])
    model = build_model(config, len(elements), len(deposit_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, elements, deposit_classes, checkpoint


def raw_feature_columns(elements: list[str], emb_dim: int) -> list[str]:
    cols = []
    for element in elements:
        cols.append(f"{element}__x")
        cols.append(f"{element}__visible_mask")
        cols.extend(f"{element}__element_emb_{i}" for i in range(emb_dim))
    return cols


def export_raw_sample_tokens(config: dict) -> Path:
    ensure_output_dirs()
    (OUT_DIR / "embeddings").mkdir(parents=True, exist_ok=True)
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    sample_id = data["sample_id"].astype(str)
    deposit_type = data["deposit_type"].astype(str)
    x = data["element_values"].astype(np.float32)
    visible = data["visible_mask"].astype(np.float32)

    model, elements, deposit_classes, _ = load_checkpoint_model(config)
    element_embedding = model.element_embedding.weight.detach().cpu().numpy().astype(np.float32)
    deposit_embedding = model.deposit_embedding.weight.detach().cpu().numpy().astype(np.float32)
    emb_dim = element_embedding.shape[1]

    parts = []
    for j in range(len(elements)):
        emb = np.broadcast_to(element_embedding[j], (x.shape[0], emb_dim))
        parts.append(x[:, [j]])
        parts.append(visible[:, [j]])
        parts.append(emb)
    raw_token_flat = np.hstack(parts)
    df = pd.DataFrame(raw_token_flat, columns=raw_feature_columns(elements, emb_dim))
    df.insert(0, "deposit_type", deposit_type)
    df.insert(0, "sample_id", sample_id)
    raw_path = OUT_DIR / "embeddings" / "sample_raw_element_token_flat.csv"
    save_dataframe(df, raw_path)

    dep_df = pd.DataFrame(deposit_embedding[: len(deposit_classes)], columns=[f"emb_{i}" for i in range(deposit_embedding.shape[1])])
    dep_df.insert(0, "deposit_type", deposit_classes)
    save_dataframe(dep_df, OUT_DIR / "embeddings" / "deposit_type_embedding_nomask.csv")

    mask_df = pd.DataFrame(deposit_embedding[[len(deposit_classes)]], columns=[f"emb_{i}" for i in range(deposit_embedding.shape[1])])
    mask_df.insert(0, "deposit_type", ["[MASK_DEPOSIT]"])
    save_dataframe(mask_df, OUT_DIR / "embeddings" / "deposit_mask_embedding.csv")

    print(f"[raw token] wrote {raw_path} with {raw_token_flat.shape[1]} features")
    return raw_path


def embedding_learning_audit(config: dict) -> None:
    (OUT_DIR / "audit").mkdir(parents=True, exist_ok=True)
    model, elements, deposit_classes, _ = load_checkpoint_model(config)

    current_element = model.element_embedding.weight.detach().cpu().numpy()
    current_deposit = model.deposit_embedding.weight.detach().cpu().numpy()

    # Reconstruct the deterministic initial model used by train.py:
    # train.py calls set_seed(seed) before build_model(...).
    set_seed(int(config.get("random_seed", 42)))
    initial_model = build_model(config, len(elements), len(deposit_classes))
    initial_element = initial_model.element_embedding.weight.detach().cpu().numpy()
    initial_deposit = initial_model.deposit_embedding.weight.detach().cpu().numpy()

    rows = []
    for name, current, initial in [
        ("element_embedding.weight", current_element, initial_element),
        ("deposit_embedding.weight", current_deposit, initial_deposit),
    ]:
        diff = current - initial
        rows.append(
            {
                "parameter": name,
                "shape": "x".join(str(x) for x in current.shape),
                "requires_grad": bool(dict(model.named_parameters())[name].requires_grad),
                "l2_delta_from_deterministic_initialization": float(np.linalg.norm(diff)),
                "mean_abs_delta": float(np.mean(np.abs(diff))),
                "max_abs_delta": float(np.max(np.abs(diff))),
                "current_mean": float(np.mean(current)),
                "current_std": float(np.std(current)),
                "initial_mean": float(np.mean(initial)),
                "initial_std": float(np.std(initial)),
            }
        )
    audit = pd.DataFrame(rows)
    save_dataframe(audit, OUT_DIR / "audit" / "embedding_learning_audit.csv")

    param_names = [name for name, _ in model.named_parameters()]
    report = f"""# Embedding Learning Audit

## What The Code Does

- `model.py` defines `self.element_embedding = nn.Embedding(n_elements, element_embedding_dim)`.
- `model.py` defines `self.deposit_embedding = nn.Embedding(n_deposit_types + 1, deposit_embedding_dim)`.
- No fixed `_weight` is passed into either embedding layer, so PyTorch initializes these lookup tables randomly during model construction.
- `train.py` builds the model after `set_seed(random_seed)`, then optimizes `model.parameters()` with AdamW. This includes `element_embedding.weight` and `deposit_embedding.weight`.
- In `make_tokens`, element embeddings are concatenated with abundance and visible mask before projection:

```python
elem_input = torch.cat([x.unsqueeze(-1), visible_mask.unsqueeze(-1), elem_emb], dim=-1)
```

## Learned vs Deterministic Initial Weights

To verify the embeddings changed during training, this audit reconstructs a fresh model with the same seed and architecture, then compares its initial embedding weights with the checkpoint weights.

{audit.to_string(index=False)}

Interpretation: non-zero deltas confirm the embedding lookup matrices were randomly initialized and then updated by training.
"""
    (OUT_DIR / "audit" / "embedding_learning_audit.md").write_text(report, encoding="utf-8")
    print(f"[audit] wrote {OUT_DIR / 'audit' / 'embedding_learning_audit.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(STAGE_DIR / "configs" / "pyrite_token_transformer.yaml"))
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    config = load_config(args.config)
    export_raw_sample_tokens(config)
    embedding_learning_audit(config)


if __name__ == "__main__":
    main()
