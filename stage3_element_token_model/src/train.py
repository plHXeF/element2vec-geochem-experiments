from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import build_model
from utils import OUTPUT_DIR, ensure_output_dirs, load_config, set_seed


def _masked_batch(
    x: torch.Tensor,
    visible: torch.Tensor,
    y: torch.Tensor,
    mask_deposit_index: int,
    element_mask_ratio: float,
    deposit_mask_prob: float,
    joint_mask_prob: float,
) -> dict[str, torch.Tensor]:
    elem_random = torch.rand_like(x) < element_mask_ratio
    elem_loss_mask = elem_random & (visible > 0.5)
    joint_mask = torch.rand(x.shape[0], device=x.device) < joint_mask_prob
    if joint_mask.any():
        elem_loss_mask[joint_mask] = (torch.rand_like(x[joint_mask]) < element_mask_ratio) & (visible[joint_mask] > 0.5)

    x_input = x.clone()
    visible_input = visible.clone()
    x_input[elem_loss_mask] = 0.0
    visible_input[elem_loss_mask] = 0.0

    deposit_mask = (torch.rand(x.shape[0], device=x.device) < deposit_mask_prob) | joint_mask
    deposit_ids = y.clone()
    deposit_ids[deposit_mask] = mask_deposit_index
    label_visible = (~deposit_mask).float()
    return {
        "x_input": x_input,
        "visible_input": visible_input,
        "deposit_ids": deposit_ids,
        "label_visible": label_visible,
        "elem_loss_mask": elem_loss_mask,
        "deposit_loss_mask": deposit_mask,
    }


def train(config: dict) -> dict:
    ensure_output_dirs()
    seed = int(config.get("random_seed", 42))
    set_seed(seed)

    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    x = torch.tensor(data["element_values"], dtype=torch.float32)
    visible = torch.tensor(data["visible_mask"], dtype=torch.float32)
    y = torch.tensor(data["label_id"], dtype=torch.long)
    elements = data["elements"].astype(str).tolist()
    deposit_classes = data["deposit_classes"].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, n_elements=len(elements), n_deposit_types=len(deposit_classes)).to(device)
    base_lr = float(config.get("learning_rate", 1e-3))
    min_lr = float(config.get("min_learning_rate", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    huber = nn.HuberLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    ce = nn.CrossEntropyLoss()
    elem_loss_fn = huber if str(config.get("loss", "huber")).lower() == "huber" else mse

    dataset = TensorDataset(x, visible, y)
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 128)), shuffle=True, drop_last=False)

    lambda_elem = float(config.get("lambda_elem", 1.0))
    lambda_type = float(config.get("lambda_type", 1.0))
    epochs = int(config.get("epochs", 100))
    scheduler_name = str(config.get("lr_scheduler", "none")).lower()
    scheduler = None
    if scheduler_name in {"cosine", "cosine_annealing", "cosineannealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    history = []
    start_time = time.time()

    print(f"[train] device={device} samples={len(dataset)} epochs={epochs} lr={base_lr} scheduler={scheduler_name}")
    for epoch in range(1, epochs + 1):
        model.train()
        rows = []
        for xb, vb, yb in loader:
            xb = xb.to(device)
            vb = vb.to(device)
            yb = yb.to(device)
            masked = _masked_batch(
                xb,
                vb,
                yb,
                model.mask_deposit_index,
                float(config.get("element_mask_ratio", 0.15)),
                float(config.get("deposit_mask_prob", 0.5)),
                float(config.get("joint_mask_prob", 0.2)),
            )
            out = model(masked["x_input"], masked["visible_input"], masked["deposit_ids"], masked["label_visible"])
            elem_mask = masked["elem_loss_mask"]
            if elem_mask.any():
                elem_loss = elem_loss_fn(out["element_pred"], xb)[elem_mask].mean()
            else:
                elem_loss = out["element_pred"].sum() * 0.0

            dep_mask = masked["deposit_loss_mask"]
            if dep_mask.any():
                type_loss = ce(out["deposit_logits"][dep_mask], yb[dep_mask])
                type_pred = out["deposit_logits"][dep_mask].argmax(dim=1)
                type_acc = (type_pred == yb[dep_mask]).float().mean()
            else:
                type_loss = out["deposit_logits"].sum() * 0.0
                type_acc = torch.tensor(float("nan"), device=device)

            loss = lambda_elem * elem_loss + lambda_type * type_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "element_loss": float(elem_loss.detach().cpu()),
                    "type_loss": float(type_loss.detach().cpu()),
                    "masked_type_accuracy": float(type_acc.detach().cpu()) if torch.isfinite(type_acc) else np.nan,
                }
            )
        epoch_row = pd.DataFrame(rows).mean(numeric_only=True).to_dict()
        epoch_row["epoch"] = epoch
        epoch_row["learning_rate"] = optimizer.param_groups[0]["lr"]
        epoch_row["elapsed_sec"] = time.time() - start_time
        history.append(epoch_row)
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"[train] epoch={epoch:03d} loss={epoch_row['loss']:.4f} "
                f"elem={epoch_row['element_loss']:.4f} type={epoch_row['type_loss']:.4f} "
                f"masked_type_acc={epoch_row['masked_type_accuracy']:.4f} "
                f"lr={epoch_row['learning_rate']:.6g}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "elements": elements,
        "deposit_classes": deposit_classes,
    }
    torch.save(checkpoint, OUTPUT_DIR / "checkpoints" / "token_transformer_last.pt")
    hist = pd.DataFrame(history)
    hist.to_csv(OUTPUT_DIR / "logs" / "training_log.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["epoch"], hist["loss"], label="total")
    ax.plot(hist["epoch"], hist["element_loss"], label="element")
    ax.plot(hist["epoch"], hist["type_loss"], label="deposit type")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Stage 3 masked modeling loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "figures" / "training_loss_curve.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figures" / "training_loss_curve.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(hist["epoch"], hist["learning_rate"], color="#7b3294", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Cosine annealing learning rate")
    ax.grid(True, alpha=0.25)
    fig.savefig(OUTPUT_DIR / "figures" / "learning_rate_curve.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "figures" / "learning_rate_curve.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"[train] saved checkpoint: {OUTPUT_DIR / 'checkpoints' / 'token_transformer_last.pt'}")
    return {"device": str(device), "epochs": epochs, "final_loss": float(hist.iloc[-1]["loss"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "pyrite_token_transformer.yaml"))
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    train(load_config(args.config))


if __name__ == "__main__":
    main()
