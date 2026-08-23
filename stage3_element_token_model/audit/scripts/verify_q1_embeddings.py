"""
Q1: 验证元素/矿床 Embedding 是否被反向传播更新。

检查: 训练前后 embedding 权重是否发生变化
检查: 梯度是否能流到 embedding 参数
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

STAGE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(STAGE_SRC))

from model import build_model
from utils import OUTPUT_DIR, load_config, set_seed

AUDIT_DIR = Path(__file__).resolve().parent.parent
CONFIG = load_config(str(Path(__file__).resolve().parents[2] / "configs" / "pyrite_token_transformer.yaml"))


def load_trained_model(config: dict):
    """Load checkpoint and return model + metadata."""
    checkpoint = torch.load(
        OUTPUT_DIR / "checkpoints" / "token_transformer_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    elements = list(checkpoint["elements"])
    deposit_classes = list(checkpoint["deposit_classes"])
    model = build_model(config, len(elements), len(deposit_classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, elements, deposit_classes


def capture_weights(model: nn.Module) -> dict[str, np.ndarray]:
    """Capture all parameter values as numpy arrays."""
    return {name: param.detach().cpu().numpy().copy() for name, param in model.named_parameters()}


def compare_weights(initial: dict, trained: dict) -> pd.DataFrame:
    """Compare initial and trained weights, return summary dataframe."""
    rows = []
    for name in sorted(initial.keys()):
        if name not in trained:
            rows.append({"param": name, "status": "MISSING_IN_TRAINED"})
            continue
        diff = initial[name] - trained[name]
        max_abs_diff = float(np.abs(diff).max())
        mean_abs_diff = float(np.abs(diff).mean())
        rel_change = float(np.abs(diff).mean() / (np.abs(trained[name]).mean() + 1e-10))
        rows.append(
            {
                "param": name,
                "initial_norm": float(np.linalg.norm(initial[name])),
                "trained_norm": float(np.linalg.norm(trained[name])),
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "rel_change": rel_change,
                "changed": bool(max_abs_diff > 1e-8),
            }
        )
    return pd.DataFrame(rows)


def verify_gradient_flow(config: dict) -> bool:
    """Run one forward+backward step and check if embeddings receive gradients."""
    set_seed(int(config.get("random_seed", 42)))
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    elements = data["elements"].astype(str).tolist()
    deposit_classes = data["deposit_classes"].astype(str).tolist()

    model = build_model(config, len(elements), len(deposit_classes))
    model.train()

    x = torch.tensor(data["element_values"][:4], dtype=torch.float32)
    visible = torch.tensor(data["visible_mask"][:4], dtype=torch.float32)
    y = torch.tensor(data["label_id"][:4], dtype=torch.long)

    # Simulate masked batch manually
    mask_idx = model.mask_deposit_index
    deposit_ids = y.clone()
    deposit_ids[0] = mask_idx
    deposit_ids[1] = mask_idx
    label_visible = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    out = model(x, visible, deposit_ids, label_visible.reshape(-1))
    element_mask = visible > 0.5
    elem_loss = nn.HuberLoss()(out["element_pred"][element_mask], x[element_mask])
    type_loss = nn.CrossEntropyLoss()(out["deposit_logits"][:2], y[:2])
    loss = elem_loss + type_loss
    loss.backward()

    grad_checks = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = float(param.grad.norm().item())
            grad_checks[name] = grad_norm > 1e-8
        else:
            grad_checks[name] = False

    return grad_checks


def main() -> None:
    print("=" * 60)
    print("Q1: Embedding 初始化 & 反向传播验证")
    print("=" * 60)

    # Check 1: Weight change before/after training
    print("\n[1a] 训练前后权重变化检测...")
    set_seed(int(CONFIG.get("random_seed", 42)))
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    elements = data["elements"].astype(str).tolist()
    deposit_classes = data["deposit_classes"].astype(str).tolist()

    # Fresh model (before training)
    fresh_model = build_model(CONFIG, len(elements), len(deposit_classes))
    initial_weights = capture_weights(fresh_model)

    # Trained model
    trained_model, trained_elements, trained_deposits = load_trained_model(CONFIG)
    trained_weights = capture_weights(trained_model)

    comparison = compare_weights(initial_weights, trained_weights)
    embedding_params = comparison[comparison["param"].str.contains("embedding")]
    print("\nElement/Deposit Embedding 权重变化:")
    print(embedding_params.to_string(index=False))

    emb_changed = embedding_params["changed"].all()
    print(f"\n✓ 结论: 所有 embedding 参数{'已' if emb_changed else '未'}通过训练更新")

    # Check 2: Gradient flow
    print("\n[1b] 梯度传播验证...")
    grad_status = verify_gradient_flow(CONFIG)
    emb_grads = {k: v for k, v in grad_status.items() if "embedding" in k}
    print("\nEmbedding 梯度状态:")
    for name, has_grad in emb_grads.items():
        status = "✓ GRADIENT FLOWING" if has_grad else "✗ NO GRADIENT"
        print(f"  {name}: {status}")

    all_grad = all(emb_grads.values())
    print(f"\n✓ 结论: Embedding {'能' if all_grad else '不能'}接收到梯度信号")

    # Save results
    comparison.to_csv(AUDIT_DIR / "outputs" / "q1_weight_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"param": k, "has_gradient": v} for k, v in grad_status.items()]).to_csv(
        AUDIT_DIR / "outputs" / "q1_gradient_status.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n输出已保存至 {AUDIT_DIR / 'outputs'}")


if __name__ == "__main__":
    main()
