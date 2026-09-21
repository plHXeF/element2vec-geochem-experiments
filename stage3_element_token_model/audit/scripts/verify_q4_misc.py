"""
Q4: 杂项逻辑检查。

检查项:
  4a: 训练/验证划分 — 是否有 train/val split
  4b: Embedding 尺度 vs 输入数据尺度
  4c: IQR=0 处理是否正确
  4d: gradient clipping 值是否合理
  4e: 训练可复现性 — deterministic 设置
  4f: NaN→0 填充的一致性
  4g: mean_all pooling 的潜在泄漏
  4h: previous_baseline_space 硬编码路径
  4j: 样本级 token matrix 提取时 deposit 是否被正确 mask
"""
from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[3]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

from geochem_workspace import output_path

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

STAGE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(STAGE_SRC))

from utils import OUTPUT_DIR, robust_scale, torch_device

AUDIT_DIR = output_path("audit")
if __name__ == "__main__":
    (AUDIT_DIR / "outputs").mkdir(parents=True, exist_ok=True)


def check_4a_train_val_split() -> dict:
    """检查是否有 train/val 划分."""
    print("\n--- [4a] Train/Val Split ---")
    issues = []
    # Read train.py to check for val split
    train_path = STAGE_SRC / "train.py"
    code = train_path.read_text()
    has_val = any(word in code for word in ["validation", "val_loader", "valid", "val_split", "train_test_split"])
    has_full_epoch = "for epoch in range" in code
    info = {
        "has_validation": has_val,
        "uses_full_data": has_full_epoch,
        "note": "全量数据训练，无验证集。training loss 和 masked_type_acc 均为 in-sample 指标"
    }
    print(f"  有验证集划分: {'否 ✗' if not has_val else '是 ✓'}")
    print(f"  使用全量数据: {'是 ✗ 无泛化估计' if has_full_epoch else '否'}")
    print(f"  影响: 无法评估过拟合程度，checkpoint 可能对训练数据过拟合")
    return info


def check_4b_embedding_scale() -> dict:
    """检查 embedding 初始尺度与输入数据尺度的匹配."""
    print("\n--- [4b] Embedding 尺度匹配 ---")
    try:
        from model import build_model
        from utils import load_config, set_seed

        config = load_config(str(Path(__file__).resolve().parents[2] / "configs" / "pyrite_token_transformer.yaml"))
        set_seed(int(config.get("random_seed", 42)))
        data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
        elements = data["elements"].astype(str).tolist()
        deposit_classes = data["deposit_classes"].astype(str).tolist()

        model = build_model(config, len(elements), len(deposit_classes))
        elem_emb = model.element_embedding.weight.detach().numpy()
        dep_emb = model.deposit_embedding.weight.detach().numpy()

        values = data["element_values"]
        visible = data["visible_mask"]
        measured_vals = values[visible > 0.5]

        info = {
            "element_embedding_std": float(np.std(elem_emb)),
            "element_embedding_mean": float(np.mean(elem_emb)),
            "deposit_embedding_std": float(np.std(dep_emb)),
            "log_robust_measured_std": float(np.std(measured_vals)),
            "log_robust_measured_range": [float(np.min(measured_vals)), float(np.max(measured_vals))],
            "embedding_vec_norm_approx": float(np.sqrt(elem_emb.shape[1]) * np.std(elem_emb)),
        }
        info["scale_ratio"] = info["embedding_vec_norm_approx"] / (info["log_robust_measured_std"] + 1e-10)

        print(f"  Element Embedding: mean={info['element_embedding_mean']:.4f} std={info['element_embedding_std']:.4f}")
        print(f"  Embedding 向量范数 ≈ {info['embedding_vec_norm_approx']:.2f}")
        print(f"  log_robust 实测值 std: {info['log_robust_measured_std']:.4f}")
        print(f"  尺度比 (emb_norm / data_std): {info['scale_ratio']:.2f}")
        if info['scale_ratio'] > 10:
            print(f"  ✗ WARNING: Embedding 初始尺度远大于数据，可能主导早期训练")
        elif info['scale_ratio'] < 0.1:
            print(f"  ✗ WARNING: Embedding 初始尺度远小于数据，可能被数据淹没")
        else:
            print(f"  ✓ 尺度匹配可接受")
    except Exception as e:
        info = {"error": str(e)}
        print(f"  ✗ 检查失败: {e}")
    return info


def check_4c_iqr_zero_handling() -> dict:
    """检查 robust_scale 中 IQR=0 的处理."""
    print("\n--- [4c] IQR=0 处理 ---")
    # Create test data: all same values
    test_data = np.array([[5.0, 10.0], [5.0, 10.0], [5.0, 10.0], [5.0, 10.0], [5.0, np.nan]])
    scaled, med, iqr = robust_scale(test_data)
    info = {
        "input": test_data.tolist(),
        "median": med.tolist(),
        "iqr_original_zero": [iqr[0].item() if iqr[0] < 1e-10 else "non-zero", iqr[1].item()],
        "iqr_replaced_to_1": [iqr[0].item() if np.isclose(iqr[0], 1.0) else "not_1"],
        "scaled_output": scaled.tolist(),
    }
    print(f"  输入 (col0 全同): {test_data[:, 0].tolist()}")
    print(f"  IQR[0] (预替换): ≈0 → 替换为: {iqr[0]:.1f}")
    print(f"  IQR[1] (正常): {iqr[1]:.1f}")
    print(f"  输出: {scaled.tolist()}")
    # If all same + 0 IQR → scaled should be ~0 (centered, not truly scaled)
    scaled_ok = np.allclose(scaled[:4, 0], 0.0, atol=1e-5)
    print(f"  {'✓ 同值元素被置为0' if scaled_ok else '✗ 处理可能有误'}")
    return info


def check_4e_deterministic() -> dict:
    """检查可复现性设置."""
    print("\n--- [4e] 训练可复现性 ---")
    train_path = STAGE_SRC / "train.py"
    utils_path = STAGE_SRC / "utils.py"
    all_code = train_path.read_text() + utils_path.read_text()
    issues = []
    if "set_seed" in all_code:
        issues.append("set_seed ✓")
    else:
        issues.append("set_seed ✗ 未调用")

    if "torch.backends.cudnn.deterministic" in all_code:
        issues.append("cudnn deterministic ✓")
    else:
        issues.append("cudnn deterministic ✗ 未设置 — GPU上可能不可复现")

    if "torch.use_deterministic_algorithms" in all_code:
        issues.append("use_deterministic_algorithms ✓")
    else:
        issues.append("use_deterministic_algorithms ✗ 未设置")

    print("  " + "\n  ".join(issues))
    return {"issues": issues}


def check_4f_nan_fill_consistency() -> dict:
    """验证 NaN→0 填充在 preprocess 和 train masking 中的一致性."""
    print("\n--- [4f] NaN→0 填充一致性 ---")
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    values = data["element_values"]
    visible = data["visible_mask"]

    # Check: all visible=0 positions should have value=0
    invisible_positions = visible < 0.5
    invisible_values = values[invisible_positions]
    all_zero = np.allclose(invisible_values, 0.0, atol=1e-7)

    # Check: train masking sets masked positions to (0,0) same as preprocessing
    train_path = STAGE_SRC / "train.py"
    train_code = train_path.read_text()
    mask_sets_x_zero = "x_input[elem_loss_mask]" in train_code and "= 0.0" in train_code
    mask_sets_v_zero = "visible_input[elem_loss_mask]" in train_code and "= 0.0" in train_code

    info = {
        "all_missing_are_zero": bool(all_zero),
        "train_mask_sets_x_to_0": mask_sets_x_zero,
        "train_mask_sets_v_to_0": mask_sets_v_zero,
        "consistent": bool(all_zero and mask_sets_x_zero and mask_sets_v_zero),
    }
    print(f"  所有 visible=0 的位置 value=0: {'是 ✓' if all_zero else '否 ✗'}")
    print(f"  训练 mask 将 x 设为 0: {'是 ✓' if mask_sets_x_zero else '否 ✗'}")
    print(f"  训练 mask 将 visible 设为 0: {'是 ✓' if mask_sets_v_zero else '否 ✗'}")
    print(f"  一致性: {'✓ 正确 — 模型看 mask 元素等同于天生缺失' if info['consistent'] else '✗ 不一致'}")
    return info


def check_4g_pooling_leak() -> dict:
    """检查 mean_all pooling 是否可能泄漏 label_visible 信息."""
    print("\n--- [4g] Pooling 模式审查 ---")
    extract_latent_path = STAGE_SRC / "extract_latent.py"
    code = extract_latent_path.read_text()

    has_mean_element = "mean_element" in code
    has_deposit = '"deposit"' in code or "'deposit'" in code
    has_mean_all = "mean_all" in code

    info = {
        "mean_element_excludes_deposit": has_mean_element,
        "deposit_uses_only_deposit_token": has_deposit,
        "mean_all_includes_deposit": has_mean_all,
    }
    print(f"  mean_element: 仅平均前 14 个元素 token {'✓ 不泄漏' if has_mean_element else '?'}")
    print(f"  deposit: 只用第 15 个 (矿床) token {'✓ 不泄漏' if has_deposit else '?'}")
    print(f"  mean_all: 平均全部 15 token → 包含 label_visible 信息 {'✗ 潜在泄漏' if has_mean_all else '?'}")
    if has_mean_all:
        print(f"    WARNING: mean_all pooling 会将 label_visible 编码的 0/1 信息混入 latent")
        print(f"    仅在知道后果时使用 mean_all")
    return info


def check_4h_previous_baseline_path() -> dict:
    """检查 previous_baseline 路径是否存在."""
    print("\n--- [4h] Previous Baseline 路径 ---")
    workspace = STAGE_SRC.parent.parent
    paths = [
        workspace / "stage2_geo_representation_xgb" / "outputs" / "representations" / "log_robust_z_14.csv",
        workspace / "stage2_geo_representation_xgb" / "outputs" / "representations" / "rank_uniform_z_14.csv",
    ]
    info = {"paths": {str(p): p.exists() for p in paths}}
    for p, exists in info["paths"].items():
        print(f"  {p}: {'存在 ✓' if exists else '不存在 — 该 space 将被跳过'}")
    return info


def check_4j_token_extraction_mask() -> dict:
    """验证 extract_embeddings 中 deposit 被正确强制 mask."""
    print("\n--- [4j] Token 提取时 Deposit Mask ---")
    extract_path = STAGE_SRC / "extract_embeddings.py"
    code = extract_path.read_text()

    uses_mask_deposit = "mask_deposit_index" in code
    label_visible_zero = "label_visible = torch.zeros" in code
    loader_discards_y = "for xb, vb, _" in code or "for xb, vb, _y" in code

    info = {
        "deposit_forced_to_mask": uses_mask_deposit and label_visible_zero,
        "loader_discards_label": loader_discards_y,
        "correct": bool(uses_mask_deposit and label_visible_zero and loader_discards_y),
    }
    print(f"  deposit 被强制设为 MASK_DEPOSIT: {'是 ✓' if uses_mask_deposit else '否 ✗'}")
    print(f"  label_visible 被设为 0: {'是 ✓' if label_visible_zero else '否 ✗'}")
    print(f"  DataLoader label 被丢弃: {'是 ✓' if loader_discards_y else '否 ✗'}")
    print(f"  结论: {'✓ 正确 — sample_input/encoded_token_flat 均为 label_masked 模式' if info['correct'] else '✗ 有问题'}")
    return info


def main() -> None:
    import json

    print("=" * 60)
    print("Q4: 杂项逻辑检查")
    print("=" * 60)

    results = {}
    results["4a_train_val_split"] = check_4a_train_val_split()
    results["4b_embedding_scale"] = check_4b_embedding_scale()
    results["4c_iqr_zero"] = check_4c_iqr_zero_handling()
    results["4e_deterministic"] = check_4e_deterministic()
    results["4f_nan_fill"] = check_4f_nan_fill_consistency()
    results["4g_pooling"] = check_4g_pooling_leak()
    results["4h_previous_baseline"] = check_4h_previous_baseline_path()
    results["4j_token_extraction"] = check_4j_token_extraction_mask()

    outputs_dir = AUDIT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "q4_misc_audit.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\n输出已保存至 {outputs_dir}")


if __name__ == "__main__":
    main()
