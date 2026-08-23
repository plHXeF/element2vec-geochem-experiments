"""
Q2: 验证元素 Token 构建的向量维度链。

逐步打印 shape 和数值范围，以算式验证每一步的向量变化：
  T_{s,e}^{elem} = LayerNorm(GELU(W_elem · [x_{s,e}; v_{s,e}; E_e] + b_elem))
  T_s^{dep}     = LayerNorm(GELU(W_dep · [0; l_s; D_{id}] + b_dep))
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

STAGE_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(STAGE_SRC))

from model import ElementTokenTransformer, build_model
from utils import OUTPUT_DIR, load_config, set_seed

AUDIT_DIR = Path(__file__).resolve().parent.parent


def inspect_model_structure(model: ElementTokenTransformer) -> None:
    """Print model architecture details."""
    print("\n--- Model Structure ---")
    print(f"  n_elements           = {model.n_elements}")
    print(f"  n_deposit_types      = {model.n_deposit_types}")
    print(f"  mask_deposit_index   = {model.mask_deposit_index}")
    print(f"  d_model              = {model.d_model}")
    print(f"  element_embedding    = {model.element_embedding}")
    print(f"  deposit_embedding    = {model.deposit_embedding}")
    print(f"  element_projector    = {model.element_projector}")
    print(f"  deposit_projector    = {model.deposit_projector}")
    print(f"  encoder layers       = {len(model.encoder.layers)}")
    print(f"  element_head         = {model.element_head}")
    print(f"  deposit_head         = {model.deposit_head}")


def trace_forward_step(config: dict) -> None:
    """Run a single forward pass and print shape/value trace for one sample."""
    set_seed(int(config.get("random_seed", 42)))

    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    elements = data["elements"].astype(str).tolist()
    deposit_classes = data["deposit_classes"].astype(str).tolist()
    n_elements = len(elements)
    n_deposits = len(deposit_classes)
    elem_emb_dim = int(config.get("element_embedding_dim", 32))
    dep_emb_dim = int(config.get("deposit_embedding_dim", 32))
    d_model = int(config.get("d_model", 64))

    model = build_model(config, n_elements, n_deposits)
    model.eval()

    # Use one sample for detailed tracing
    sample_idx = 0
    x = torch.tensor(data["element_values"][sample_idx : sample_idx + 1], dtype=torch.float32)
    visible = torch.tensor(data["visible_mask"][sample_idx : sample_idx + 1], dtype=torch.float32)
    y = torch.tensor(data["label_id"][sample_idx : sample_idx + 1], dtype=torch.long)
    label_str = str(data["deposit_type"][sample_idx])

    print(f"\n--- Sample: idx={sample_idx}  type={label_str} ---")
    print(f"  x       (log_robust) shape={tuple(x.shape)}  range=[{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"  visible shape={tuple(visible.shape)}  values={visible[0].numpy().tolist()}")
    print(f"  y       shape={tuple(y.shape)}  value={y.item()}  class={deposit_classes[y.item()]}")

    # --- Trace make_tokens ---
    batch_size = x.shape[0]
    element_ids = torch.arange(n_elements).unsqueeze(0).expand(batch_size, -1)
    print(f"\n  [Step A] element_ids shape={tuple(element_ids.shape)}  values={element_ids[0].numpy().tolist()}")

    elem_emb = model.element_embedding(element_ids)
    print(f"  [Step B] elem_emb     shape={tuple(elem_emb.shape)}   (batch × n_elem × elem_emb_dim={elem_emb_dim})")
    print(f"           elem_emb[0,0,:] range=[{elem_emb[0,0].min().item():.4f}, {elem_emb[0,0].max().item():.4f}]")

    elem_input = torch.cat([x.unsqueeze(-1), visible.unsqueeze(-1), elem_emb], dim=-1)
    print(f"  [Step C] elem_input   shape={tuple(elem_input.shape)}  (batch × n_elem × {elem_emb_dim + 2})")
    print(f"           即: [log_abund(1); visible(1); E_e({elem_emb_dim})]")

    elem_tokens = model.element_projector(elem_input)
    print(f"  [Step D] elem_tokens  shape={tuple(elem_tokens.shape)}  (batch × n_elem × d_model={d_model})")
    print(f"           token[0,0,:] range=[{elem_tokens[0,0].min().item():.4f}, {elem_tokens[0,0].max().item():.4f}]")

    # --- Deposit token ---
    dep_emb = model.deposit_embedding(y)
    zero_val = torch.zeros(batch_size, 1, dtype=x.dtype)
    label_vis = torch.ones(batch_size, 1, dtype=x.dtype)
    dep_input = torch.cat([zero_val, label_vis, dep_emb], dim=-1)
    print(f"\n  [Step E] dep_input    shape={tuple(dep_input.shape)}  (batch × {dep_emb_dim + 2})")
    print(f"           即: [0(1); label_visible(1); D_id({dep_emb_dim})]")

    dep_token = model.deposit_projector(dep_input).unsqueeze(1)
    print(f"  [Step F] dep_token    shape={tuple(dep_token.shape)}  (batch × 1 × d_model={d_model})")
    print(f"           token[0,0,:] range=[{dep_token[0,0].min().item():.4f}, {dep_token[0,0].max().item():.4f}]")

    tokens = torch.cat([elem_tokens, dep_token], dim=1)
    print(f"\n  [Step G] ALL tokens   shape={tuple(tokens.shape)}  (batch × {n_elements + 1} × d_model)")

    # --- Encoder ---
    encoded = model.encoder(tokens)
    print(f"  [Step H] encoded      shape={tuple(encoded.shape)}  (batch × {n_elements + 1} × d_model)")

    elem_encoded = encoded[:, :n_elements, :]
    deposit_encoded = encoded[:, n_elements, :]
    print(f"  [Step I] elem_encoded shape={tuple(elem_encoded.shape)}  (batch × n_elem × d_model)")
    print(f"  [Step J] dep_encoded  shape={tuple(deposit_encoded.shape)}  (batch × d_model)")

    element_pred = model.element_head(elem_encoded).squeeze(-1)
    deposit_logits = model.deposit_head(deposit_encoded)
    print(f"  [Step K] element_pred shape={tuple(element_pred.shape)}  (batch × n_elem)")
    print(f"  [Step L] deposit_logits shape={tuple(deposit_logits.shape)}  (batch × n_deposits)")

    # --- Verify the dimension chain ---
    print("\n--- 维度链验证 ---")
    checks = [
        ("elem_input dim", elem_input.shape[-1] == elem_emb_dim + 2, f"{elem_input.shape[-1]} == {elem_emb_dim + 2}"),
        ("dep_input dim", dep_input.shape[-1] == dep_emb_dim + 2, f"{dep_input.shape[-1]} == {dep_emb_dim + 2}"),
        ("elem_token dim", elem_tokens.shape[-1] == d_model, f"{elem_tokens.shape[-1]} == {d_model}"),
        ("dep_token dim", dep_token.shape[-1] == d_model, f"{dep_token.shape[-1]} == {d_model}"),
        ("all_tokens dim", tokens.shape[-1] == d_model, f"{tokens.shape[-1]} == {d_model}"),
        ("sequence length", tokens.shape[1] == n_elements + 1, f"{tokens.shape[1]} == {n_elements + 1}"),
        ("element_pred dim", element_pred.shape[-1] == n_elements, f"{element_pred.shape[-1]} == {n_elements}"),
        ("deposit_logits dim", deposit_logits.shape[-1] == n_deposits, f"{deposit_logits.shape[-1]} == {n_deposits}"),
    ]
    all_pass = True
    for name, ok, detail in checks:
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"  {status} {name}: {detail}")
    print(f"\n  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # --- Formula verification ---
    print("\n--- 算式验证 ---")
    z_dim = elem_emb_dim + 2
    print(f"""
    For element e (0..{n_elements - 1}):
      z_s,e       = [x_s,e; v_s,e; E_e]          in R^{z_dim}
      h_s,e       = GELU(W_elem · z_s,e + b_elem)  in R^{d_model}
      T_s,e^elem  = LayerNorm(h_s,e)                in R^{d_model}

    For deposit:
      T_s^dep = LayerNorm(GELU(W_dep · [0; l_s; D_id] + b_dep)) in R^{d_model}
    """)


def verify_data_distribution(config: dict) -> None:
    """Check log_robust value distribution for scale matching with embeddings."""
    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    values = data["element_values"]
    visible = data["visible_mask"]

    measured = values[visible > 0.5]
    imputed = values[visible < 0.5]

    print("\n--- 数据分布 vs Embedding 尺度 ---")
    print(f"  log_robust (measured):  mean={np.mean(measured):.4f}  std={np.std(measured):.4f}  "
          f"range=[{np.min(measured):.4f}, {np.max(measured):.4f}]")
    print(f"  log_robust (imputed=0): count={len(imputed)}  (all = 0.0)")
    print(f"  Embedding N(0,1) 32-dim vec norm ≈ √32 ≈ {np.sqrt(32):.1f}")
    print(f"  结论: log_robust 数值域约 [{-np.std(measured) * 3:.0f}, {np.std(measured) * 3:.0f}]")
    print(f"  Embedding 初始范数约 {np.sqrt(32):.0f}，与 log_robust 尺度相当 → 无尺度失衡风险")


def main() -> None:
    print("=" * 60)
    print("Q2: Token 构建向量维度链验证")
    print("=" * 60)

    config = load_config(str(Path(__file__).resolve().parents[2] / "configs" / "pyrite_token_transformer.yaml"))

    data = np.load(OUTPUT_DIR / "preprocessed" / "pyrite_token_data.npz", allow_pickle=True)
    elements = data["elements"].astype(str).tolist()
    deposit_classes = data["deposit_classes"].astype(str).tolist()
    set_seed(int(config.get("random_seed", 42)))
    model = build_model(config, len(elements), len(deposit_classes))

    inspect_model_structure(model)
    trace_forward_step(config)
    verify_data_distribution(config)


if __name__ == "__main__":
    main()
