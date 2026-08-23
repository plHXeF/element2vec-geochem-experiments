# Stage 3 代码逻辑审核报告

**审核日期**: 2026-06-29
**审核范围**: `stage3_element_token_model/` 全部 10 个源文件
**审核脚本**: `audit/scripts/verify_q1~q4_*.py`

---

## 总览

| 审核项 | 结论 | 严重度 |
|--------|------|--------|
| Q1: Embedding 初始化 & 反向传播 | ✅ 通过 — 初始化正确，梯度流通 | — |
| Q2: Token 构建向量维度链 | ✅ 通过 — 所有维度匹配，算式正确 | — |
| Q3: Mask 机制 | ✅ 通过 — 元素/矿床 mask 逻辑正确，无泄漏 | — |
| Q4a: Train/Val 划分 | ⚠️ 无验证集，全量训练 | 中 |
| Q4b: Embedding 尺度 | ✅ 通过 — 嵌入范数与数据量级相当 | — |
| Q4c: IQR=0 处理 | ✅ 通过 — 零方差异常处理正确 | — |
| Q4e: 训练可复现性 | ⚠️ 缺少 cudnn deterministic | 低 |
| Q4f: NaN→0 填充 | ✅ 通过 — preprocess 与 train mask 行为一致 | — |
| Q4g: mean_all pooling | ⚠️ 包含 label_visible 信息 | 低 |
| Q4j: Token 提取 mask | ✅ 通过 — label_masked 模式正确 | — |

**核心代码逻辑总体正确，无严重 bug。** 主要问题集中在训练协议层面（缺少验证集）和可复现性设置。

---

## 1. Q1: Embedding 初始化与反向传播

### 1.1 初始化

```python
# model.py:26-27
self.element_embedding = nn.Embedding(n_elements, element_embedding_dim)    # 14×32
self.deposit_embedding  = nn.Embedding(n_deposit_types + 1, deposit_embedding_dim) # 7×32
```

PyTorch `nn.Embedding` 默认初始化 `N(0, 1)`，无显式 Xavier/Glorot 初始化。

### 1.2 训练后权重变化（实证验证）

| 参数 | 初始范数 | 训练后范数 | 最大绝对差异 | 相对变化 |
|------|----------|-----------|-------------|---------|
| `element_embedding.weight` | 20.41 | 17.97 | 0.562 | 19.1% |
| `deposit_embedding.weight` | 15.02 | 15.03 | 0.611 | 20.4% |

**结论**: ✅ 两个 embedding 表都经训练显著变化 — 反向传播成功生效。

### 1.3 梯度传播验证

在一次前向+反向传播后检查 `param.grad`:
- `element_embedding.weight.grad` ≠ 0 ✓
- `deposit_embedding.weight.grad` ≠ 0 ✓

梯度完整流经 `index_lookup → concat → Linear → GELU → LayerNorm → TransformerEncoder → prediction_head → loss`。

---

## 2. Q2: Token 构建 — 向量变换算式

### 2.1 元素 Token 构建

给定样本 s、元素 e ∈ {0..13}:

```
输入:
  x_{s,e}      ∈ R       log10 丰度，robust标准化（缺失→0）
  v_{s,e}      ∈ {0,1}   visible mask
  E_e          ∈ R^{32}  element_embedding.weight[e]  (跨样本常量)

Step 1 — 拼接:
  concat_{s,e} = [x_{s,e}; v_{s,e}; E_e]  ∈ R^{34}

Step 2 — Linear + GELU (element_projector[0:2]):
  h_{s,e} = GELU(W_elem · concat_{s,e} + b_elem)  ∈ R^{64}
  where W_elem ∈ R^{64×34}, b_elem ∈ R^{64}

Step 3 — LayerNorm (element_projector[2]):
  T_{s,e}^{elem} = γ_elem ⊙ (h_{s,e} − μ_elem) / σ_elem + β_elem  ∈ R^{64}
```

### 2.2 矿床 Token 构建

```
  l_s          ∈ {0,1}   label_visible (0=掩蔽, 1=可见)
  D_{id}       ∈ R^{32}  deposit_embedding.weight[id]

  T_s^{dep} = LayerNorm(GELU(W_dep · [0; l_s; D_{id}] + b_dep))  ∈ R^{64}
```

**注**: 矿床 token 始终在"丰度"槽位填充常量 0，与元素的 `x_{s,e}` 位置对应。

### 2.3 完整 Token 序列

```
T_s = [T_{s,0}^{elem}, ..., T_{s,13}^{elem}, T_s^{dep}]  ∈ R^{15×64}
```

### 2.4 维度验证结果

所有 8 项维度检查全部通过：

```
✓ elem_input dim:     34 == 32+2
✓ dep_input dim:      34 == 32+2
✓ elem_token dim:     64 == d_model
✓ dep_token dim:      64 == d_model
✓ all_tokens dim:     64 == d_model
✓ sequence length:    15 == 14+1
✓ element_pred dim:   14 == n_elements
✓ deposit_logits dim: 6 == n_deposit_types
```

### 2.5 数据尺度匹配

| 指标 | 值 |
|------|-----|
| log_robust 实测值 std | 0.774 |
| Embedding 32 维向量范数 ≈ | 5.45 |
| 尺度比 (emb_norm / data_std) | 7.0 |

尺度比 7:1 在可接受范围内，不会导致 embedding 主导或淹没数据信号。

---

## 3. Q3: Mask 机制

### 3.1 元素 Mask

```
elem_random = rand(14) < 0.15
elem_loss_mask = elem_random & (visible > 0.5)  ← 只能mask有测值的元素

x_input[elem_loss_mask] = 0.0          ← 值归零
visible_input[elem_loss_mask] = 0.0     ← 可见性归零
```

**验证结果**: 5000 样本审计中，167 个被 mask 的元素位置 **无一** 是 visible=0 的元素 — 机制正确。

### 3.2 矿床 Mask

```
deposit_mask = (rand() < 0.5) | joint_mask      ← 50%随机 + joint强制
deposit_ids[deposit_mask] = mask_deposit_index   ← 替换为[MASK_DEPOSIT]
label_visible = (~deposit_mask).float()           ← 0=masked, 1=visible
```

有效矿床 mask 概率: P(deposit) + P(joint) − P(both) = 0.50 + 0.20 − 0.10 = **0.60**（60% 样本的矿床被 mask）。

### 3.3 Joint Mask 重采样（⚠️ 误报修正）

代码位置: `train.py:33-34`

```python
if joint_mask.any():
    elem_loss_mask[joint_mask] = (torch.rand_like(...) < element_mask_ratio) & (visible > 0.5)
```

**之前分析曾将此处标记为潜在的泄漏 bug**（认为重采样发生在 x_input 清零之后）。验证后确认：**重采样 (L33-34) 发生在 x_input 清零 (L38-39) 之前**，因此不存在 mask 与输入的不一致。

该重采样的实际效果：对 joint_mask=True 的 20% 样本，用新随机 mask 替换原始 mask。由于两次采样独立同分布，统计效果等价于直接用原始 mask。**该行代码是冗余的（不产生 bug，但不必要）。**

| 指标 | 值 |
|------|-----|
| 每样本平均测值元素数 | 11.2 / 14 |
| 有效元素 mask 率 | 12.0%（约占全部 14 元素） |
| 实际预期 mask 元素数/样本 | ~1.67 个 |

### 3.4 损失计算

```
元素损失: Huber(element_pred[elem_loss_mask], original_x[elem_loss_mask])
矿床损失: CrossEntropy(deposit_logits[deposit_mask], y[deposit_mask])
```

**仅在被 mask 的位置计算损失** — 符合 BERT masked modeling 范式。未 mask 的位置不产生梯度信号。

---

## 4. Q4: 其他发现

### 4.1 ⚠️ 无 Train/Val 划分（中等严重度）

`train.py:91` 循环 `for epoch in range(1, epochs + 1)` 对全量 7778 个样本训练。Training loss 和 masked_type_accuracy 均为 in-sample 指标。

- **影响**: 无法感知过拟合程度。1000 epoch 训练可能严重过拟合，但脚本无任何机制检测。
- **上下文**: 项目定位为 "exploratory sanity check" 而非严格 ML benchmark — 可接受但需明确认知。
- **建议**: 至少预留 10% 数据做 held-out loss 监控，或在文档中显式声明全量训练的后果。

### 4.2 ✅ Embedding 初始尺度可接受

Embedding 向量范数 ≈ 5.45 vs 数据 std = 0.77, 比率 7:1。虽略偏高，但在 GELU + LayerNorm 后不会主导训练。

### 4.3 ✅ IQR=0 异常处理正确

`utils.py:157`: 当某元素全部样本取值相同（IQR=0），自动替换为 1.0。验证确认：同值元素被正确地 scale 为 0（仅居中，不缩放）。

### 4.4 Gradient Clipping 值为 1.0

`train.py:126`: `clip_grad_norm_(model.parameters(), 1.0)` — 对 embedding + Transformer + MLP 组合，1.0 的梯度范数上限较为激进。未发现训练曲线异常，但若出现 plateau 可考虑放宽至 5.0。

### 4.5 ⚠️ 训练不可完全复现（低严重度）

| 设置 | 状态 |
|------|------|
| `set_seed()` | ✓ 已设置 |
| `torch.backends.cudnn.deterministic` | ✗ 未设置 |
| `torch.use_deterministic_algorithms()` | ✗ 未设置 |

GPU 上同一 seed 可能产生不同 checkpoint。对研究可复现性有影响。建议在 `train.py` 或 `utils.py` 中添加确定性设置。

### 4.6 ✅ NaN→0 填充一致性

- Preprocess: `robust_scale()` 将 NaN→0（`utils.py:159`）
- Train mask: `x_input[elem_loss_mask] = 0.0`（`train.py:38`）

两者行为完全一致：(value=0, visible=0) 即代表元素不可用。模型无法也无需区分"天生缺失"和"训练中被 mask"，这是正确的 masked modeling 设计。

### 4.7 ⚠️ `mean_all` Pooling 潜在泄漏（低严重度）

`extract_latent.py:30-31`:
```python
if pooling in {"mean_all", "all"}:
    return encoded.mean(dim=1)
```

`mean_all` 对全部 15 个 token 求平均，包含 deposit token。若 `label_visible=1`，该信息会通过 deposit token 混入 latent。`mean_element`（仅平均前 14 元素 token）无此问题。主结果使用 `mean_element`，正确。

**建议**: 将 `mean_all` 标记为 "仅供分析，不作为主结果"。

### 4.8 ✅ Previous Baseline 路径可用

Stage 2 输出文件均存在于预期路径，`build_spaces.py` 可正常创建 `previous_baseline_space.csv`。

### 4.9 ✅ Token 提取时 Deposit 被正确强制 Mask

`extract_embeddings.py:57-58`:
```python
dep_ids = torch.full((xb.shape[0],), model.mask_deposit_index, ...)
label_visible = torch.zeros(xb.shape[0], ...)
```

`sample_input_token_flat.csv` 和 `sample_encoded_token_flat.csv` 均在 label_masked 模式下生成 — 无目标泄漏。

---

## 5. 修复建议

### 必须修复

_无。核心逻辑无 bug。_

### 建议修复

| 序号 | 文件 | 行号 | 修改内容 | 优先度 |
|------|------|------|----------|--------|
| 1 | `train.py` | 33-34 | 删除 joint_mask 分支内的 `elem_loss_mask` 重赋值（冗余代码，容易误导阅读者） | 低 |
| 2 | `utils.py` | 128 | 添加 `torch.backends.cudnn.deterministic = True` | 中 |
| 3 | `train.py` | — | 可选：添加 held-out set 用于 training curve 监控（不参与梯度更新） | 中 |
| 4 | `model.py` | 26 | 可选：`nn.Embedding` 改用 `xavier_uniform` 初始化，或添加 `nn.init.xavier_uniform_` | 低 |

### 修复 1 的具体做法

`train.py:33-34` 当前代码：
```python
if joint_mask.any():
    elem_loss_mask[joint_mask] = (torch.rand_like(x[joint_mask]) < element_mask_ratio) & (visible[joint_mask] > 0.5)
```

该 if 块及内部两行可以**整体删除**。因为 MASK_1 和 MASK_2 是独立同分布随机采样，替换是统计上等价的。删除后 joint_mask 仅影响 deposit_mask 的计算：
```python
deposit_mask = (torch.rand(x.shape[0]) < deposit_mask_prob) | joint_mask
```
这已经实现了"joint_mask 样本的矿床一定被 mask"的语义。

### 修复 2 的具体做法

`utils.py:set_seed` 在 `torch.manual_seed(seed)` 后添加：
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 6. 附录：代码逻辑总结

### 数据流

```
pyrite_benchmark/data/processed/*.csv
  → preprocess.py → pyrite_token_data.npz (log_robust + visible + label_id)
    → train.py → token_transformer_last.pt
    → extract_embeddings.py → element_embedding.csv, sample_input/encoded_token_flat.csv
    → extract_latent.py → transformer_latent_label_masked_mean_element.csv
    → build_spaces.py → 6 benchmark space CSVs
    → evaluate_spaces.py → all_space_benchmark_summary.csv
```

### 核心算式

```
T_s = TransformerEncoder([
    LayerNorm(GELU(W_elem · [x_s,e; v_s,e; E_e] + b_elem))  for e in 0..13,
    LayerNorm(GELU(W_dep · [0; l_s; D_id] + b_dep))
])  ∈ R^{15×64}

元素预测: element_head(T_s[0:14])          ∈ R^{14}     (回归)
矿床预测: deposit_head(T_s[14])            ∈ R^{6}      (分类, 仅在被mask时计算loss)
```

### 模型参数总量

```
ElementTokenTransformer:
  element_embedding.weight     14 × 32 = 448
  deposit_embedding.weight      7 × 32 = 224
  element_projector           34×64 + 2×64 = 2,304
  deposit_projector           34×64 + 2×64 = 2,304
  TransformerEncoder (2层)    ~66,000
  element_head                64×1 + 1 = 65
  deposit_head                64×6 + 6 = 390
  ─────────────────────────────────────
  Total                        ~72,000
```

---

## 7. 审核文件清单

```
stage3_element_token_model/audit/
├── STAGE3_AUDIT_REPORT.md         ← 本报告
├── scripts/
│   ├── verify_q1_embeddings.py    ← 验证 embedding 初始化和梯度
│   ├── verify_q2_token_construction.py ← 验证 token 构建维度链
│   ├── verify_q3_masking.py       ← 验证 mask 机制和泄漏检测
│   └── verify_q4_misc.py          ← 杂项逻辑检查
└── outputs/
    ├── q1_weight_comparison.csv   ← 训练前后权重对比
    ├── q1_gradient_status.csv     ← 梯度流通状态
    ├── q3_mask_audit.json         ← mask 审计结果
    └── q4_misc_audit.json         ← 杂项检查汇总
```
