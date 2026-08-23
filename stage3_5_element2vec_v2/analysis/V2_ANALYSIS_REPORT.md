# Stage 3.5 Element2Vec V2 — 模型分析报告

**分析日期**: 2026-07-02
**分析范围**: `stage3_5_element2vec_v2/` 全部源码 + 训练日志
**不包含**: 运行时验证 (checkpoint 加载、latent 提取、benchmark)

---

## 0. 总览

| 维度 | 结论 | 状态 |
|------|------|------|
| 代码架构 | 维度链完整, 组件衔接正确 | ✅ 23/23 checks |
| Mask 机制 | one-mask-per-sample 正确实现, 提取时防泄漏 | ✅ 20/20 checks |
| 数据流水线 | Transform/Vocab/SQLite 均正确 | ✅ 15/15 checks |
| 训练收敛 | Loss 下降 59.6x, 但最终 checkpoint 非最优 | ⚠️ 见下文 |
| 模型规模 | 12-layer / d=768 / ~85M params | 偏大但合理 |

**核心发现**: 代码逻辑无 bug，但训练在收敛质量和 checkpoint 保存时机上存在可优化空间。

---

## 1. Phase 1a: 架构维度链验证 ✅ 23/23

### 1.1 Token 构建

```
label_embedding_lookup → R^{509}
  + value               → R^{1}
  + mask_tag_flag       → R^{1}
  + mask_value_flag     → R^{1}
  = token_raw           → R^{512}

TokenProjector:
  Linear(512→768) → GELU → LayerNorm → Dropout
  → Linear(768→768) → GELU → LayerNorm → Dropout
  → Linear(768→768)                    (输出)
  最终: token ∈ R^{768}
```

设计亮点：509 维 label embedding + 3 个标量 = 512，对齐 2^9 的 GPU 友好边界。

### 1.2 CLS Token

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
```

- `[CLS]` 是 768 维直接可训练参数 (非 MLP 投影)，prepend 到序列最前
- CLS 的 attention_mask = 1 (对所有 token 可见)
- 提取时 `encoded[:, 0, :]` 获取 CLS latent ✓

### 1.3 No Positional Embedding

- 训练时 `random.shuffle(tokens)` — 每 epoch 随机打乱
- `mask_position` 在 shuffle 后正确重新映射至新位置
- Transformer 不使用位置编码 → 纯置换不变 (permutation invariant)
- `build_model()` 显式 `pop("use_positional_embedding")` 避免传入 encoder

### 1.4 Forward 中的 mask_position 处理

```python
positions = batch["mask_positions"] + 1  # +1 跳过 CLS (position 0)
masked_hidden = out["encoded"][torch.arange(bsz), positions]
```

正确使用高级索引，从编码序列中取出被 mask 的那个位置的 hidden state。

### 1.5 Decoder 维度

| Decoder | 输入 | 输出 | 任务类型 |
|---------|------|------|---------|
| `numeric_value` | d_model(768) | 1 | Huber 回归 |
| `numeric_tag` | 768 | 35 (34+1) | CrossEntropy 分类 |
| `rock_group` | 768 | 4 (3+1) | CrossEntropy 分类 |
| `rock_origin` | 768 | 10 (9+1) | CrossEntropy 分类 |
| `rock_type` | 768 | 82 (81+1) | CrossEntropy 分类 |

---

## 2. Phase 1b: Mask 机制 ✅ 20/20

### 2.1 One-Mask-Per-Sample 策略

```
每样本每 epoch 随机选择 1 个任务:
  - mask_numeric_value  45%  → 预测被mask的元素值 (Huber回归)
  - mask_numeric_tag    15%  → 预测被mask的元素名称 (35分类)
  - mask_pure_label     40%  → 预测被mask的岩石标签
    ├── rock_type   60%  → 81类中最难, 给予最高采样权重
    ├── rock_origin 25%  → 9类, 中等
    └── rock_group  15%  → 3类, 最简单
```

训练信号总量: 270,241 × 50 = **13.5M signals**

### 2.2 Mask Flags 机制

| Mask 类型 | value | mask_value_flag | tag_id | mask_tag_flag |
|-----------|-------|-----------------|--------|---------------|
| 正常 (未mask) | 实测值 | 0 | 标签ID | 0 |
| mask_numeric_value | 0 | **1** | 标签ID | 0 |
| mask_numeric_tag | 实测值 | 0 | [MASK] | **1** |
| mask_pure_label | 0 | 0 | [MASK_field] | **1** |
| 提取模式 | 实测值 | 0 | [MASK_field] | **1** (全部) |

模型通过 mask_tag_flag 和 mask_value_flag 区分 "天然缺失" vs "被 mask"，这是 Stage 3 pyrite 模型的改进。

### 2.3 提取时防泄漏

```python
# extract_embeddings.py:38
dataset = WholeRockTokenDataset(config, vocab, mode="extract", mask_pure_labels=True)
```

- 所有 pure label token 替换为 field-level `[MASK]` token
- `model.eval()` 关闭 dropout → 确定性输出
- CLS latent 和 mean_numeric latent 均不含标签信息

---

## 3. Phase 1c: 数据流水线 ✅ 15/15

### 3.1 Value Transform

| 类型 | 字段 | 处理 |
|------|------|------|
| Major oxides (13) | SiO2, TiO2, Al2O3, ... | direct robust-z |
| Trace elements (21) | Rb_ppm, Sr_ppm, ... | log10(x+eps) → robust-z |
| Clip | 所有 | |z| > 12 → clip |

### 3.2 数据覆盖度

| 字段 | 覆盖度 |
|------|--------|
| Major oxides | 99–100% |
| Trace elements | 35–70% |
| LOI | 32.7% (最低) |

11 个字段覆盖度 < 50%: `LOI, CO2, La_ppm, Nd_ppm, Th_ppm, U_ppm, Pb_ppm, Sc_ppm, Co_ppm, Ga_ppm, Yb_ppm`

### 3.3 Vocab 结构

| Vocab | 数量 | 说明 |
|-------|------|------|
| Numeric tags | 34 + 1 `[MASK]` | 35 个 token |
| Pure labels | 96 | 3+9+81 类 + 3个 field-level `[MASK]` |
| 总样本 | 270,241 | 来自 `v_high_quality_dataset` view |

### 3.4 509 维 Label Embedding 设计

`509 + 1(value) + 1(mask_tag) + 1(mask_value) = 512` — 凑到 2^9 的 GPU 对齐边界。509 本身是素数，避免了维度与下游 linear 的意外对称耦合。

---

## 4. Phase 2a: 训练质量分析 ⚠️

### 4.1 Loss 收敛曲线

| 指标 | 初始 (step 1) | 最终 (step 52,800) | 最优 | 下降倍数 |
|------|--------------|-------------------|------|---------|
| **Total** | 18.072 | 0.303 | **0.105** (step 50,437) | 59.6× (best 172×) |
| numeric_value | 0.856 | 0.079 | — | 11× |
| numeric_tag | 5.050 | 0.050 | — | 102× |
| rock_group | 1.131 | ~0.000 | — | >10^6 |
| rock_origin | 4.004 | ~0.001 | — | 4495× |
| rock_type | 7.030 | **0.174** | — | 40× |

**图表**: 见 `analysis/outputs/p2a_loss_analysis.png` (6 面板: total/component losses, LR, final 50% zoom, loss velocity, token histogram)

### 4.2 ⚠️ 关键问题: Checkpoint 非最优

```
最优 step = 50,437  (loss = 0.105)
最终 step = 52,800  (loss = 0.303)
最终 loss 是最优的 2.9×
```

**原因**: Cosine LR scheduler 将 LR 衰减至 0，最后 2,000 步 (~5 epochs) 模型无法有效学习，batch 间的随机噪声主导 loss 波动。当前保存的 checkpoint 在 step 52,800，不是最优参数。

### 4.3 ⚠️ 最后阶段的高方差

```
Final 20% steps:
  CV (变异系数) = 0.332
  Range = [0.105, 1.363]
```

Loss 在 0.10 和 1.36 之间剧烈摆动。这通常表明：
1. LR 过小 → 更新方向随机 (已完成检验 ✓)
2. Batch 间难度差异大 → 可能某些 batch 的 task/mask 组合特别难
3. 缺乏 momentum 积累 → 可在 AdamW 中调大 β1

### 4.4 Learning Rate 曲线

| 阶段 | Step | LR |
|------|------|-----|
| 初始 | 1 | 3.77 × 10⁻⁶ |
| Warmup 完成 | 150 | 2.83 × 10⁻⁴ |
| 峰值 | 148 | 2.83 × 10⁻⁴ |
| 最终 | 52,800 | **0** |

**图表**: 见 `analysis/outputs/p2a_lr_warmup.png`

Cosine 衰减在 50 epochs 结束时自然到达 0。这是正常行为，但意味着训练不会进一步收敛。若需更多训练，需重新设置 scheduler。

### 4.5 分任务收敛状态

| 任务 | 最终 loss | 状态 |
|------|----------|------|
| rock_group (3类) | ≈0 | 已记忆化 — 交叉熵接近 0 |
| rock_origin (9类) | ≈0.001 | 近乎记忆化 |
| numeric_tag (35类) | 0.050 | 良好收敛 |
| numeric_value (回归) | 0.079 | 有一定残差 |
| **rock_type (81类)** | **0.174** | ⚠️ 尚未完全收敛 |

81 类交叉熵 0.174 意味着平均困惑度 ≈ exp(0.174) ≈ 1.19，即模型对每个样本有约 1.2 个候选标签 — 说明 top-1 准确率还不够高。这是 81 类的固有难度。

### 4.6 模型参数量估算

| 组件 | 参数量 |
|------|--------|
| numeric_tag_embedding (35×509) | ~18K |
| pure_label_embedding (96×509) | ~49K |
| TokenProjector (512→768→768→768) | ~1.6M |
| CLS token (768) | 768 |
| TransformerEncoder (12层, d=768, 12 heads) | ~85M |
| Decoders (5 head) | ~0.1M |
| **Total** | **~87M** |

对于 270K 样本、13.5M 训练信号的任务来说，87M 参数是相当大的。可能存在过拟合风险，尤其是在无验证集的情况下。

---

## 5. 与 Stage 3 (pyrite) 对比

| 方面 | Stage 3 (pyrite) | Stage 3.5 V2 (whole-rock) |
|------|------------------|--------------------------|
| **数据** | 7,778 样本, 14 元素 | 270,241 样本, 34 元素 |
| **Token 类型** | 固定 15 (14 elem + deposit) | 可变 ~25 (observed only + CLS) |
| **Token 输入** | 34-dim (32+1+1) → 64 | 512-dim (509+1+1+1) → 768 |
| **Transformer** | 2-layer, d=64, 4 heads | 12-layer, d=768, 12 heads |
| **位置编码** | 隐式(固定顺序) | 无 (随机 shuffle → 置换不变) |
| **Mask 策略** | 15% tokens + deposit prediction | 1 task/sample, weighted (45/15/40) |
| **Mask 标志** | 仅 visible_mask (0/1) | mask_tag_flag + mask_value_flag (更精细) |
| **防泄漏** | label_masked → deposit [MASK] | mask_pure_labels → 全 field [MASK] |
| **Embedding 表** | 14 elem + 7 deposit | 35 numeric tags + 96 pure labels |
| **参数** | ~72K | ~87M |
| **解码器** | element_head(64→1) + deposit_head(64→6) | 5 个独立 Linear head |

**核心改进**: V2 引入了 token 级别的 mask flag、无位置编码的置换不变性、field-level [MASK] token，以及更精细的多任务 mask 策略。但模型规模从 72K 扩展到 87M，而训练信号密度从每个样本 ~1.7 个 mask 变为 1 个 mask — 训练效率未同比例提升。

---

## 6. 建议

### 6.1 必须处理

| # | 建议 | 说明 |
|---|------|------|
| 1 | **从最优 step 恢复 checkpoint** | Step 50,437 (loss=0.105) 而非 52,800 (loss=0.303)。可在 train_v2 的 save 逻辑中增加 `best_loss` 追踪和独立保存。 |
| 2 | **运行 extract + benchmark** | 当前 latents/ 和 metrics/ 为空，这是完成分析流程的最后一步。 |

### 6.2 建议优化

| # | 建议 | 说明 |
|---|------|------|
| 3 | **增加验证集** | 随机留出 5-10% 数据做 held-out loss 监控，检测 overfitting |
| 4 | **LR schedule 调优** | 考虑 warmup 50→300 steps (当前 150)，或使用 cosine with warm restart 避免过早衰减至 0 |
| 5 | **增加 batch_size 或使用 gradient accumulation** | batch_size=256 在 87M 参数下偏小，考虑 512 或 accumulate 2-4 steps |
| 6 | **调大 rock_type 权重** | rock_type 81 类仍未完全收敛，可在 loss_weights 中将其权重设为 2.0-5.0 |
| 7 | **删除 joint_mask 相关的冗余代码** | train.py:33-34 的 `elem_loss_mask` 重赋值是冗余的 (同 Stage 3 的问题) |

### 6.3 长期考虑

| # | 建议 | 说明 |
|---|------|------|
| 8 | **Multiple masks per sample** | 从 1 mask/sample 改为 ~15% tokens，类似于 BERT，可大幅提升训练信号密度 |
| 9 | **混合精度训练** | `mixed_precision: true` 可加速 2-3× 并降低显存，允许更大 batch_size |
| 10 | **cudnn deterministic** | 添加 `torch.backends.cudnn.deterministic = True` 提高可复现性 |

---

## 7. 分析产物清单

```
stage3_5_element2vec_v2/analysis/
├── V2_ANALYSIS_REPORT.md               ← 本报告
├── scripts/
│   ├── p1a_verify_architecture.py       ← 23/23 checks passed
│   ├── p1b_verify_masking.py            ← 20/20 checks passed
│   ├── p1c_verify_data_pipeline.py       ← 15/15 checks passed
│   └── p2a_analyze_training.py          ← loss curves + metrics
└── outputs/
    ├── p1a_arch_checks.csv
    ├── p1b_mask_checks.csv
    ├── p1c_data_checks.csv
    ├── p2a_convergence_metrics.csv
    ├── p2a_loss_analysis.png / .svg
    └── p2a_lr_warmup.png
```
