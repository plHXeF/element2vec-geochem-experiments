# Goal

在以下三个已有实验目录 / Git repository 的 checkpoint 基础上，实现一个独立、可复现的 **Latent Evaluation Benchmark v0.1**，统一比较三个实验的 latent 与传统 baseline。

实验位置：

```text
EXPERIMENT_1 = <填入目录或 Git>
EXPERIMENT_2 = <填入目录或 Git>
EXPERIMENT_3 = <填入目录或 Git>
```

不要重新训练三个原模型，不要修改其训练逻辑，不要覆盖原 checkpoint。

允许读取原项目的：

* config；
* dataset schema；
* preprocessing；
* tokenizer；
* model definition；
* checkpoint；
* latent extraction 相关实现。

Benchmark 本身放在独立目录：

```text
latent_evaluation_benchmark/
```

如果当前 workspace 已包含三个项目，自动识别并建立 adapter；否则从上述三个路径读取。

---

# 1. 首先完成 repo audit

对三个实验分别确认：

1. checkpoint 路径；
2. model load 方法；
3. latent 的定义与提取位置；
4. 输入字段及 token/schema；
5. dataset 路径；
6. sample ID；
7. train/val/test split；
8. 标准化统计；
9. missing-value 表示；
10. 当前模型能否直接接受 synthetic record。

生成：

```text
outputs/audit/
  experiment_1.md
  experiment_2.md
  experiment_3.md
  compatibility.json
```

如果三个实验的数据版本存在差异：

* 优先使用 sample ID 交集进行真实数据 benchmark；
* 明确记录 feature/schema 差异；
* synthetic ProcessBank 对所有模型输入相同的地球化学状态；
* 模型不支持的字段按其原始 missing-value 机制处理；
* 不得通过针对某一个模型单独优化 synthetic 数据来提高表现。

---

# 2. 建立统一 Representation API

实现类似：

```python
class Representation:
    def fit_reference(self, train_records): ...
    def encode(self, records) -> np.ndarray: ...
```

至少建立：

```text
ZVectorRepresentation
GeoNormRepresentation
PCAGeoRepresentation
Experiment1Representation
Experiment2Representation
Experiment3Representation
```

learned representation 只 load checkpoint 并 encode。

缓存所有 latent，避免重复推理。

---

# 3. Baselines

## Z-Vector

固定 feature order。

统计量只使用 train split。

数值：

```text
z = (x - train_mean) / train_std
```

missing value 设置为 standardized 0，并附加 missing mask。

## GeoNorm

实现统一 preprocessing：

* major：closure + CLR；
* trace：log transform + robust scaling；
* REE：加入统一 reference-normalized log ratios；
* missing mask 保留。

所有 pseudocount、reference constants、feature list 和训练统计写入 config，不允许隐式存在代码中。

## PCA-Geo

```text
GeoNorm -> PCA
```

保留 95% variance，最大 64 dimensions。

PCA 只 fit train split。

---

# 4. Decodability Benchmark

Targets 第一版：

```text
rock_group
rock_origin
SiO2
Mg#
ΣREE
```

如果 target 本身是 encoder 输入，生成该任务数据时从所有 representation 的输入中 mask 相应字段。

Probe 固定为：

```text
Linear
MLP[32]
MLP[128,64]
MLP[512,256,128]
```

统一：

```text
GELU
AdamW
early stopping
same search budget
same train/val/test
```

分类：

```text
balanced_accuracy
macro_F1
```

回归：

```text
R2
Spearman
```

记录每个 probe 的实际 trainable parameter count。

生成：

```text
outputs/decodability/metrics.parquet
outputs/decodability/curves/*.png
outputs/decodability/decodability_overview.png
```

`decodability_overview.png` 必须是一张包含所有 target 的 small-multiple Figure；每个 panel 中同时画 ZVector、GeoNorm、PCAGeo 和三个实验的 Probe-Complexity Curve。

横轴：

```text
log10(probe parameter count)
```

纵轴为对应 performance。

同时保留每条单独曲线。

计算 curve AUC，但不能用 AUC 替代原始 curve。

---

# 5. Information Sufficiency

## 5.1 Masked Recovery

在真实 test records 中随机 mask 20% 可用数值字段。

同一 mask realization 必须供所有 representation 共用。

分组：

```text
major
trace
REE
isotope
age/derived
```

encode masked record 后，只使用 linear probe 恢复 masked value。

输出：

```text
R2
Spearman
```

## 5.2 Nuisance Leakage

实现：

```text
source/reference prediction
missingness-cluster prediction
same-source kNN enrichment
```

missing pattern 先编码并聚成固定数量类别，再使用 linear probe。

输出 Wanted Information vs Nuisance Leakage scatter figure。

---

# 6. Semantic Geometry

所有距离评价在原始 representation 中计算。

不得在 PCA / UMAP 后计算 benchmark metric。

距离统一使用 Euclidean distance；representation 只允许统一 global scale normalization，不进行 whitening。

## 6.1 Continuous Semantic Triplets

指标：

```text
SiO2
Mg#
total alkali
ΣREE
La/Yb
Eu/Eu*
Sr/Y
```

对每个 anchor：

* positive 从 target difference 的低分位候选中采样；
* negative 从高分位候选中采样；
* 尽量在同 rock_group / 相似 availability 条件内匹配。

计算：

```text
distance(anchor, positive) < distance(anchor, negative)
```

得到 triplet accuracy。

## 6.2 Hierarchy Triplets

基于：

```text
rock_group
rock_origin
```

分别构造 same-origin / different-origin 与 same-group / different-group triplet。

## 6.3 Local Semantic Smoothness

构建：

```text
k = 20
```

latent kNN graph。

对连续指标计算 graph smoothness，并用多次 random permutation 归一化。

输出：

```text
continuous_triplet_accuracy
hierarchy_triplet_accuracy
local_semantic_smoothness
```

---

# 7. 建立 GeoProcessBank

目录：

```text
process_bank/
  minerals.py
  rock_generator.py
  trace_generator.py
  operators/
    mixing.py
    fractionation.py
    partial_melting.py
    alteration.py
  composite.py
  episode.py
```

所有 process episode 必须可以序列化。

每个 episode 保存：

```text
episode_id
seed
source_type       # synthetic / semi-synthetic
initial_state
process_sequence
process_parameters
states
progress_parameter
stage_boundaries
```

---

# 8. Ideal Mineral Generator

通过矿物化学式计算 oxide mass fraction。

至少实现：

```text
Quartz
Ab
An
Or
Fo
Fa
En
Fs
Di
Hd
Magnetite
Ilmenite
Apatite
```

建立 solid-solution calibration：

```text
Ab -> An
Fa -> Fo
selected pyroxene paths
```

每条默认：

```text
33 states
```

不要手填连续过程中的 oxide vector；由 stoichiometry 计算。

---

# 9. Ideal Rock Generator

modal groups：

```text
[Qz, Pl, Kfs, Px, Ol, Accessory]
```

centers：

```text
Felsic       [0.30, 0.30, 0.25, 0.08, 0.00, 0.07]
Intermediate [0.10, 0.45, 0.12, 0.25, 0.02, 0.06]
Mafic        [0.00, 0.45, 0.02, 0.40, 0.08, 0.05]
Ultramafic   [0.00, 0.05, 0.00, 0.35, 0.55, 0.05]
```

sample：

```python
modal ~ Dirichlet(30 * modal_center)
```

同时随机：

```text
XAn
XFo
pyroxene component ratios
accessory composition
```

whole-rock major：

```text
sum(modal_fraction * mineral_composition)
```

然后 closure 至 100%。

REE pattern：

```text
log(CN_i) = a0 + a1*u_i + a2*u_i^2 + A_Eu * I(i == Eu)
```

随机生成总丰度、slope、curvature、Eu anomaly。

其它 trace 从真实 train split 拟合低维 empirical prior 后生成。

所有 synthetic range 要与真实训练集 percentile 对照；记录 OOD 程度，但不得为了让某个模型表现更好而单独裁剪。

---

# 10. Semi-synthetic Initial States

从真实 train/test 数据的合法 chemistry records 中随机采样初始状态，再应用与 synthetic 完全相同的 operators。

Synthetic 与 Semi-synthetic 必须分别统计。

---

# 11. Process Operators

## Mixing

```text
x(t) = (1-t)A + tB
t ∈ [0,1]
33 states
```

## Fractionation

Trace：

```text
C_i(F) = C0_i * F^(D_i - 1)
F: 1 -> 0.1
```

D 按 compatible / neutral / incompatible group 生成相关随机参数。

Major：

逐步移除随机 cumulate mineral assemblage，并做质量平衡。

## Batch Partial Melting

```text
C_L,i = C0_i / (D_i + F*(1-D_i))
F ∈ [0.01,0.5]
```

v0.1 law metric 重点评价 trace。

## Alteration

```text
log C_i(t) = log C_i(0) + t*a_i
```

`a` 为 sparse vector：

* 约 70% 元素不变；
* 其余 enrichment/depletion；
* selected element groups share common component。

这是 synthetic law test，不要把它命名成某种具体自然热液蚀变模型。

---

# 12. Process Geometry Metrics

对每条 encoded trajectory 计算：

## Tangent coherence

```text
mean cosine(delta_z[i], delta_z[i+1])
```

## Tortuosity / geometric simplicity

```text
T = total_path_length / endpoint_distance
simplicity = 1 / T
```

## Local Rank-1

连续 7 个 trajectory states 滑窗 PCA：

```text
lambda1 / sum(lambda)
```

取 median。

## Folding

忽略 index distance <= 3 的邻居。

比较 non-local nearest distance 与 typical adjacent distance，计算 non-local collision rate。

保存 episode-level metric，不能只保存 overall mean。

---

# 13. Linear Process Operator

每种 process、每种 representation 分别训练：

```text
delta_z / delta_t = A z + B theta + b
```

使用 Ridge regression。

train/val/test 必须按 episode 和 initial state 切分：

```text
70 / 15 / 15
```

禁止将同一 trajectory 的不同 state 拆到不同 split。

评价：

```text
normalized next-state prediction error
R2 where meaningful
```

输出 representation × process 的 operator-error matrix。

---

# 14. Composite Process Generator

随机生成：

```text
sequence length L = 2 or 3
```

例如：

```text
Fractionation -> Mixing -> Alteration
```

后一 stage 的输入必须严格为前一 stage 最终状态。

每个 stage 独立随机 process parameters。

如果产生非法 chemistry：

```text
reject episode and resample
```

生成两组：

```text
heterogeneous composites
same-process composites
```

same-process composites 用于 law composition。

---

# 15. Composite Evaluation

## Boundary Detection

真实 stage boundary 已知。

计算：

```text
1 - cosine(delta_z[i-1], delta_z[i])
```

进行简单 change-point detection。

预测位置在真实 boundary ±2 trajectory steps 内视为命中。

## Operator Composition

比较：

```text
T_hat_2(T_hat_1(z))
```

与：

```text
R(T_2(T_1(x)))
```

计算 normalized error。

对于 Fractionation / Alteration 等具有明确 composition law 的情况，额外直接检查对应 semigroup relation。

## Non-commutativity

从同一 initial state 和相同参数构造：

```text
A -> B
B -> A
```

计算 raw chemistry 中的 order difference 与 latent 中的 order difference。

报告：

```text
Spearman(raw_order_effect, latent_order_effect)
```

---

# 16. Monte Carlo

v0.1：

每种单 process：

```text
1000 synthetic
1000 semi-synthetic
```

即：

```text
2000 episodes / process
```

Composite：

```text
2000 episodes
```

所有 representation 必须使用同一个已保存 EpisodeBank。

绝对禁止每个模型独立随机生成 process。

对 model vs baseline 使用 paired statistics：

```text
delta_metric[episode]
```

报告：

```text
median
bootstrap 95% CI
5th percentile
95th percentile
win rate
```

bootstrap：

```text
2000 resamples
```

如果 normalized metric 的 95% CI half-width > 0.02，或 rate metric > 2 percentage points：

```text
2000 -> 4000 -> 8000 episodes
```

继续追加 episode，直到精度达到阈值或达到 8000。

保存 convergence curve。

---

# 17. Process Animation

可视化只用于解释，所有 metric 仍在原始 latent space 中计算。

为每种 representation 单独 fit 一个固定 PCA3：

```text
representation -> PCA3
```

PCA 使用整个 ProcessBank visualization reference set 一次性拟合。

严禁每个 episode 重新 PCA。

动画采用同步 small multiples。

建议布局：

```text
Raw chemistry / element evolution | ZVector PCA3
                                   | GeoNorm PCA3
                                   | PCA-Geo
                                   | Experiment 1 PCA3
                                   | Experiment 2 PCA3
                                   | Experiment 3 PCA3
```

每一帧显示：

```text
current process
current t / F / other parameter
current chemistry
past trajectory trail
current PCA point
process direction
stage boundary
```

Composite 动画必须显示当前 stage 名称和 boundary。

至少生成：

```text
Mixing
Fractionation
Alteration
Composite
```

的动画。

每种选择：

```text
median episode
high model-baseline disagreement episode
fixed-seed random episode
```

不得人工 cherry-pick。

优先输出：

```text
MP4
```

同时保存必要的静态 keyframes；如果环境没有 ffmpeg，再 fallback 到 GIF。

---

# 18. Figures

必须生成：

```text
figures/
  decodability_overview.png
  decodability_tasks/
  sufficiency_vs_leakage.png
  semantic_geometry.png
  process_benchmark.png
  process_operator_error.png
  monte_carlo_convergence.png
```

`process_benchmark.png` 至少包含：

```text
Process ×
[
  Tangent coherence,
  Geometric simplicity,
  Local Rank-1,
  No-fold,
  Linear operator error,
  Boundary detection,
  Composition error
]
```

优先显示 model 相对 GeoNorm / ZVector 的 paired delta 与 bootstrap CI，而不是只显示绝对值。

---

# 19. Outputs

建议结构：

```text
latent_evaluation_benchmark/
  configs/
  src/
    adapters/
    representations/
    probes/
    semantic/
    process_bank/
    operators/
    visualization/
    statistics/
  cache/
  outputs/
    audit/
    decodability/
    sufficiency/
    semantic_geometry/
    process/
    composite/
    figures/
    animations/
    tables/
    logs/
  tests/
  README.md
  REPORT.md
```

核心数值结果同时输出：

```text
CSV/Parquet
JSON summary
```

不要让结果只存在 PNG 中。

---

# 20. Reproducibility

固定：

```text
master seed
episode seeds
data split hashes
checkpoint hashes
feature-schema hashes
baseline statistics
PCA parameters
probe configs
```

生成：

```text
outputs/run_manifest.json
```

所有 Figure / Table / Animation 必须能从 cache + manifest 重新生成，而不重新跑模型。

---

# 21. Sanity Tests

正式运行前必须验证：

1. Binary Mixing 在 ZVector 中应近似形成直线；
2. 同一 EpisodeBank 对六个 representation 完全一致；
3. PCA visualization 不参与任何 metric；
4. train statistics 没有使用 test data；
5. operator 没有 trajectory leakage；
6. mask target 没有直接泄漏；
7. 同一 checkpoint 重复 encode 数值一致；
8. Monte Carlo paired episode ID 完全对齐。

为这些行为写 automated tests。

---

# 22. 最终报告

`REPORT.md` 不输出一个人为加权的总分。

按四级组织：

```text
1. Decodability
2. Information Sufficiency
3. Semantic Geometry
4. Process / Law
```

重点比较每个 experiment 相对：

```text
ZVector
GeoNorm
PCA-Geo
```

分别：

* 得到了什么；
* 丢失了什么；
* 哪些 process 被组织得更简单；
* 哪些 raw-space 原本很简单的关系反而被 encoder 扭曲；
* operator complexity 是否下降；
* Composite / process-order structure 是否得到保留。

最终回答的核心问题不是：

> Which model has the highest latent score?

而是：

> **What information and geological processes does each learned representation reorganize relative to conventional geochemical coordinate systems, and what does it sacrifice in doing so?**

---

# Acceptance Criteria

任务完成必须同时满足：

* 三个 checkpoint 均可稳定提取 latent；
* 三个传统 baseline 可复现；
* Probe-Complexity Curve 完整；
* Sufficiency 与 nuisance leakage 完成；
* Semantic Geometry 三类指标完成；
* Synthetic + Semi-synthetic ProcessBank 完成；
* Mixing / Fractionation / Partial Melting / Alteration 可生成并通过基本 sanity test；
* 四个 trajectory metric 完成；
* linear process operator 完成；
* Composite generator、boundary、composition 与 non-commutativity test 完成；
* paired Monte Carlo + bootstrap CI 完成；
* 原始 chemistry + 六种 representation PCA3 的同步动画完成；
* 所有核心结果有机器可读数据；
* README 写明一条命令如何复现实验；
* 不修改、不覆盖三个原始实验的训练结果。
