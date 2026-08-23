# 当前进展与下一步计划

## 1. 这份文档要说明什么

这份文档集中回答两个问题：

1. 当前项目已经推进到了哪一步；
2. 接下来最值得投入的工作是什么。

从整体上看，项目已经走过了“先把 clean 数据集做出来，再在其上验证地球化学表征”的阶段，正在进入下一步：把**特定地球化学指标**整理成统一、可复用、可进入模型的编码对象。

## 2. 当前已经完成的工作

### 2.1 clean 数据治理已经完成

这部分工作的重点，是把数据库中的记录整理成可信、可比较的样品对象。只有这一层打稳，后面的训练、切分和评估才有意义。

当前的 clean 工作流主要分成三步。

#### 第一步：属性层 canonical mapping

先处理“同一个实体被多个 ID 表示”的问题。已经完成 canonical 化的维表包括：

- `reference`
- `method`
- `age`

这一层最典型的情况是：

1. 同一篇文献被多个 `ref_id` 表示；
2. 同一种分析方法被多个 `method_id` 表示；
3. 同一个年龄对象被多个 `age_id` 表示。

如果这些 ID 不先统一，样品级去重和 train/test 隔离都会受到干扰。举一个最直接的例子：两条样品记录其实来自同一篇文献，但因为 `ref_id` 不同，系统就可能把它们误判为两个独立来源。这样一来，跨文献去重、文献隔离协议和最终结论的可信度都会受到影响。

因此，属性层 canonical mapping 的作用，是先把样品所依附的来源实体统一成可比较的对象。

#### 第二步：样品级去重

第二步处理“不同记录是否其实对应同一个真实样品”的问题。

当前的去重并不是简单按某一列去重，而是结合以下信息进行跨文献判断：

1. 空间位置；
2. 主量元素指纹；
3. 微量元素指纹；
4. 同位素指纹。

同时，当前策略明确区分了两类情况：

1. 同文献内重复：这类记录可能反映采样设计本身，因此不会直接删除；
2. 跨文献重复：更可能是同一样品被多次收录，会在重复组内优先保留字段更完整的样本。

这一步的目标，是让后续模型尽量面向唯一、可信的样品对象训练，而不是在重复记录上反复学习。

#### 第三步：重建 clean 研究视图

在 canonical mapping 和样品级去重完成后，项目没有继续直接在原始表上做实验，而是重建了统一的 clean 研究视图：

- `sample_canonical_ids`
- `v_non_duplicate`
- `v_high_quality_dataset`

这三个对象可以理解为一条连续的加工链：

1. `sample_canonical_ids`
   - 先把样品挂接到 canonical 后的属性 ID；
2. `v_non_duplicate`
   - 在 canonical 样品基础上去掉跨文献重复；
3. `v_high_quality_dataset`
   - 在去重基础上进一步加入主量完整、坐标有效、年龄有效等研究约束。

目前绝大多数模型实验都默认建立在 `v_high_quality_dataset` 上。

#### 为什么这一部分是后续实验的前提

clean 数据治理会同时影响三个层面：

1. 样品对象是否可信；
2. 协议切分是否可信，尤其是 `spatial_and_ref` 这类更严格的切分；
3. 实验结论是否可信，避免重复样品或脏 `ref_id` 穿过 train/test 边界。

可以用一句话概括这一部分工作的意义：

- 当前项目后续所有表征学习与指标编码实验，都建立在 clean 后的样品对象之上，而不是直接建立在原始数据库记录之上。

### 2.2 纯地球化学表征 backbone 已经跑通

在 clean 数据集上，当前已经完成了纯地球化学表征 backbone 的验证。具体包括：

1. 基于 `major + trace` 的两模态 Cross-Attention MAE 预训练；
2. 微量补全任务上的 `scratch vs pretrained` 对比；
3. 多个下游标签任务上的表征迁移实验；
4. `full` 与 `major_only` 两种输入视角的对比；
5. `spatial_only` 与部分 `spatial_and_ref` 协议的对比。

这里需要说明清楚：所谓“多种表征迁移实验”，本质上就是**预训练之后的下游实验**。它们遵循同一条流程：

1. 先用 `major + trace` 做自监督预训练；
2. 提取 `pretrained embedding`；
3. 在具体标签任务上检验这个 embedding 是否优于 `scratch embedding` 或原始浅层特征。

这些实验真正回答的问题是：预训练得到的地球化学表征，能否稳定迁移到具体下游任务。

到目前为止，这条线已经说明了四点：

1. clean 数据治理是必要前提；
2. 纯地球化学表征在微量补全上有稳定收益；
3. 纯地球化学 latent 在部分分类任务上具有迁移性；
4. 迁移收益与任务类型关系很大，并不是所有任务都会同步受益。

#### 几种 baseline / embedding 的区别

为了便于后续讨论，这里把当前实验中几种常见设置的含义统一说明如下。

1. `Raw major baseline`
   - 不经过表征学习；
   - 直接把标准化后的主量特征送入线性分类器；
   - 用来检验主量本身的线性可分性。

2. `Raw major+trace baseline`
   - 不经过表征学习；
   - 直接把主量和微量拼接后送入线性分类器；
   - 用来检验原始地球化学观测本身的线性可分性。

3. `Scratch embedding (full)`
   - 使用与主模型相同的 encoder 结构；
   - 不加载预训练权重；
   - 用随机初始化的模型提取 latent；
   - 用来检验“模型结构本身”能带来多少收益。

4. `Pretrained embedding (full)`
   - 使用相同的 encoder 结构；
   - 加载自监督预训练权重；
   - 输入包含 `major + trace`；
   - 用来检验预训练得到的地球化学表征能否迁移到下游任务。

5. `Pretrained embedding (major_only)`
   - 仍然使用预训练权重；
   - 但把 trace 视角整体当作 missing；
   - 只给主量输入；
   - 用来检验预训练表征在视角缺失条件下是否仍然稳健。

这组对比实际对应三个问题：

1. 原始观测值本身够不够用；
2. 收益来自模型结构，还是来自预训练；
3. 预训练表征在缺失视角下是否仍然稳定。

#### 纯地球化学表征的结构原理图

下面用简化的方式画出当前 backbone 与几种 baseline 的关系。

`Raw major baseline`

```text
major values
   |
standardize
   |
linear classifier
   |
prediction
```

`Raw major+trace baseline`

```text
major values -----.
                  +--> concatenate --> linear classifier --> prediction
trace values -----'
```

`Scratch embedding (full)`

```text
major values -----> major encoder ----.
                                       \
                                        > major<->trace fusion --> pooled latent --> linear classifier --> prediction
                                       /
trace values -----> trace encoder ----'
```

说明：

- 使用与主模型相同的网络结构；
- 不加载预训练权重；
- 输入中 `major` 和 `trace` 都是完整的。

`Pretrained embedding (full)`

```text
major values -----> pretrained major encoder ----.
                                                  \
                                                   > pretrained major<->trace fusion --> pooled latent --> linear classifier --> prediction
                                                  /
trace values -----> pretrained trace encoder ----'
```

说明：

- 结构与 `Scratch embedding (full)` 相同；
- encoder 和 fusion 权重来自自监督预训练。

`Pretrained embedding (major_only)`

```text
major values -----> pretrained major encoder ----.
                                                  \
                                                   > pretrained major<->trace fusion --> pooled latent --> linear classifier --> prediction
                                                  /
trace input = missing ---------------------------'
```

说明：

- 使用同一个 pretrained 模型；
- 只是把 `trace` 一侧整体当作 missing；
- 用来检验视角缺失时的鲁棒性。

#### 当前已经拿到的关键下游结果

下面列出当前最关键的一组下游结果，用于后续统一表述。

`rock_type_top15`（`spatial_only`）

- `Raw major+trace baseline`：MacroF1 = `0.2951`
- `Scratch embedding (full)`：MacroF1 = `0.6089`
- `Pretrained embedding (full)`：MacroF1 = `0.6344`
- `Pretrained embedding (major_only)`：MacroF1 = `0.6419`

这一结果说明，细粒度岩石学任务对预训练表征是有响应的，而且 `major_only` 视角并没有明显失效。

`igneous_origin_2way`（`spatial_only`）

- `Raw major baseline`：MacroF1 = `0.7318`
- `Raw major+trace baseline`：MacroF1 = `0.7276`
- `Scratch embedding (full)`：MacroF1 = `0.7471`
- `Pretrained embedding (full)`：MacroF1 = `0.7360`
- `Pretrained embedding (major_only)`：MacroF1 = `0.7458`

在这个任务上，预训练没有稳定超过 scratch，说明表征优势并不明显。

`rock_group_3way`（`spatial_only`）

- `Raw major+trace baseline`：MacroF1 = `0.5495`
- `Scratch embedding (full)`：MacroF1 = `0.6317`
- `Pretrained embedding (full)`：MacroF1 = `0.6229`
- `Pretrained embedding (major_only)`：MacroF1 = `0.6116`

`rock_group_3way`（`spatial_and_ref`）

- `Raw major+trace baseline`：MacroF1 = `0.3918`
- `Scratch embedding (full)`：MacroF1 = `0.6048`
- `Pretrained embedding (full)`：MacroF1 = `0.5840`
- `Pretrained embedding (major_only)`：MacroF1 = `0.5074`

可以看出，在更严格协议下，任务难度明显提高，当前预训练也没有稳定胜过 scratch。

`age_bin_5way`（`spatial_only`）

- `Raw major+trace baseline`：MacroF1 = `0.3287`
- `Scratch embedding (full)`：MacroF1 = `0.3778`
- `Pretrained embedding (full)`：MacroF1 = `0.3891`
- `Pretrained embedding (major_only)`：MacroF1 = `0.3892`

年龄分箱任务上，预训练有小幅正增益，但幅度有限。

`rock_origin_7way`

当前已有两份结果：

- `spatial_only`：`Pretrained embedding (full)` MacroF1 = `0.3609`
- `spatial_and_ref`：`Pretrained embedding (full)` MacroF1 = `0.3270`

这个任务目前仍然比较困难，粗粒度混合标签的可迁移性还不稳定。

### 2.3 指标覆盖实验已经完成，并留下了具体指标

这一部分在之前的文档里写得不够清楚。实际上，第一轮指标覆盖实验已经给出了明确的指标名单、覆盖率和计算公式，结果保存在：

- [results/reports/indicator_coverage_experiment_clean.md](/mnt/wd_disk/Projects/Geo-chemFoundation/results/reports/indicator_coverage_experiment_clean.md)
- [results/reports/indicator_coverage_experiment_clean.csv](/mnt/wd_disk/Projects/Geo-chemFoundation/results/reports/indicator_coverage_experiment_clean.csv)

当前已经明确的指标可以分成三组。

#### 第一批优先指标

这批指标覆盖率高、分布稳定，足以作为统一编码的第一层派生指标：

1. `mali`
   - 来源：`computed`
   - 有效覆盖率：`99.98%`
   - 中位数：`-1.972`
2. `cia`
   - 来源：`computed`
   - 有效覆盖率：`99.98%`
   - 中位数：`45.062`
3. `wip`
   - 来源：`computed`
   - 有效覆盖率：`99.98%`
   - 中位数：`76.393`
4. `asi`
   - 来源：`computed`
   - 有效覆盖率：`99.97%`
   - 中位数：`0.820`
5. `mg_number`
   - 来源：`computed`
   - 有效覆盖率：`99.97%`
   - 中位数：`0.534`
6. `sr_y`
   - 公式：`sr_ppm / y_ppm`
   - 有效覆盖率：`62.30%`
   - 中位数：`10.043`

#### 第二批扩展候选

这批指标已经可以计算，但覆盖率明显低于第一批，更适合作为扩展层：

1. `la_sc`
   - 公式：`la_ppm / sc_ppm`
   - 有效覆盖率：`35.50%`
   - 中位数：`1.600`
2. `th_sc`
   - 公式：`th_ppm / sc_ppm`
   - 有效覆盖率：`31.72%`
   - 中位数：`0.290`
3. `ce_pb`
   - 公式：`ce_ppm / pb_ppm`
   - 有效覆盖率：`31.66%`
   - 中位数：`5.000`
4. `la_yb`
   - 公式：`la_ppm / yb_ppm`
   - 有效覆盖率：`30.26%`
   - 中位数：`10.009`
5. `nb_u`
   - 公式：`nb_ppm / u_ppm`
   - 有效覆盖率：`28.29%`
   - 中位数：`5.500`
6. `dy_yb`
   - 公式：`dy_ppm / yb_ppm`
   - 有效覆盖率：`17.80%`
   - 中位数：`1.870`

#### 当前还不能正式纳入的指标

这组指标并不缺少地学意义，只是当前字段条件或规则版本还不够稳定：

1. `eu_eu_star_proxy`
   - 公式：`eu_ppm / sqrt(sm_ppm * gd_ppm)`
   - 当前有效覆盖率：`20.24%`
   - 主要问题：需要锁定 REE 归一化基准，目前只能算 proxy
2. `ce_ce_star_proxy`
   - 公式：`ce_ppm / sqrt(la_ppm * pr_ppm)`
   - 当前有效覆盖率：`0.00%`
   - 主要问题：`v_high_quality_dataset` 目前没有暴露 `pr_ppm`
3. `zr_ti`
   - 公式：`zr_ppm / ti_ppm`
   - 当前有效覆盖率：`0.00%`
   - 主要问题：当前视图没有暴露 `ti_ppm`
4. `ti_y`
   - 公式：`ti_ppm / y_ppm`
   - 当前有效覆盖率：`0.00%`
   - 主要问题：当前视图没有暴露 `ti_ppm`

因此，现在可以明确地说：运行过程中已经留下了具体指标，而且这些指标已经完成了第一轮覆盖率筛选。

### 2.4 indicator 如何进入模型，已经有了初步结论

围绕 indicator 如何进入模型，当前已经比较过四种方案：

1. 线性拼接；
2. 轻量三模态预训练；
3. MLP adapter；
4. `indicator -> backbone` cross-attention adapter。

目前形成的结论比较清晰：

1. 轻量三模态预训练没有超过“现有两模态 pretrained + indicator 拼接”；
2. 线性拼接能够带来可见提升，说明 indicator 本身有信息量；
3. MLP adapter 在部分任务上优于线性拼接；
4. `indicator -> backbone` cross-attention adapter 在细粒度岩石学任务上效果最好；
5. 这种收益具有明显的任务依赖性。

换句话说，问题已经不再是“indicator 有没有用”，而是“不同任务更适合怎样的 indicator 融合方式”。

#### 方式一：线性拼接

流程如下：

1. 先用现有两模态 backbone 提取 `pretrained embedding`；
2. 把 indicator 编码成一个向量，包含：
   - value
   - observed flag
3. 将 `embedding` 与 `indicator` 直接横向拼接；
4. 用线性分类器做下游任务。

这种方式最简单，也最稳，但表达能力相对有限。它没有显式建模“某个指标应该从地球化学表征中读取哪一部分信息”。

结构图如下：

```text
major -----> pretrained backbone ----.
trace -----> pretrained backbone ----+--> pooled embedding --.
                                     |                      |
indicator values --------------------'                      |
        |                                                  |
   value + flag encoding ----------------------------------+--> concatenate --> linear classifier --> prediction
```

#### 方式二：MLP adapter

流程如下：

1. 固定现有两模态 backbone；
2. 提取 `pretrained embedding`；
3. 将 indicator 编码成向量；
4. 让 `embedding` 分支和 `indicator` 分支分别经过一个小型 MLP；
5. 将两路结果融合后做分类。

这种方式比线性拼接更灵活，但仍然属于向量级融合，没有显式建模 token 级条件读取关系。

结构图如下：

```text
major -----> pretrained backbone ----.
trace -----> pretrained backbone ----+--> pooled embedding --> emb MLP ---.
                                                                         |
indicator values --> value + flag encoding --> indicator MLP ------------+--> fusion MLP --> prediction
```

#### 方式三：`indicator -> backbone` cross-attention adapter

这是当前最新、在细粒度岩石学任务上效果最好的一种方式。

流程如下：

1. 保留原来的两模态 backbone：
   - `major -> encoder`
   - `trace -> encoder`
   - `major <-> trace cross-attention`
2. 得到 `major_ctx tokens` 和 `trace_ctx tokens`；
3. 把两者拼成一条 `backbone token sequence`；
4. 将 indicator 编码成 `indicator tokens`；
5. 让 `indicator tokens` 作为 query，对 `backbone tokens` 做 cross-attention；
6. 分别池化：
   - `major_ctx`
   - `trace_ctx`
   - `indicator-conditioned tokens`
7. 最后拼接并做分类。

这个方案最核心的想法，是让 indicator 作为条件查询，去读取纯地球化学 backbone 中已经学到的结构信息。

结构图如下：

```text
major -----> pretrained major encoder ----.
                                           \
                                            > pretrained major<->trace fusion --> backbone tokens ----.
                                           /                                                          |
trace -----> pretrained trace encoder ----'                                                           |
                                                                                                      |
indicator values --> indicator token encoder --> indicator tokens ------------------------------------+
                                                                                                      |
                                                                                                      v
                                                                        indicator --> backbone cross-attention
                                                                                                      |
                                                                                           indicator-conditioned tokens
                                                                                                      |
                               pool(major_ctx) ----.                                                   |
                               pool(trace_ctx) ----+-----------------------------.                     |
                               pool(indicator_ctx) ------------------------------+--> concatenate --> classifier --> prediction
```

它与前两种方式最大的区别，在于 indicator 不再只是一个附加向量，而是被当作 token 序列，主动去读取 backbone 的 token 序列。

#### 当前已经得到的 indicator 融合结果

目前已经拿到的关键结果如下。

`rock_type_top15`

- `Pretrained embedding (linear)`：MacroF1 = `0.6338`
- `Pretrained embedding + core indicator (linear)`：MacroF1 = `0.6425`
- `Pretrained embedding + MLP adapter`：MacroF1 = `0.6754`
- `Pretrained embedding + cross-attn adapter`：MacroF1 = `0.7014`

`igneous_origin_2way`

- `Pretrained embedding (linear)`：MacroF1 = `0.7370`
- `Pretrained embedding + core indicator (linear)`：MacroF1 = `0.7367`
- `Pretrained embedding + MLP adapter`：MacroF1 = `0.7257`
- `Pretrained embedding + cross-attn adapter`：MacroF1 = `0.7426`

`rock_group`

- `Pretrained embedding (linear)`：MacroF1 = `0.5255`
- `Pretrained embedding + core indicator (linear)`：MacroF1 = `0.5259`
- `Pretrained embedding + MLP adapter`：MacroF1 = `0.4982`
- `Pretrained embedding + cross-attn adapter`：MacroF1 = `0.5254`

`age_bin_5way`

- `Pretrained embedding (linear)`：MacroF1 = `0.3887`
- `Pretrained embedding + core indicator (linear)`：MacroF1 = `0.4026`
- `Pretrained embedding + MLP adapter`：MacroF1 = `0.4165`
- `Pretrained embedding + cross-attn adapter`：MacroF1 = `0.3647`

从这些结果可以得到一个相当稳定的判断：

1. `cross-attn adapter` 对细粒度岩石学标签最有效；
2. `igneous_origin_2way` 上也有小幅正增益；
3. `rock_group` 上与线性 pretrained baseline 基本持平；
4. `age_bin_5way` 上并不占优，线性拼接或 MLP 更稳。

## 3. 当前实验说明了什么

综合当前结果，可以把结论概括成三点。

### 3.1 纯地球化学表征已经成立

基于 `major + trace` 的两模态 backbone 已经证明，纯地球化学观测可以学到稳定的样品表征。这条线在微量补全上已经表现出稳定收益，在一部分下游任务上也显示出迁移能力。

### 3.2 特定地球化学指标可以进入统一编码

第一轮指标覆盖实验已经表明，至少有一批指标可以作为稳定编码对象进入系统，包括：

- `mg_number`
- `asi`
- `mali`
- `cia`
- `wip`
- `sr_y`

同时也已经看到，指标进入模型时不能只保留一个数值，还需要同时保留观测状态与有效性信息。

### 3.3 indicator 融合方式具有明显任务依赖性

目前最清楚的结论，不是“已经找到一种对所有任务都最优的融合结构”，而是：

- 特定地球化学指标确实有用；
- 但它们进入模型的方式，和任务类型关系非常密切。

对细粒度岩石学标签，`indicator -> backbone cross-attention` 显示出明显优势；对年龄类任务，线性拼接或 MLP 融合反而更稳。

## 4. 现在最合理的研究主线

接下来的主线，不宜继续定义为“再增加几个分类任务”或“继续堆更多模型结构”。更自然的推进方式是：

1. 先把特定地球化学指标整理成统一编码对象；
2. 再把这些对象组织成标准化的样品编码接口；
3. 最后用不同任务去检验这套编码接口的有效性与任务依赖性。

换句话说，下一阶段真正要做的，是把观测值、指标值、状态信息和来源信息组织成一套统一编码，而不是继续零散试验。

## 5. 统一编码现在应该具备什么结构

建议把统一编码写成四层结构。

### 5.1 原始观测层

这一层保留尽量少解释、尽量少加工的基础观测：

- major
- trace
- isotope
- age
- rockgroup / petrology
- method
- reference
- location

这一层回答的是：样品实际观测到了什么。

### 5.2 派生指标层

这一层记录从原始观测稳定计算出的指标。

当前建议分成三组：

第一层核心指标：

- `mg_number`
- `asi`
- `mali`
- `cia`
- `wip`
- `sr_y`

第二层扩展指标：

- `la_sc`
- `th_sc`
- `ce_pb`
- `la_yb`
- `nb_u`
- `dy_yb`

暂缓纳入指标：

- `eu_eu_star`
- `ce_ce_star`
- `zr_ti`
- `ti_y`

这一层回答的是：从观测值可以稳定得到哪些指标。

### 5.3 状态层

每个观测或指标都应当带有状态，而不只是数值。

当前阶段至少需要统一以下几类状态：

- `measured`
- `derived`
- `missing`
- `invalid`
- `below_detection`
- `uncertain`

对每个 indicator，至少应同时输出：

1. value；
2. observed flag；
3. validity flag；
4. state。

### 5.4 provenance 与约束层

每个指标都应当记录：

- 来源字段；
- 计算公式；
- 适用前提；
- 缺失失效条件；
- 版本号；
- 数据来源或规则来源。

这一层回答的是：该指标是如何得到的，以及在什么条件下可信。

## 6. 下一步工作建议

后续工作适合按“先编码、后接口、再实验”的顺序推进。

### 6.1 第一步：把编码资产写成机器可读对象

这一阶段优先级最高。建议首先生成四份机器可读资产：

1. `feature_dictionary`
2. `derived_indicator_registry`
3. `observation_state_schema`
4. `provenance_schema`

目标是先把“统一编码”写成清晰、可执行的规则。

### 6.2 第二步：构建统一编码接口

在不破坏现有 clean 工作流的前提下，新增统一编码输出。

具体工作包括：

1. 基于 `v_high_quality_dataset` 新增一个 indicator / encoding view；
2. 为第一层核心指标生成标准字段：
   - 数值列
   - observed flag
   - validity flag
   - state 字段
3. 固定命名规范；
4. 输出标准化编码表，供模型与分析统一使用。

这一步完成后，特定地球化学指标就会从“实验变量”升级为“统一编码接口的一部分”。

### 6.3 第三步：基于统一编码接口复现实验

模型实验的角色，应当是验证编码接口，而不是继续零散扩展。

建议的顺序如下：

1. 比较 `raw-only` 与 `raw + indicator`；
2. 在统一编码接口上验证任务依赖性：
   - `rock_type_top15`
   - `igneous_origin_2way`
   - `rock_group`
   - `age_bin_5way`
3. 固定每类任务的推荐融合方式：
   - 细粒度岩石学：`cross-attn adapter`
   - 年龄类：线性或 MLP

## 7. 当前不宜优先推进的方向

按照现在的状态，以下几件事暂时不应作为主线：

1. 无约束地继续增加更多分类标签任务；
2. 在编码规范尚未确定时继续扩展更复杂的三模态预训练；
3. 在状态层和 provenance 层未写清之前直接追求“大一统模型”。

原因很简单：

1. 当前瓶颈已经不在“任务够不够多”；
2. 当前最缺的是标准化编码接口；
3. 当前实验已经表明 indicator 的融合方式明显依赖任务类型，必须先把编码结构固定下来，再做系统比较。

## 8. 当前可以稳定采用的研究口径

截至目前，可以统一采用下面这套表述：

1. 数据库已经完成属性层 canonical 化与样品级去重；
2. clean 版地球化学表征 backbone 已经建立；
3. 预训练在微量补全上已表现出稳定收益；
4. 表征迁移收益具有明显的任务依赖性；
5. 特定地球化学指标的第一轮覆盖率筛选已经完成，并留下了具体指标、公式与覆盖率结果；
6. 多种 indicator 融合方案已经完成比较；
7. 后续工作的重点是建立统一地球化学样品编码标准；
8. 这套标准将以 `原始观测 + 特定指标 + 状态 + provenance` 为基本结构。

## 9. SQLite 数据库导出为 CSV

为了便于后续检查 clean 数据、外部分析或交给其他工具使用，已经新增：

- [`scripts/export_sqlite_tables_to_csv.py`](/mnt/wd_disk/Projects/Geo-chemFoundation/scripts/export_sqlite_tables_to_csv.py)

默认导出当前 clean 工作流核心对象：

- `canonical_reference_map`
- `canonical_method_map`
- `canonical_age_map`
- `dedup_decisions`
- `sample_canonical_ids`
- `v_non_duplicate`
- `v_high_quality_dataset`

从项目根目录执行：

```bash
python scripts/export_sqlite_tables_to_csv.py --overwrite
```

输出目录：

- `results/exports/sqlite_csv/`

如果只需要导出多个指定表或视图，可以直接把对象名连续写在命令后：

```bash
python scripts/export_sqlite_tables_to_csv.py sample v_high_quality_dataset --overwrite
```

查看数据库中可导出的表和视图：

```bash
python scripts/export_sqlite_tables_to_csv.py --list
```

全量导出所有用户表和视图：

```bash
python scripts/export_sqlite_tables_to_csv.py --all --overwrite
```
