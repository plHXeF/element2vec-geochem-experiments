# Element2Vec V2 多尺度低维语义子空间复验报告

## 1. Executive summary

- 原 registry 的 50 个实验均进入兼容性矩阵；技术/字段不可用项保持不可用，没有用投影伪造结果。
- 主空间定量行数：223；随机正交投影使用每个目标维度 10 个固定种子。
- 严格门槛下，`Low-rank semantic recovery` 行数为 2，`Low-rank but confounded` 为 0，二维/三维 visualization-only 或 suspected projection artifact 为 4。
- 结论必须以 `experiment_dimension_matrix.csv` 的 FULL/PCA/RP 差值、bootstrap 与 source/missingness 对照为准；二维或三维图本身不计为证据。

按原 primary metric（仅修正 D4 的“越低越好”方向）逐实验取最大值，最佳空间计数为：

| best_space_by_primary_metric | experiment_count |
| --- | --- |
| FULL | 33 |
| PCA3 | 6 |
| PCA2 | 4 |
| PCA32 | 3 |
| PCA64 | 2 |

当前证据更支持“不同表示层级具有不同有效维度，且低秩改善是局部的”，而不是“存在一个对所有任务通用的 32D/64D 地质语义空间”。特别是静态 numeric tags 的样本秩只有 34，PCA64 不可被合法估计，PCA32 已接近秩上限；contextual/sample 空间则允许完整比较。

## 2. Projection audit

详见 `audit/subspace_representation_audit.md` 与 `audit/subspace_available_dimensions.json`。PCA 使用固定 8,000 样品 reference pool；评价 pool 为互斥的 4,000 样品；token pool 固定为 20,000 行。H/I4/X3 的样品—标签比较使用同一个 joint basis。

## 3. Dimension-response map

完整 50×维度矩阵见 `experiment_dimension_matrix.csv`。64/32D 改善最大的候选（尚须逐项检查指标方向和混杂）如下：

| experiment_id | projection_method | metric_value | effective_delta_vs_full | delta_vs_random_projection | status |
| --- | --- | --- | --- | --- | --- |
| F2_single_element_jacobian | PCA32 | 0.4572 | 0.06222 | 0.07964 | Low-rank semantic recovery |
| G4_magmatic_sequences | PCA32 | 0.65 | 0.05 | 0.135 | Projection-sensitive |
| F2_single_element_jacobian | PCA64 | 0.4433 | 0.0483 | 0.04912 | Low-rank semantic recovery |
| F1_lithology_perturbation | PCA32 | 0.3308 | 0.0383 | 0.02549 | Projection-sensitive |
| G5_other_geological_relations | PCA32 | 0.2731 | 0.03599 | 0.05756 | Projection-sensitive |
| B3_numeric_scale_consistency | PCA32 | -0.08245 | 0.02283 | -0.01402 | Projection-sensitive |
| F3_combined_interventions | PCA32 | 0.9647 | 0.02234 | 0.02385 | Preserved under compression |
| F1_lithology_perturbation | PCA64 | 0.3148 | 0.02231 | 0.04038 | Projection-sensitive |
| G2_volcanic_plutonic | PCA32 | 0.4417 | 0.01589 | 0.06115 | Projection-sensitive |
| F3_combined_interventions | PCA64 | 0.9552 | 0.01286 | 0.01282 | Preserved under compression |
| K2_input_mode_stability | PCA32 | 0.9867 | 0.0125 | 0.01816 | Preserved under compression |
| K2_input_mode_stability | PCA64 | 0.9815 | 0.007314 | 0.01249 | Preserved under compression |
| G1_label_neighbors | PCA64 | 0.7828 | 0.006452 | 0.006022 | Preserved under compression |
| I4_label_residual_norms | PCA64 | 0.2296 | 0.005738 | -0.03512 | Preserved under compression |
| G2_volcanic_plutonic | PCA64 | 0.4287 | 0.002872 | 0.003788 | Projection-sensitive |
| X4_random_missingness_followup | PCA32 | 0.3185 | 0.002839 | 0.05396 | Preserved under compression |
| I3_ideal_norm_retrieval | PCA32 | 0.6528 | 0.002778 | 0.04713 | Projection-sensitive |
| X4_random_missingness_followup | PCA64 | 0.3176 | 0.002033 | 0.02644 | Preserved under compression |
| B3_numeric_scale_consistency | PCA64 | -0.1037 | 0.001604 | -0.01584 | Projection-sensitive |
| K1_representation_baselines | PCA64 | 0.4597 | 0.0009333 | 0.0284 | Preserved under compression |

## 4. Low-rank semantic recoveries

| experiment_id | projection_method | metric_value | delta_vs_full | delta_vs_random_projection | status |
| --- | --- | --- | --- | --- | --- |
| F2_single_element_jacobian | PCA64 | 0.4433 | 0.0483 | 0.04912 | Low-rank semantic recovery |
| F2_single_element_jacobian | PCA32 | 0.4572 | 0.06222 | 0.07964 | Low-rank semantic recovery |

只有同时优于 FULL、同维随机投影，且没有同步增强混杂的条目才进入这一节。缺少原实验级 bootstrap 的旧指标不会因为一张好看的图被升级为强证据。

## 5. Visualization-only findings

| experiment_id | projection_method | metric_value | delta_vs_full | delta_vs_random_projection | status |
| --- | --- | --- | --- | --- | --- |
| A2_geochemical_affinity | PCA3 | 0.0003662 | 0.001224 | -0.01552 | Projection artifact suspected |
| A2_geochemical_affinity | PCA2 | 0.0312 | 0.03205 | -0.01657 | Projection artifact suspected |
| B3_numeric_scale_consistency | PCA3 | 0.1059 | 0.2112 | -0.1688 | Projection artifact suspected |
| B3_numeric_scale_consistency | PCA2 | 0.1014 | 0.2067 | -0.1249 | Projection artifact suspected |

## 6. Contextual token results

Contextual token 的原始 C/X 实验已在五个主空间中配对重放；来源控制和角色 proxy 保持原定义。直接样品邻域诊断显示：

| projection_name | geology_metric | source_metric | missingness_metric | metric_value |
| --- | --- | --- | --- | --- |
| FULL | 0.6471 | 0.3634 | 0.6055 | 0.2837 |
| PCA64 | 0.6459 | 0.3632 | 0.6038 | 0.2827 |
| PCA32 | 0.642 | 0.355 | 0.5935 | 0.287 |
| PCA3 | 0.5371 | 0.1885 | 0.4574 | 0.3486 |
| PCA2 | 0.4952 | 0.1556 | 0.3888 | 0.3396 |

这里的 `geology-minus-source` 只作描述性差值，不替代分别报告的三类 purity。

## 7. Geochemical perturbation geometry

F1–F3 已按各主空间与 RP controls 重放。S2 额外重建 F2 的 102 个单元素×背景有限差分方向，报告其 PC 能量与有效秩；方向能量中位数如下：

| direction_family | energy_top2 | energy_top3 | energy_top32 | energy_top64 |
| --- | --- | --- | --- | --- |
| confounder | 0.7094 | 0.8335 | 0.9988 | 0.9996 |
| single_element | 0.0186 | 0.03395 | 0.3438 | 0.5258 |
| source | 0.2524 | 0.312 | 0.9675 | 0.9915 |

连续轨迹或高 additivity 若也被随机投影保留，应解释为连续网络/局部线性的普遍性质，而不是独立地质语义。

## 8. Label relation algebra

G1/G2/G4/G5 在纯标签 projected space 中重放；PCA basis 只使用完整标签词表。volcanic→plutonic、岩浆序列和 alkaline shift 均与 10-seed RP 比较。2D/3D 平行关系不会单独改变状态。

## 9. Process trajectories

E1–E7 的端元、步长、闭合方式和参数顺序均保持不变。报告区分 parameter monotonicity、straightness 与相对于 RP 的额外增益；“轨迹更平滑”不自动等价于“地质更有效”。

## 10. Source and missingness confounding

前 64 PCs 的归因分类如下：

| representation_level | pc_class | PC count |
| --- | --- | --- |
| cls | geology-dominated | 41 |
| cls | mixed | 7 |
| cls | unexplained | 16 |
| contextual_mean | geology-dominated | 47 |
| contextual_mean | mixed | 7 |
| contextual_mean | unexplained | 10 |
| projected_mean | geology-dominated | 43 |
| projected_mean | mixed | 5 |
| projected_mean | source-dominated | 6 |
| projected_mean | unexplained | 10 |

source/reference 使用 eta-squared，连续化学量使用绝对 Spearman；这是可读性归因，不是因果分解。完整结果见 `pc_semantic_attribution.csv`。

## 11. Negative and anomalous findings

| experiment_id | projection_method | metric_value | delta_vs_full | delta_vs_random_projection | status |
| --- | --- | --- | --- | --- | --- |
| A2_geochemical_affinity | PCA32 | -0.03665 | -0.03579 | -0.0699 | High-dimensional dependence |
| C1_same_value_context | PCA3 | 0.5737 | -0.3684 | 0.002493 | High-dimensional dependence |
| C1_same_value_context | PCA2 | 0.4925 | -0.4496 | -0.02411 | High-dimensional dependence |
| C2_element_role_proxies | PCA32 | 0.1521 | -0.06909 | 0.01241 | High-dimensional dependence |
| C2_element_role_proxies | PCA3 | -0.005588 | -0.2268 | 0.08302 | High-dimensional dependence |
| C2_element_role_proxies | PCA2 | -0.01699 | -0.2382 | 0.1064 | High-dimensional dependence |
| C3_contextual_neighbors | PCA32 | 0.1046 | -0.02635 | -0.002749 | High-dimensional dependence |
| C3_contextual_neighbors | PCA3 | 0.0217 | -0.1092 | 0.01828 | High-dimensional dependence |
| C3_contextual_neighbors | PCA2 | 0.002658 | -0.1283 | 0.01149 | High-dimensional dependence |
| D1_rock_continuity | PCA3 | 0.2454 | -0.1543 | 0.1368 | High-dimensional dependence |
| D1_rock_continuity | PCA2 | 0.175 | -0.2247 | 0.09726 | High-dimensional dependence |
| D2_local_geological_neighbors | PCA3 | 0.5052 | -0.1063 | 0.08063 | High-dimensional dependence |
| D2_local_geological_neighbors | PCA2 | 0.461 | -0.1505 | 0.05393 | High-dimensional dependence |
| D3_global_axes | PCA64 | 0.2456 | -0.2311 | -0.1965 | High-dimensional dependence |
| D3_global_axes | PCA32 | 0.3214 | -0.1552 | -0.0963 | High-dimensional dependence |
| D3_global_axes | PCA3 | 0.4197 | -0.05699 | 0.1905 | High-dimensional dependence |
| D3_global_axes | PCA2 | 0.3596 | -0.1171 | 0.1312 | High-dimensional dependence |
| D4_geographic_neighbors | PCA3 | 8361 | 2428 | -210.2 | High-dimensional dependence |
| D4_geographic_neighbors | PCA2 | 8792 | 2859 | -59.56 | High-dimensional dependence |
| D5_external_isotopes_age | PCA64 | -0.022 | -0.2774 | -0.2272 | High-dimensional dependence |
| D5_external_isotopes_age | PCA32 | 0.1677 | -0.08768 | 0.004748 | High-dimensional dependence |
| D5_external_isotopes_age | PCA3 | 0.051 | -0.2044 | 0.06426 | High-dimensional dependence |
| D5_external_isotopes_age | PCA2 | -0.009216 | -0.2646 | 0.02161 | High-dimensional dependence |
| E1_two_endmember_mixing | PCA3 | 0.849 | -0.1123 | 0.1971 | High-dimensional dependence |
| E1_two_endmember_mixing | PCA2 | 0.7154 | -0.2459 | 0.205 | High-dimensional dependence |
| E2_multiendmember_simplex | PCA3 | 0.5997 | -0.3203 | 0.07929 | High-dimensional dependence |
| E2_multiendmember_simplex | PCA2 | 0.5033 | -0.4167 | 0.1638 | High-dimensional dependence |
| E3_magmatic_differentiation | PCA64 | 0.25 | -0.6 | -0.58 | High-dimensional dependence |
| E3_magmatic_differentiation | PCA32 | 0.65 | -0.2 | -0.115 | High-dimensional dependence |
| E3_magmatic_differentiation | PCA3 | 0.7 | -0.15 | 0.2 | High-dimensional dependence |

严格 CIPW 保持 `Failed technically`；I1–I4 的成功部分继续标记为 NNLS normative proxy。缺失 detection-limit、完整 REE、Hf/Ta/Cs 等词表限制不因降维消失。

## 12. Final interpretation

证据驱动分类为：

1. 不支持一个对所有层级统一的 64D/32D 语义子空间；
2. 允许部分局部关系在 64D/32D 被保留或增强，但必须逐项通过 RP 与混杂门槛；
3. 仅在 PCA2/PCA3 中明显的关系被隔离为 visualization-only；
4. source/missingness 占据部分高方差方向，但降维是否改善地质/来源比值取决于 representation 和维度；
5. 静态元素语义的失败主要不是“高维稀疏”一个原因，词表覆盖与训练目标仍是硬限制；
6. contextual token、pooled latent、CLS 与纯标签空间表现出不同的最优有效维度。

## Stability

| representation_level | mean_principal_angle_deg | max_principal_angle_deg |
| --- | --- | --- |
| contextual_mean | 19.05 | 74.02 |
| pure_label_projected | 20.83 | 83.67 |
