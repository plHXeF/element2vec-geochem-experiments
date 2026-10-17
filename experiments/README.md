# 实验总表

由实验卡生成。运行路径相对工作区；卡片是实验定义和结论的唯一来源。

| 编号 | 实验 | 状态 / 结论 | 报告 |
|---|---|---|---|
| [E001](E001/experiment.json) | 黄铁矿基础表示 | completed / exploratory | [报告](../pyrite_benchmark/results/representation_benchmark/benchmark_report.md) |
| [E002](E002/experiment.json) | 地质先验表示 | completed / mixed | [报告](../stage2_geo_representation_xgb/outputs/stage2_report.md) |
| [E003](E003/experiment.json) | 黄铁矿 token Transformer | completed / exploratory | [报告](../stage3_element_token_model/outputs/stage3_report.md) |
| [E004](E004/experiment.json) | 全岩 Element2Vec V2 | completed / exploratory | [报告](../EXPERIMENT_STATUS.md) |
| [E005](E005/experiment.json) | 表示空间审计 | completed / mixed | [报告](../stage3_6_element2vec_space_exploration/reports/element2vec_exploration_report.md) |
| [E006](E006/experiment.json) | PCA 与子空间复验 | completed / mixed | [报告](../stage3_6_element2vec_space_exploration/subspace_audit/reports/subspace_exploration_report.md) |
| [E007](E007/experiment.json) | Value RoPE-z | completed / mixed | [报告](../stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPORT.md) |
| [E008](E008/experiment.json) | RoPE_geo + 15% mask | completed / mixed | [报告](../stage3_5_element2vec_v2/experiments/rope_geo_mask15/REPORT.md) |
| [E009](E009/experiment.json) | Latent Evaluation Benchmark | completed / mixed | [报告](../reports/E009/REPORT.md) |
| [E010](E010/experiment.json) | 全量地球化学数据、有限角度 RoPE 与嵌入维度扫描 | planned / not_evaluated | [报告](../experiments/E010/README.md) |

## E001 · 黄铁矿基础表示

[实验卡](E001/experiment.json) · [代码](../pyrite_benchmark) · 配置由代码提交和运行种子固定

log/rank 是主要 value-only baseline；mask 与 raw robust 保留为对照。

- [历史运行 legacy-baseline](../../runs/E001/legacy-baseline/summary.md) · [来源与状态](../../runs/E001/legacy-baseline/manifest.json) · [逐文件校验](../../runs/E001/legacy-baseline/artifact_checksums.json)
- [历史运行 legacy-dimension-sweep](../../runs/E001/legacy-dimension-sweep/summary.md) · [来源与状态](../../runs/E001/legacy-dimension-sweep/manifest.json) · [逐文件校验](../../runs/E001/legacy-dimension-sweep/artifact_checksums.json)
- 权重：本实验未训练独立模型；依赖见实验卡。

## E002 · 地质先验表示

[实验卡](E002/experiment.json) · [代码](../stage2_geo_representation_xgb) · [配置](../stage2_geo_representation_xgb/configs/stage2_config.json)

地质先验特征便于解释，尚未整体超过 value baseline。

- [历史运行 legacy-stage2](../../runs/E002/legacy-stage2/summary.md) · [来源与状态](../../runs/E002/legacy-stage2/manifest.json) · [逐文件校验](../../runs/E002/legacy-stage2/artifact_checksums.json)
- 权重：本实验未训练独立模型；依赖见实验卡。

## E003 · 黄铁矿 token Transformer

[实验卡](E003/experiment.json) · [代码](../stage3_element_token_model) · [配置](../stage3_element_token_model/configs/pyrite_token_transformer.yaml)

上下文化 token 有结构；全量预训练后的结果属于探索性检查。

- [历史运行 legacy-stage3](../../runs/E003/legacy-stage3/summary.md) · [来源与状态](../../runs/E003/legacy-stage3/manifest.json) · [逐文件校验](../../runs/E003/legacy-stage3/artifact_checksums.json)
- 权重：[token_transformer_last.pt](../../runs/E003/legacy-stage3/checkpoints/token_transformer_last.pt)

## E004 · 全岩 Element2Vec V2

[实验卡](E004/experiment.json) · [代码](../stage3_5_element2vec_v2) · [配置](../stage3_5_element2vec_v2/configs/v2_base.yaml)

已完成 50 epoch 全量数据训练；不代表外部泛化。

- [历史运行 legacy-v2](../../runs/E004/legacy-v2/summary.md) · [来源与状态](../../runs/E004/legacy-v2/manifest.json) · [逐文件校验](../../runs/E004/legacy-v2/artifact_checksums.json)
- 权重：[full_model.pt](../../runs/E004/legacy-v2/checkpoints/full_model.pt)

## E005 · 表示空间审计

[实验卡](E005/experiment.json) · [代码](../stage3_6_element2vec_space_exploration) · [配置](../stage3_6_element2vec_space_exploration/configs/exploration.yaml)

保留 50 个正、负、模糊及不可用子实验；来源与缺失结构显著。

- [历史运行 legacy-space-audit](../../runs/E005/legacy-space-audit/summary.md) · [来源与状态](../../runs/E005/legacy-space-audit/manifest.json) · [逐文件校验](../../runs/E005/legacy-space-audit/artifact_checksums.json)
- 权重：本实验未训练独立模型；依赖见实验卡。

## E006 · PCA 与子空间复验

[实验卡](E006/experiment.json) · [代码](../stage3_6_element2vec_space_exploration) · [配置](../stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml)

严格低秩恢复主要见于 F2 的 PCA32/PCA64；不支持通用低维语义空间。

- [历史运行 legacy-pca-sweep](../../runs/E006/legacy-pca-sweep/summary.md) · [来源与状态](../../runs/E006/legacy-pca-sweep/manifest.json) · [逐文件校验](../../runs/E006/legacy-pca-sweep/artifact_checksums.json)
- [历史运行 legacy-subspace-audit](../../runs/E006/legacy-subspace-audit/summary.md) · [来源与状态](../../runs/E006/legacy-subspace-audit/manifest.json) · [逐文件校验](../../runs/E006/legacy-subspace-audit/artifact_checksums.json)
- 权重：本实验未训练独立模型；依赖见实验卡。

## E007 · Value RoPE-z

[实验卡](E007/experiment.json) · [代码](../stage3_5_element2vec_v2) · [配置](../stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml)

局部切向稳定性改善，整体岩石邻域和来源混杂变差。

- [历史运行 legacy-rope-z](../../runs/E007/legacy-rope-z/summary.md) · [来源与状态](../../runs/E007/legacy-rope-z/manifest.json) · [逐文件校验](../../runs/E007/legacy-rope-z/artifact_checksums.json)
- 权重：[full_model.pt](../../runs/E007/legacy-rope-z/checkpoints/full_model.pt) · [full_model.pt](../../runs/E007/legacy-rope-z/smoke/checkpoints/full_model.pt)

## E008 · RoPE_geo + 15% mask

[实验卡](E008/experiment.json) · [代码](../stage3_5_element2vec_v2) · [配置](../stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml)

部分小步长局部几何改善；整体岩石邻域下降，来源聚集增强。

- [历史运行 legacy-geo-mask15](../../runs/E008/legacy-geo-mask15/summary.md) · [来源与状态](../../runs/E008/legacy-geo-mask15/manifest.json) · [逐文件校验](../../runs/E008/legacy-geo-mask15/artifact_checksums.json)
- 权重：[full_model.pt](../../runs/E008/legacy-geo-mask15/checkpoints/full_model.pt) · [full_model.pt](../../runs/E008/legacy-geo-mask15/smoke/checkpoints/full_model.pt)

## E009 · Latent Evaluation Benchmark

[实验卡](E009/experiment.json) · [代码](../latent_evaluation_benchmark) · [配置](../latent_evaluation_benchmark/configs/benchmark.json)

比较可解码性、信息充分性与过程几何；不使用加权总分。

- [历史运行 legacy-latent-v01](../../runs/E009/legacy-latent-v01/summary.md) · [来源与状态](../../runs/E009/legacy-latent-v01/manifest.json) · [逐文件校验](../../runs/E009/legacy-latent-v01/artifact_checksums.json)
- 权重：本实验未训练独立模型；依赖见实验卡。

## E010 · 全量地球化学数据、有限角度 RoPE 与嵌入维度扫描

[实验卡](E010/experiment.json) · [代码](../stage3_5_element2vec_v2) · [配置](../stage3_5_element2vec_v2/configs/e010/prepare.json)

原实现与独立验收通过；本机CPU全量985996条输入已准备、核验并冻结，登记为新的准备版本，保留与历史float32角度合并数量的差异。Slurm/RTX A6000×4入口仅执行后12配置、每卡独立训练，总计16核CPU、35G主机内存起试，登记在计算节点作业内完成。尚未启动真实训练，科学结论未评价。

- [历史运行 20261007T065305Z-s42-79ed84](../../runs/E010/20261007T065305Z-s42-79ed84/summary.md) · [来源与状态](../../runs/E010/20261007T065305Z-s42-79ed84/manifest.json) · [逐文件校验](../../runs/E010/20261007T065305Z-s42-79ed84/artifact_checksums.json)
- [历史运行 20261007T135913Z-s42-773907](../../runs/E010/20261007T135913Z-s42-773907/summary.md) · [来源与状态](../../runs/E010/20261007T135913Z-s42-773907/manifest.json) · [逐文件校验](../../runs/E010/20261007T135913Z-s42-773907/artifact_checksums.json)
- [历史运行 20261007T140915Z-s42-d6c104](../../runs/E010/20261007T140915Z-s42-d6c104/summary.md) · [来源与状态](../../runs/E010/20261007T140915Z-s42-d6c104/manifest.json) · [逐文件校验](../../runs/E010/20261007T140915Z-s42-d6c104/artifact_checksums.json)
- [历史运行 20261007T141044Z-s42-4534fe](../../runs/E010/20261007T141044Z-s42-4534fe/summary.md) · [来源与状态](../../runs/E010/20261007T141044Z-s42-4534fe/manifest.json) · [逐文件校验](../../runs/E010/20261007T141044Z-s42-4534fe/artifact_checksums.json)
- [历史运行 20261007T141158Z-s42-520611](../../runs/E010/20261007T141158Z-s42-520611/summary.md) · [来源与状态](../../runs/E010/20261007T141158Z-s42-520611/manifest.json) · [逐文件校验](../../runs/E010/20261007T141158Z-s42-520611/artifact_checksums.json)
- [历史运行 20261017T010750Z-s42-793cbd](../../runs/E010/20261017T010750Z-s42-793cbd/summary.md) · [来源与状态](../../runs/E010/20261017T010750Z-s42-793cbd/manifest.json) · [逐文件校验](../../runs/E010/20261017T010750Z-s42-793cbd/artifact_checksums.json)
- [历史运行 20261017T012647Z-s42-1c7c3f](../../runs/E010/20261017T012647Z-s42-1c7c3f/summary.md) · [来源与状态](../../runs/E010/20261017T012647Z-s42-1c7c3f/manifest.json) · [逐文件校验](../../runs/E010/20261017T012647Z-s42-1c7c3f/artifact_checksums.json)
- [历史运行 20261017T013023Z-s42-1bb3ea](../../runs/E010/20261017T013023Z-s42-1bb3ea/summary.md) · [来源与状态](../../runs/E010/20261017T013023Z-s42-1bb3ea/manifest.json) · [逐文件校验](../../runs/E010/20261017T013023Z-s42-1bb3ea/artifact_checksums.json)
- [历史运行 20261017T013114Z-s42-096a4b](../../runs/E010/20261017T013114Z-s42-096a4b/summary.md) · [来源与状态](../../runs/E010/20261017T013114Z-s42-096a4b/manifest.json) · [逐文件校验](../../runs/E010/20261017T013114Z-s42-096a4b/artifact_checksums.json)
- [历史运行 20261017T013210Z-s42-5a5264](../../runs/E010/20261017T013210Z-s42-5a5264/summary.md) · [来源与状态](../../runs/E010/20261017T013210Z-s42-5a5264/manifest.json) · [逐文件校验](../../runs/E010/20261017T013210Z-s42-5a5264/artifact_checksums.json)
- [历史运行 20261017T013747Z-s42-0a349d](../../runs/E010/20261017T013747Z-s42-0a349d/summary.md) · [来源与状态](../../runs/E010/20261017T013747Z-s42-0a349d/manifest.json) · [逐文件校验](../../runs/E010/20261017T013747Z-s42-0a349d/artifact_checksums.json)
- 权重：本实验未训练独立模型；依赖见实验卡。
