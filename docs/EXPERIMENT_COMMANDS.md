# 按实验继续运行

从仓库根目录执行。先 `python scripts/experiment.py new-run E00N`，再将下面每一步放到 `python scripts/experiment.py exec --run-dir /path/to/run --` 后面。所有步骤使用同一运行目录；这里的 `python` 由 `exec` 自动补上。

| 实验 | 步骤，依次执行 |
|---|---|
| E001 基础表示 | `pyrite_benchmark/scripts/benchmark_representations.py`；`pyrite_benchmark/scripts/diagnose_representation_anomalies.py`；`pyrite_benchmark/dimension_sweep/run_dimension_sweep.py` |
| E002 地质表示 | `stage2_geo_representation_xgb/run_stage2.py` |
| E003 黄铁矿模型 | `stage3_element_token_model/src/preprocess.py`；`src/train.py`；`src/extract_embeddings.py`；`src/extract_latent.py --mode label_masked --pooling mean_element`；`src/build_spaces.py`；`src/evaluate_spaces.py`。后五项补全相同的 `stage3_element_token_model/` 前缀。 |
| E004 V2 | `-m stage3_5_element2vec_v2.src.training.train_v2 --config stage3_5_element2vec_v2/configs/v2_base.yaml` |
| E005 空间审计 | `-m stage3_6_element2vec_space_exploration.src.audit_project`；随后同前缀 `extract_cache`、`run_experiments`、`build_reports`、`verify_outputs`。各项使用 `--config stage3_6_element2vec_space_exploration/configs/exploration.yaml`。 |
| E006 子空间复验 | 同 Stage 3.6 模块前缀，依次执行 `subspace_audit`、`run_subspace_replay`、`subspace_specialty`、`build_subspace_report`、`verify_subspace_audit`；配置为 `configs/subspace_audit.yaml`（补全 Stage 3.6 前缀）。`--phase random`、`--phase aux` 的重放另按原协议执行。 |
| E007 RoPE-z | V2 训练模块，配置换成 `stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml`。 |
| E008 RoPE_geo | V2 训练模块，配置换成 `stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml`。 |
| E009 统一评价 | `latent_evaluation_benchmark/run.py prepare`，然后同一入口执行 `extract-real`、`bank`、`extract-bank`、`evaluate`、`render`、`verify`。 |

E006 的 PCA 扫描另建运行并选择配置：`new-run E006 --config stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml`；执行同模块前缀 `run_pca_sweep`、`build_pca_sweep_report`、`verify_pca_sweep`。两类复验的基础输入均来自固定的 E005 历史运行。

E007/E008 的配对审计按照原实验 README、PAIRED_PROTOCOL 执行。新训练的权重需在审计命令中明确传入 `--rope-checkpoint` 或 `--geo-checkpoint`；默认资产仍指向已经冻结的历史权重。`--baseline` 读取 E007 历史 audit，输出写入本次运行的 `audit/`，汇总写入 `reports/`。

V2/RoPE 默认使用冻结的 V2 词表和统计量。若研究问题要求重新拟合词表，应使用新的配置与数据版本，把 `data.vocab_dir` 指向 `run://vocab`，先执行 `-m stage3_5_element2vec_v2.src.data.build_vocab`。不覆盖已发布的词表。

首次数据准备或验证解析规则使用 E001 调试/准备运行，执行 `pyrite_benchmark/scripts/prepare_pyrite_dataset.py`。结果在该运行的 `prepared_data/`，正式采用前登记为新的数据版本。

模型训练不会自动启动，也没有新增训练断点恢复功能。E009 可恢复已经完成的缓存步骤；改变科学配置应新建运行。

## 配对审计的路径写法

`--legacy`、`--v2-checkpoint`、`--sqlite-path` 等输入支持逻辑资产。E007 的准备示例（仍需使用 `exec --run-dir` 前缀）：

```text
-m stage3_6_element2vec_space_exploration.src.value_rope_audit prepare --legacy asset://space_audit --v2-checkpoint asset://v2_checkpoint --rope-checkpoint asset://rope_z_checkpoint --sqlite-path asset://whole_rock --device cpu
```

随后使用相同参数执行 `extract`、`analyze`、`probe`。如需评价本次新训练权重，将 `--rope-checkpoint` 指向当前 run 的 `checkpoints/full_model.pt`。补充控制和汇总仍按原 RoPE 协议执行，输出已转入当前 run。

E008 先调用 `-m stage3_5_element2vec_v2.src.evaluation.prepare_geo_comparison --config stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml --sqlite-path asset://whole_rock` 建立配对协议，再按原协议调用 geo_mask15 的 global、quality、geometry_extract、geometry、numerical、probe/checks/publish。每步共用 run；新模型需显式传入其权重。`geo_mask15_publish` 的训练图读取本次 run 的 logs；只检查历史模型而没有本次训练日志时使用 `--tables-only`，或单独登记引用的训练运行。
