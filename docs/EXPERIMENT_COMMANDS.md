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

E001–E008 历史训练入口不会自动启动，也没有新增训练断点恢复功能。E009 可恢复已经完成的缓存步骤；E010 另有严格校验的训练恢复入口，见下文。改变科学配置应新建运行。

## 配对审计的路径写法

`--legacy`、`--v2-checkpoint`、`--sqlite-path` 等输入支持逻辑资产。E007 的准备示例（仍需使用 `exec --run-dir` 前缀）：

```text
-m stage3_6_element2vec_space_exploration.src.value_rope_audit prepare --legacy asset://space_audit --v2-checkpoint asset://v2_checkpoint --rope-checkpoint asset://rope_z_checkpoint --sqlite-path asset://whole_rock --device cpu
```

随后使用相同参数执行 `extract`、`analyze`、`probe`。如需评价本次新训练权重，将 `--rope-checkpoint` 指向当前 run 的 `checkpoints/full_model.pt`。补充控制和汇总仍按原 RoPE 协议执行，输出已转入当前 run。

E008 先调用 `-m stage3_5_element2vec_v2.src.evaluation.prepare_geo_comparison --config stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml --sqlite-path asset://whole_rock` 建立配对协议，再按原协议调用 geo_mask15 的 global、quality、geometry_extract、geometry、numerical、probe/checks/publish。每步共用 run；新模型需显式传入其权重。`geo_mask15_publish` 的训练图读取本次 run 的 logs；只检查历史模型而没有本次训练日志时使用 `--tables-only`，或单独登记引用的训练运行。

## E010 全量 RoPE 维度扫描

[实验协议](../experiments/E010/PROTOCOL.md)、[已完成验收与执行命令](../experiments/E010/README.md)。全量输入 `asset://e010_prepared` 已冻结；含化学观测985,996条，143个字段身份。

`python scripts/e010_scan.py plan` 只预览16个配置，不创建run或训练。使用者显式创建父run，再通过 `experiment.py exec` 调用扫描器 `run`；短消融配置 `stage3_5_element2vec_v2/configs/e010/ablations.json`，完整扫描配置 `stage3_5_element2vec_v2/configs/e010/scan.json`。两个入口独立，不自动连跑。

各子run执行 `stage3_5_element2vec_v2.src.e010.cli` 的 `preflight`、`train`、`verify-training`。预检无优化器更新。训练断点含优化器、随机状态、数据/配置指纹和完整遍历游标，`resume` 只接受原配置和原输入；支持的成功恢复必须保留失败历史并经过验证后才能冻结。已冻结run不可恢复。

评价使用同一模块的 `extract --checkpoint ...` 和 `evaluate`，配置 `analysis.json`。所有样本级缓存及权重均只写私有run。E010实现交付不包含任何真实数据训练。

Slurm / NVIDIA A800×4 的后12项入口、数据迁移和显式提交/恢复步骤见 [E010 HPC 说明](../experiments/E010/HPC.md)。`python scripts/e010_slurm.py plan` 只预览：每卡一配置、四队列并行，不是 DDP。新 CUDA 配置保留原科学参数；不会启动原扫描前4项、短消融或额外种子。当前 clone 尚缺私有输入，修改完成后等待使用者拷贝数据库。
