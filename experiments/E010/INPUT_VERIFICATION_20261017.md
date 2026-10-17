# E010 新 clone 数据库拷贝核验

日期：2026-10-17。来源运行：`20261017T010750Z-s42-793cbd`，代码提交 `e4ed473`。运行已通过 `check --rehash`，冻结为 `completed / not_evaluated`。本文件仅包含聚合输入检查，不含样本记录。

| 检查 | 结果 |
| --- | --- |
| 数据库大小 | 707,801,088 bytes |
| SHA-256 | `ff84ce42a5824b6422ade689d280a0c0585ab8429eaf4b931911f0d7557ebf14` |
| 与历史验收数据库一致 | 是 |
| sample 总数 | 1,022,092 |
| E010 全覆盖查询行数 | 985,996 |
| 唯一 sample_id 数 | 985,996 |
| 原始非空观测格数 | 23,629,373 |
| 原始库访问方式 | SQLite `mode=ro`，`query_only=ON` |
| 优化器更新 | 0 |

检查直接使用当前提交中 E010 的原始字段定义和 `prepare.source_query`，没有替换成旧高覆盖视图，没有调整行筛选、连接或预期数量。哈希与 `ACCEPTANCE.md` 的原库相同，覆盖数也完全一致。完整本地检查记录保存在本次 run 的 `reports/database_verification.json`，原库保持只读。

本次不是数据准备运行，未生成 `prepared_data`，不得将它登记为 `asset://e010_prepared`。本机现有 Python 3.12 检查环境缺少 PyTorch 等科学计算依赖；没有执行准备、参数矩阵、CUDA 预检、Slurm 作业或训练。历史准备 run 的身份保持不变，不用本次核验 run 替换它。

下一步是在 Python 3.11 科学计算环境中迁移原冻结准备资产，或使用已验证数据库新建准备 run、执行 prepare / verify-data / matrix、核验冻结，再在本地设置中登记该准备资产。命令见 [HPC 说明](HPC.md)。正式训练仍限定为后 12 个配置、四个独立单卡队列。
