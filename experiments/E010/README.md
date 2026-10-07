# E010 实现与执行入口

正式训练尚未运行，科学结论 `not_evaluated`。协议见 [PROTOCOL.md](PROTOCOL.md)。实现集中在 `stage3_5_element2vec_v2/src/e010`，沿用现有 RoPE encoder 与工作区运行管理。

覆盖基准：985,996 条含任一化学观测的记录、23,629,373 个原始非空格。词表 118 元素 + 25 氧化物/辅助量。16 配置为 fixed/native 两组各 8 个宽度。fixed 固定 768 主干、共享线性 projector；总参数仍有 embedding/projector 的已列示差异。

## 输入准备

所有命令在仓库根目录，用工作区登记 Python 运行。正式执行前工作树须干净。`asset://e010_prepared` 应指向已完成核验并冻结的版本。

```bash
python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/prepare.json
# 将上一步输出填入 PREP_RUN
python scripts/experiment.py exec --run-dir "$PREP_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli prepare
python scripts/experiment.py exec --run-dir "$PREP_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli verify-data
python scripts/experiment.py exec --run-dir "$PREP_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli matrix
```

准备数据不能覆盖已有目录；修改转换规则必须建新 run。冻结核验通过的准备 run 后，在本机 workspace.local.json 注册资产路径。不得提交样本级准备物。

## 使用者启动短消融和正式扫描

下列 `run` 命令会实际训练；实现验收期间不执行。

```bash
PILOT_RUN=$(python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/ablations.json)
python scripts/experiment.py exec --run-dir "$PILOT_RUN" -- scripts/e010_scan.py run --config stage3_5_element2vec_v2/configs/e010/ablations.json

SCAN_RUN=$(python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/scan.json)
python scripts/experiment.py exec --run-dir "$SCAN_RUN" -- scripts/e010_scan.py run --config stage3_5_element2vec_v2/configs/e010/scan.json
```

两块由使用者分别决定启动。扫描器为每配置创建独立正式 run，依次 preflight→train→verify-training→freeze，任何失败停止，不启动下一项。预览只打印计划：

```bash
python scripts/e010_scan.py plan
```

可恢复的训练中断（父目录仍未冻结）执行：

```bash
python scripts/experiment.py exec --run-dir "$SCAN_RUN" -- scripts/e010_scan.py resume --config stage3_5_element2vec_v2/configs/e010/scan.json
```

扫描恢复不会跳过失败的非训练步骤；需要新 run 的故障保留原证据。冻结时仅 E010 经严格 checkpoint/全量覆盖验证的成功 resume 可使用 `--allow-verified-resume`，不会清除失败历史。

## 后续评价

显式填写同一准备版本的 checkpoint 路径，不用 latest 导航别名。导出和评价共用一个新分析 run，chemistry-only：

```bash
EVAL_RUN=$(python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/analysis.json)
python scripts/experiment.py exec --run-dir "$EVAL_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli extract --config stage3_5_element2vec_v2/configs/e010/analysis.json --checkpoint "$CHECKPOINT" --device mps
python scripts/experiment.py exec --run-dir "$EVAL_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli evaluate --config stage3_5_element2vec_v2/configs/e010/analysis.json --device mps
python scripts/experiment.py check --run-dir "$EVAL_RUN" --rehash
```

评价要记录预训练暴露、固定/原生容量差异、来源/文献/缺失混杂、缺失元素未训练和数值角度有限精度。输出接口齐全不等于方法已被真实训练验证。
