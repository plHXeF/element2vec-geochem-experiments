# E010 实现与执行入口

实现与 gpt-6.1-sol 独立验收通过。真实数据训练尚未运行，科学结论 `not_evaluated`。协议见 [PROTOCOL.md](PROTOCOL.md)。实现集中在 `stage3_5_element2vec_v2/src/e010`，沿用现有 RoPE encoder 与工作区运行管理。

覆盖基准：985,996 条含任一化学观测的记录、23,629,373 个原始非空格。词表 118 元素 + 25 氧化物/辅助量。16 配置为 fixed/native 两组各 8 个宽度。fixed 固定 768 主干、共享线性 projector；总参数仍有 embedding/projector 的已列示差异。

## 已完成的验收

实现提交 `a840190`：Stage 3.5 共54项通过；工作区管理14项、Stage 3.6 14项、E009基础11项通过。全量数据与16架构参数矩阵核验完成；native1024、fixed1024→768及native8三个正式MPS预检都通过，权重不变、梯度有限、优化器更新为0。历史AppleDouble复验和数值库线程限制见 [验收记录](ACCEPTANCE.md)。

数据准备 run `20261007T135913Z-s42-773907` 已冻结，注册为 `asset://e010_prepared`。覆盖、有限精度和精确参数数目见 [覆盖与参数矩阵](COVERAGE_AND_MATRIX.md)。40个元素身份没有观测；保留嵌入槽位但不宣称已训练。

完整50epoch矩阵是长期计算：native1024最大面板短测1.4594秒/32条，按这个密集面板速度外推约12.5小时/epoch；实际有稀疏面板与优化/存储开销差异，不能视为准确完工时间。短消融和主扫描分别启动，不自动从短消融跳到主扫描。

建议本机测试/评价保留 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`。

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
