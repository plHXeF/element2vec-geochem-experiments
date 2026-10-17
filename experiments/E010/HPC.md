# E010：Slurm / NVIDIA A800 × 4

2026-10-17 实现说明。仅运行原 `scan.json` 顺序中的第 5–16 项，跳过 fixed 的 8/16/32/64D。用户选择四个独立配置并行，每配置一张 GPU。没有 DDP、批量放大、混合精度或学习率缩放；原 50 epochs、effective batch 1024、micro batch 32、seed 42、损失、probe 和保存协议保持不变。原 MPS 配置与历史报告保留。科学结果仍为 `not_evaluated`。

| Slurm task / GPU 队列 | 依次执行的配置 |
| --- | --- |
| 0 | native1024 → native8 → native16 |
| 1 | fixed1024 → fixed128 → native32 |
| 2 | fixed512 → native512 → native64 |
| 3 | fixed256 → native256 → native128 |

队列是按容量安排的静态分组，不保证同一时间结束。一张卡空闲后不会重复启动其他队列。`scripts/e010_slurm.py plan` 不依赖数据库或 PyTorch，不创建运行；会核对 12 项恰好覆盖原计划后 12 项、无重复，CUDA 配置与对应原配置仅设备字段不同。

## 当前停止点

本次只完成代码修改和本地检查，没有提交 Slurm 作业、进行真实数据预检或训练。当前 clone 无数据库及 `e010_prepared`。本机已建立被 Git 忽略的 `workspace.local.json`，工作区为本仓库目录；数据库可拷贝到：

```text
<workspace>/data/whole_rock/whole-rock-v1/geochem.db
```

仅拷贝数据库仍需显式执行一次 E010 准备，或者另外拷贝原已冻结的准备运行。旧卡片记录的 985,996 行和原准备 run 是历史证据，不能据此声称新机器已有输入。原库 SHA-256 见 `ACCEPTANCE.md`；输入变更需独立记录，不能放宽覆盖检查去迎合结果。

## HPC 环境与路径

在 HPC 建立 Python 3.11 环境，安装 `environment.yml` 的依赖及与集群驱动兼容的 CUDA PyTorch。原示例脚本的 `sae-fsdet` 环境属于另一项目，不应直接复用。作业脚本沿用该示例的 `nvidia_gpu` 分区和 `apps/anaconda3/2021.05` 模块，默认环境名 `element2vec-geochem`，可用 `E010_CONDA_ENV` 指定。它申请一节点、4 tasks、每 task 8 CPU、4 GPU、总内存 140G；集群如需 account、walltime 或 GPU 类型名，按管理员实际配置给 `sbatch` 加参数。没有猜测 A800 的 GRES 类型别名；worker 会验证实际设备为 A800。

HPC 工作区建议与仓库分开。以下变量填本机实际路径，不将绝对路径提交 Git：

```bash
export GEOCHEM_WORKSPACE="$HOME/element2vec-workspace"
mkdir -p "$GEOCHEM_WORKSPACE/data/whole_rock/whole-rock-v1"
cp docs/workspace.e010.example.json "$GEOCHEM_WORKSPACE/workspace.local.json"
# 将私有 geochem.db 拷贝到上述 whole-rock-v1/；不要覆盖已有冻结版本。
# 激活 Python 3.11 / CUDA PyTorch 环境；所有后续命令在仓库根目录执行。
python scripts/experiment.py list
python scripts/e010_slurm.py plan
```

`workspace.local.json` 模板的 `e010_prepared` 是待配置位置，不是现有数据。两个准备方式二选一：

1. 推荐迁移原已冻结准备 run `20261007T135913Z-s42-773907` 的整个目录，放入本工作区 `runs/E010/` 下；将本地 `e010_prepared` 映射设为该 run 的 `prepared_data`。保留原 manifest 和校验，不补写当前提交。使用 `experiment.py check --run-dir ... --rehash` 检查；检查还要求其原始 `whole_rock` 输入能在迁移后的工作区按登记位置找到。
2. 只迁移数据库时，用以下显式步骤新建准备 run。该过程可能耗时，勿在不允许计算的登录节点执行；可放入 CPU 作业。不会自动启动训练。

```bash
PREP_RUN=$(python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/prepare.json)
python scripts/experiment.py exec --run-dir "$PREP_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli prepare
python scripts/experiment.py exec --run-dir "$PREP_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli verify-data
python scripts/experiment.py exec --run-dir "$PREP_RUN" -- -m stage3_5_element2vec_v2.src.e010.cli matrix
python scripts/experiment.py check --run-dir "$PREP_RUN" --rehash
python scripts/experiment.py freeze --run-dir "$PREP_RUN" --status completed --outcome not_evaluated \
  --summary 'E010 full-coverage preparation and schema/coverage checks passed; no model training.' \
  --algorithms 'E010 immutable full-coverage preparation; frozen robust statistics' --device cpu
```

最后将本地 `workspace.local.json` 的 `e010_prepared` 改为新冻结 run 的 `prepared_data` 路径（相对工作区），而非直接使用模板里的待配置位置。准备和所有训练必须使用干净的已提交代码；不自动下载旧 checkpoint，新训练从初始化开始。

## 显式提交后 12 次训练

确认数据库与准备资产就绪后才执行。创建一个协调 run，提交时将调度器日志也放进该 run。调度器负责 GPU 隔离：一个 srun step 启动四 tasks，使用 `--gpus-per-task=1 --gpu-bind=single:1`，沿用 Slurm 提供的 `CUDA_VISIBLE_DEVICES`，代码不自行按 rank 改写设备编号。参见 [Slurm GRES 文档](https://slurm.schedmd.com/gres.html) 和 [srun 文档](https://slurm.schedmd.com/srun.html)。需要集群启用相应 GPU/task 绑定支持。

```bash
LAUNCH_RUN=$(python scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/hpc/scan_last12.json)
sbatch --output="$LAUNCH_RUN/logs/batch-%j.out" --error="$LAUNCH_RUN/logs/batch-%j.err" \
  scripts/slurm/e010_a800.sh "$LAUNCH_RUN"
```

协调器先创建四个独立队列 run，再让每个 worker 在自己的队列中调用原扫描器。每项有独立训练 run，依次执行 preflight（无优化器更新）→ train → verify-training → check --rehash → freeze。设备正确记录为 CUDA；CUDA 预检会同步计时并记录设备及峰值显存。任何 task 失败，`srun --kill-on-bad-exit=1` 停止同作业其他 task，不继续发起新训练。已完成的训练 run 保持冻结；不会进行后续评价、额外 seed 或短消融。

`slurm_registry.json` 只由协调器写入，四个 worker 只读它，各写自己的队列 registry。训练参数、数据指纹和代码版本由运行工具固定，输入映射漂移会被拒绝。单项日志在其训练 run 的 `logs/`，各 task 输出在协调 run 的 `logs/slurm-*-slot-*.log`。结束后先检查所有训练 run；协调器的成功 receipt 是 `slurm_complete.json`。科学优劣需要另做评价，训练完成不表示科学正结果。

## 恢复中断

保留原协调 run 和四个队列 run。使用相同干净提交、同一准备资产、相同 CUDA backend / PyTorch 版本和相同 GPU 分配方式重新提交：

```bash
sbatch --output="$LAUNCH_RUN/logs/batch-%j.out" --error="$LAUNCH_RUN/logs/batch-%j.err" \
  scripts/slurm/e010_a800.sh "$LAUNCH_RUN" resume
```

从原 `slurm_registry.json` 定位队列和训练 run；已完成项跳过，存在 durable `latest.pt` 的训练以 `--resume` 继续。冻结输入不能变。失败历史保留。原流程仅支持训练步骤的严格恢复：失败发生在 preflight/verify 等非训练步骤，或训练尚无第一个 durable checkpoint 时，需保留旧证据并创建新 run，不能删除日志冒充恢复。配置更改也需新 run。

Slurm 强制终止可能留下 `.execution.lock` 或未写回的命令状态。必须先用调度器确认旧作业及所有子进程已停止，再检查对应锁与运行证据，必要时由维护者处理残留锁；脚本不会自动删除锁。协调/队列 run 暂保持可写以支持恢复；训练子 run 验证后冻结。父 run 有失败调度历史时不适用仅针对训练子 run 的 `--allow-verified-resume`，不要强行冻结为成功；记录调度中断及所有已冻结子 run 后单独收尾。

## 验证边界

本次在本机 Python 3.12 上完成 13 项新增无数据调度测试、14 项原运行管理测试，全部通过；Python 编译检查、Bash 语法检查和公开树审计通过。测试虚拟环境只补充 PyYAML，留在 Git 忽略目录，没有安装训练环境或生成研究 run。

本机未提供 Slurm、A800、Python 3.11/PyTorch 科学计算环境或私有数据库，因此未运行 Stage 3.5 的 PyTorch 测试或 CUDA 实测。这些管理逻辑检查不代表已经通过 HPC 实机验收。拷贝数据库和准备输入后，首次真实执行仍须通过每个配置的 CUDA preflight。
