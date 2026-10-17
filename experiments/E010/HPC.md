# E010：Slurm / NVIDIA RTX A6000 × 4

2026-10-17 实现说明。仅运行原 `scan.json` 顺序中的第 5–16 项，跳过 fixed 的 8/16/32/64D。用户选择四个独立配置并行，每配置一张 GPU。没有 DDP、批量放大、混合精度或学习率缩放；原 50 epochs、effective batch 1024、micro batch 32、seed 42、损失、probe 和保存协议保持不变。原 MPS 配置与历史报告保留。科学结果仍为 `not_evaluated`。

| Slurm task / GPU 队列 | 依次执行的配置 |
| --- | --- |
| 0 | native1024 → native8 → native16 |
| 1 | fixed1024 → fixed128 → native32 |
| 2 | fixed512 → native512 → native64 |
| 3 | fixed256 → native256 → native128 |

队列是按容量安排的静态分组，不保证同一时间结束。一张卡空闲后不会重复启动其他队列。`scripts/e010_slurm.py plan` 不依赖数据库或 PyTorch，不创建运行；会核对 12 项恰好覆盖原计划后 12 项、无重复，CUDA 配置与对应原配置仅设备字段不同。

## 初始交付时的停止点

本次只完成代码修改和本地检查，没有提交 Slurm 作业、进行真实数据预检或训练。当前 clone 无数据库及 `e010_prepared`。本机已建立被 Git 忽略的 `workspace.local.json`，工作区为本仓库目录；数据库可拷贝到：

```text
<workspace>/data/whole_rock/whole-rock-v1/geochem.db
```

仅拷贝数据库仍需显式执行一次 E010 准备，或者另外拷贝原已冻结的准备运行。旧卡片记录的 985,996 行和原准备 run 是历史证据，不能据此声称新机器已有输入。原库 SHA-256 见 `ACCEPTANCE.md`；输入变更需独立记录，不能放宽覆盖检查去迎合结果。

后续更新（同日）：使用者已拷贝数据库，新 clone 的只读核验通过，SHA-256、985,996 条唯一训练记录和 23,629,373 个原始观测均与历史一致。见 [本次输入核验](INPUT_VERIFICATION_20261017.md)。本机环境由 [uv CPU 环境](../../environments/local-cpu/README.md) 管理；环境验收后，已另行完成本地准备并冻结为 `20261017T013747Z-s42-0a349d`。当前 `asset://e010_prepared` 指向此新 run，迁移时复制整个冻结目录；聚合核验和与历史的数值差异见 [本地准备报告](LOCAL_PREPARATION_20261017.md)。下文准备命令用于将来需要新版本时，不必为本次迁移重复生成。

**集群硬约束：文件管理节点禁止任何计算。** 以下所有 Python 命令、数据准备和测试在本机执行。HPC 端只在分配到的计算节点作业内执行环境检查、运行登记、输入哈希和训练；文件管理节点只承担集群允许的文件传输、配置文件编辑与作业提交。

## HPC 环境与路径

### 使用规范核实与作业265409故障

2026-10-17 已核对 [Slurm sbatch 官方文档](https://slurm.schedmd.com/sbatch.html)、学校信息网络与数据中心的 [Conda安装帮助](https://nic.cugb.edu.cn/c/2024-01-24/797977.shtml)、[Module帮助](https://nic.cugb.edu.cn/c/2024-01-23/797941.shtml) 和 [GPU环境搭建教程](https://nic.cugb.edu.cn/c/2024-01-23/797881.shtml)。学校GPU教程发布于2024-01-23，其示例先用 `salloc` 申请1张GPU，再进入分配节点安装环境，因此在计算节点安装有官方示例支持；该示例不是要求所有安装都必须占GPU，更没有要求占4卡。通用Conda帮助未明确安装节点范围，不能据此断言登录节点允许或禁止安装。CPU安装作业是节约GPU分配时间的实现选择，不是学校明文要求。

Slurm的 `sbatch` 先提交作业，获分配后在计算节点运行一份批处理脚本；四个训练进程由后续 `srun` 启动。`--ntasks=4` 和 `--cpus-per-task=4` 申请总计16个CPU，`--mem=35G` 是一节点作业总主机内存，不是显存或每task内存。35G是否足够仍需实测。Slurm文档不代替本校节点使用规定。

用户提供的265409输出显示依赖安装成功且 `pip check` 通过，随后在激活系统Anaconda环境时因 `PS1: 未绑定的变量` 退出；所附日志未进入输入核验、运行登记或训练。根因是脚本 `set -u` 与旧版Conda非交互激活代码的兼容性。三个作业入口在加载Conda前显式初始化并导出 `PS1`，保留其他严格错误检查。模拟安装、复用和嵌套GPU启动测试覆盖未设置PS1的场景；尚未在HPC复测。

这次失败不需要重新下载依赖。旧脚本在激活前已完成安装清单和环境ready标记，修正后可直接复用 `.hpc/e010-cu118-v1/`，使用 `sbatch scripts/slurm/e010_a6000_start.sh new` 重新提交GPU检查及训练；不是训练断点恢复。保留原日志和环境，不删除重建。

### 首次安装环境、配置工作区并启动训练

数据库及完整准备 run 按下列结构上传完成后，在仓库根目录提交：

```text
data/whole_rock/whole-rock-v1/geochem.db
runs/E010/20261017T013747Z-s42-0a349d/  # 完整冻结目录
```

```bash
bash scripts/submit_e010.sh new
```

此入口默认工作区为仓库根目录，只调用两次 `sbatch`，不在提交节点运行 Python、conda、哈希或数据检查。第一项在 `cpu_compute` 申请4核CPU、16G内存，**不申请GPU**，安装专用 Python 3.11 / torch 2.5.1+cu118 环境，逐文件核验上传的冻结运行和数据库，兼容 Windows 校验清单中的反斜杠路径，再写入被 Git 忽略的 `workspace.local.json`。原冻结文件完全不改写。若已有配置中的两个输入资产指向别处则停止，避免静默改绑输入。

第二项用 Slurm `afterok` 依赖等待第一项成功，才会申请4张 RTX A6000、16核CPU、35G主机内存；若安装失败，依赖作业取消，不占GPU安装软件。GPU入口只激活已装环境、做四卡 CUDA 前向/反向检查，再调用下文的 `e010_a6000.sh new` 开始后12项训练。环境位于 `<workspace>/.hpc/e010-cu118-v1/`，后续复用，不自动升级；安装清单保留在环境内。首次安装需要CPU计算节点可访问 conda-forge、PyPI 和 PyTorch 下载站，且环境路径须由CPU/GPU节点共享。安装会消耗CPU作业时间，不申请GPU；网络受限时需使用平台允许的离线安装流程。参见 [Slurm依赖说明](https://slurm.schedmd.com/sbatch.html#OPT_dependency) 和 [Linux CUDA 环境](../../environments/hpc-cu118/README.md)。

若数据位于独立共享工作区，可在提交前设置 `GEOCHEM_WORKSPACE`；否则无需设置。所有计算均在作业内。CPU安装日志为 `slurm-e010-env-<jobid>.out/err`，GPU日志为 `slurm-e010-<jobid>.out/err`，协调 run 路径在GPU输出中。环境已成功安装时，可直接提交 `sbatch scripts/slurm/e010_a6000_start.sh new`，不再排队安装；恢复原运行时使用同一已安装环境：

```bash
sbatch scripts/slurm/e010_a6000_start.sh resume "$LAUNCH_RUN"
```

本次本地通过28项调度/安装模拟/迁移核验测试；模拟不实际下载或安装 Linux 环境，也不代表 Slurm/CUDA 实机验收。用户已提交旧版作业265409，更新脚本不会改变Slurm保存的旧作业脚本，不应在活跃训练期间更新源码。旧版若已在安装，等待本次完成并复用环境即可；此修正取消后续在GPU作业内安装软件的路径。本机未连接HPC，无法确认该作业当前状态。

### 已有环境时的手工入口

在 HPC 建立 Python 3.11 环境，安装 `environment.yml` 的依赖及与集群驱动兼容、支持 RTX A6000 的 CUDA PyTorch；本机 Windows CPU 环境不迁移。原示例脚本的 `sae-fsdet` 环境属于另一项目，不应直接复用。作业脚本改用平台显示的 RTX A6000 分区 `nvidia_gpu2`，保留 `apps/anaconda3/2021.05` 模块，默认环境名 `element2vec-geochem`，可用 `E010_CONDA_ENV` 指定。

当前申请一节点、4 tasks、每 task 4 CPU（总计 16 核）、4 GPU、整节点作业合计 `--mem=35G` 主机内存。35G 是首次尝试的申请值，尚未测得四训练进程的实际内存峰值；它不是每卡 35G，也不是显存。初版 140G 只是原示例单卡 35G 的四倍估算，已撤销。首次运行后在集群允许的入口查看作业 OOM 状态和内存统计，再决定是否调整申请；不会自动改变训练批量或协议。

集群如需 account、walltime 或 GPU 类型名，按管理员实际配置给 `sbatch` 加参数。没有猜测 RTX A6000 的 GRES 类型别名；worker 会验证实际设备为 RTX A6000、每 task 仅见一张卡，并在队列 run 的 `slurm_device.json` 及协调 run 的成功 receipt 中记录实际设备。驱动、CUDA 环境与显存余量仍须在计算节点实测。

HPC 工作区建议与仓库分开。以下变量填本机实际路径，不将绝对路径提交 Git：

```bash
export GEOCHEM_WORKSPACE="$HOME/element2vec-workspace"
mkdir -p "$GEOCHEM_WORKSPACE/data/whole_rock/whole-rock-v1"
cp docs/workspace.e010.example.json "$GEOCHEM_WORKSPACE/workspace.local.json"
# 将私有 geochem.db 拷贝到上述 whole-rock-v1/；不要覆盖已有冻结版本。
# 在配置文件中登记已从本机迁移的完整冻结准备 run 的 prepared_data 路径。
# 文件管理节点不运行 python、conda、uv、nvidia-smi 或哈希检查。
```

`workspace.local.json` 模板的 `e010_prepared` 是待配置位置，不是现有数据。两个准备方式二选一：

1. 如原已冻结准备 run `20261007T135913Z-s42-773907` 可取得，迁移整个目录至本工作区 `runs/E010/`，将本地 `e010_prepared` 映射设为其 `prepared_data`。保留原 manifest 和校验，不补写当前提交。完整性与原库输入核验在本机或允许的计算节点进行，不在文件管理节点运行。
2. 如仅有数据库而没有可迁移的准备资产，在本机 Windows 新建准备 run，使用以下命令。本次已经完成这些步骤，迁移现有新冻结 run 即可。不要在 HPC 文件管理节点生成数据，也不为准备占用 GPU 卡时。

```powershell
$PREP_RUN = ./scripts/local_cpu.ps1 scripts/experiment.py new-run E010 --config stage3_5_element2vec_v2/configs/e010/prepare.json
./scripts/local_cpu.ps1 scripts/experiment.py exec --run-dir "$PREP_RUN" '--' -m stage3_5_element2vec_v2.src.e010.cli prepare
./scripts/local_cpu.ps1 scripts/experiment.py exec --run-dir "$PREP_RUN" '--' -m stage3_5_element2vec_v2.src.e010.cli verify-data
./scripts/local_cpu.ps1 scripts/experiment.py exec --run-dir "$PREP_RUN" '--' -m stage3_5_element2vec_v2.src.e010.cli matrix
./scripts/local_cpu.ps1 scripts/experiment.py check --run-dir "$PREP_RUN" --rehash
./scripts/local_cpu.ps1 scripts/experiment.py freeze --run-dir "$PREP_RUN" --status completed --outcome not_evaluated `
  --summary 'E010 full-coverage preparation and schema/coverage checks passed; no model training.' `
  --algorithms 'E010 immutable full-coverage preparation; frozen robust statistics' --device cpu
```

最后将本地 `workspace.local.json` 的 `e010_prepared` 改为新冻结 run 的 `prepared_data` 路径（相对工作区），而非直接使用模板里的待配置位置。迁移至 HPC 时复制原始数据库和整个冻结准备 run，再在 HPC 本地配置中映射同一输入。准备和所有训练必须使用干净的已提交代码；不自动下载旧 checkpoint，新训练从初始化开始。

## 显式提交后 12 次训练

确认数据库与冻结准备资产迁移就绪后才执行。在集群允许的提交入口从仓库根目录提交；脚本的 `new` 模式会在计算节点作业内创建协调 run，因此提交前无需运行 Python。调度器负责 GPU 隔离：一个 srun step 启动四 tasks，使用 `--gpus-per-task=1 --gpu-bind=single:1`，沿用 Slurm 提供的 `CUDA_VISIBLE_DEVICES`，代码不自行按 rank 改写设备编号。参见 [Slurm GRES 文档](https://slurm.schedmd.com/gres.html) 和 [srun 文档](https://slurm.schedmd.com/srun.html)。需要集群启用相应 GPU/task 绑定支持。

```bash
sbatch --output=slurm-e010-%j.out --error=slurm-e010-%j.err \
  scripts/slurm/e010_a6000.sh new
```

协调器先创建四个独立队列 run，再让每个 worker 在自己的队列中调用原扫描器。每项有独立训练 run，依次执行 preflight（无优化器更新）→ train → verify-training → check --rehash → freeze。设备正确记录为 CUDA；CUDA 预检会同步计时并记录设备及峰值显存。任何 task 失败，`srun --kill-on-bad-exit=1` 停止同作业其他 task，不继续发起新训练。已完成的训练 run 保持冻结；不会进行后续评价、额外 seed 或短消融。

`slurm_registry.json` 只由协调器写入，四个 worker 只读它，各写自己的队列 registry。训练参数、数据指纹和代码版本由运行工具固定，输入映射漂移会被拒绝。调度器启动日志为被 Git 忽略的 `slurm-e010-*.out/err`，其中打印新协调 run 的路径；登记后的命令日志和所有研究产物仍在各 run 中。单项日志在其训练 run 的 `logs/`，各 task 输出在协调 run 的 `logs/slurm-*-slot-*.log`。结束后的 Python 完整性检查在本机或允许的计算节点执行；协调器的成功 receipt 是 `slurm_complete.json`。科学优劣需要另做评价，训练完成不表示科学正结果。

## 恢复中断

保留原协调 run 和四个队列 run。使用相同干净提交、同一准备资产、相同 CUDA backend / PyTorch 版本和相同 GPU 分配方式重新提交：

```bash
sbatch --output="$LAUNCH_RUN/logs/batch-%j.out" --error="$LAUNCH_RUN/logs/batch-%j.err" \
  scripts/slurm/e010_a6000.sh resume "$LAUNCH_RUN"
```

从原 `slurm_registry.json` 定位队列和训练 run；已完成项跳过，存在 durable `latest.pt` 的训练以 `--resume` 继续。冻结输入不能变。失败历史保留。原流程仅支持训练步骤的严格恢复：失败发生在 preflight/verify 等非训练步骤，或训练尚无第一个 durable checkpoint 时，需保留旧证据并创建新 run，不能删除日志冒充恢复。配置更改也需新 run。

Slurm 强制终止可能留下 `.execution.lock` 或未写回的命令状态。必须先用调度器确认旧作业及所有子进程已停止，再检查对应锁与运行证据，必要时由维护者处理残留锁；脚本不会自动删除锁。协调/队列 run 暂保持可写以支持恢复；训练子 run 验证后冻结。父 run 有失败调度历史时不适用仅针对训练子 run 的 `--allow-verified-resume`，不要强行冻结为成功；记录调度中断及所有已冻结子 run 后单独收尾。

## 验证边界

本次在本机 Python 3.12 上完成 13 项新增无数据调度测试、14 项原运行管理测试，全部通过；Python 编译检查、Bash 语法检查和公开树审计通过。测试虚拟环境只补充 PyYAML，留在 Git 忽略目录，没有安装训练环境或生成研究 run。

上段记录的是初始交付时的验收边界。数据库现已拷贝核验，本机 Python 3.11 CPU 环境的后续验收另见环境说明。本次切换 RTX A6000 后，本机通过 16 项调度/bootstrap 测试、14 项工作区测试、Bash 语法检查和公开树审计，涵盖资源总量、设备拒绝、GPU 隔离和实际设备 receipt。当前没有 Slurm/CUDA 实机验收；首次真实执行仍须通过每个配置的 CUDA preflight，35G 主机内存是否足够尚待实测。没有提交作业或启动训练。
