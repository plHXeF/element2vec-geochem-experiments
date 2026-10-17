# 本机 CPU 科研环境

本目录只声明依赖，不把阶段目录安装成软件包。Python 3.11 的具体补丁版本由 `.python-version` 固定，依赖和下载校验由 `uv.lock` 固定。适用 Windows x86-64；HPC Linux/CUDA 保持独立环境。

在本机用户设置中配置 uv 工具位置、`UV_PYTHON_INSTALL_DIR` 和 `UV_CACHE_DIR`。工具、Python 和缓存统一放在本机数据盘；每个项目分别维护环境与锁文件。实际绝对路径只保存在用户设置和 `workspace.local.json`，不写进共享科学配置。

所有命令在仓库根目录执行：

```powershell
uv sync --locked --project environments/local-cpu
uv run --locked --project environments/local-cpu python scripts/experiment.py list
```

也可使用 `./scripts/local_cpu.ps1 scripts/experiment.py list`。它直接使用已经同步的环境，设置 UTF-8 和单线程数值库，不自动下载或更新依赖；退出状态透传给调用者。直接调用解释器时，也应保留这些环境设置。

PowerShell 脚本会消费未加引号的 `--`，调用 `exec` 时需将它写为字面量 `'--'`：

```powershell
./scripts/local_cpu.ps1 scripts/experiment.py exec --run-dir $RUN '--' -m stage3_5_element2vec_v2.src.e010.cli verify-data
```

包装器还为直接调用设置项目环境内的 Matplotlib 缓存；通过 `experiment.py exec` 启动时，仍使用各 run 独立的缓存目录。

此环境的解释器为本目录 `.venv/Scripts/python.exe`，本机 `workspace.local.json` 应指向它。已有仓库根目录 `.venv` 保留，不能与这里的 Python 3.11 环境混用。新终端可以从用户 PATH 找到 uv；已打开的终端需重新打开或刷新用户环境变量。

## 安装与更新约定

- NumPy、pandas、SciPy、scikit-learn、Matplotlib、Seaborn、PyYAML、Pillow、openpyxl 来自 PyPI。PyTorch 显式来自官方 CPU 源；不装 CUDA、Intel GPU 扩展或不需要的图像/音频模型组件。
- 初次解析使用仓库原有依赖范围，生成精确锁文件并经过合成验收；日常只执行 `--locked` 同步，实验执行时不自动升级。
- 完整分析扩展按需使用 `uv sync --locked --project environments/local-cpu --extra analysis`。默认不安装 XGBoost、UMAP、SHAP、HDBSCAN、Plotly；扩展本身仍记录在锁文件中。同步时应保持相同 extra，避免精确同步移除扩展。
- 需要更新时，修改依赖声明、显式运行 `uv lock --project environments/local-cpu`（版本升级使用 `--upgrade`），重新同步并跑相关测试，审查后提交声明与锁文件。正式实验需使用干净提交。
- 仅提交环境声明、版本固定文件、锁文件与说明。环境、缓存、数据库和样本级结果不得提交；机器上的 Python 与锁文件版本都记录进实验运行环境。
- 根目录 `environment.yml` 和 `requirements.txt` 保留为历史/跨平台范围声明；本目录是新的 Windows CPU 精确环境，不用于安装 HPC CUDA 依赖。

## 本地准备与计算边界

本轮仅建立环境并执行合成检查，不自动运行 E010 全量准备或训练。数据库已经只读核验；之后另建准备 run，依次 prepare、verify-data、matrix、check、freeze，再登记其 `prepared_data` 为本机资产。不能把只读数据库核验 run 当成准备资产。

首次全量准备只启动一个进程，数值库线程为 1。建议先关闭大型应用，保留约 12 GiB 可用内存和至少 20 GiB 磁盘空间；这是保守启动条件，不是峰值内存保证。

完整冻结准备 run 和数据库可迁移至 HPC，虚拟环境不可跨 Windows/Linux 复制。HPC 文件管理节点只做集群允许的文件传输与提交操作，不执行 Python、环境检查、运行登记、哈希或数据准备。训练前登记和输入核验由计算节点作业内完成。
