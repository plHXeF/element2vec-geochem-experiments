# 实验工作区管理规则

## 从哪里开始

工作区与 Git 仓库分开：工作区根目录保存 `data/`、`runs/`、`worktrees/`、`deliverables/`、`archive/` 和 `workspace.local.json`；`repo/` 是继续维护的代码主仓库。

先阅读 [实验总表](../experiments/README.md)，再阅读对应实验卡。每张 `experiment.json` 是实验问题、协议、依赖和结论的权威来源。现有报告是冻结证据，不能用新结果覆盖。

阶段目录保留原名。Stage 表示长期模块；实验编号表示研究问题；Git 分支表示开发中的实现；运行编号表示一次执行。改变种子或超参数应新建运行；改变假设或对照定义应新建实验；改变代码实现才需要开发分支。

## 每天的操作

在 `repo/` 中使用含 PyTorch 的 Python 环境运行。正式运行前提交代码；临时调试可以使用 `--scratch`，工具会保存工作源码和修改补丁。

```bash
python scripts/experiment.py list
python scripts/experiment.py new-run E009
```

第二条命令返回完整运行目录。后续各步骤使用同一个目录：

```bash
python scripts/experiment.py exec --run-dir /path/to/run -- latent_evaluation_benchmark/run.py prepare
python scripts/experiment.py exec --run-dir /path/to/run -- latent_evaluation_benchmark/run.py extract-real
python scripts/experiment.py check --run-dir /path/to/run --rehash
```

`exec` 记录命令、退出状态与日志，并向旧脚本传递运行上下文。也可直接向研究脚本传 `--run-dir`，但需要自行留存执行记录；正式流程推荐使用 `exec`。完整步骤见 [实验命令](EXPERIMENT_COMMANDS.md)。长训练、全量评估都由研究者明确启动。

完成检查后写明结论、实际算法与设备，再冻结：

```bash
python scripts/experiment.py freeze --run-dir /path/to/run \
  --status completed --outcome mixed \
  --summary '填写本次结果、限制和下一步' \
  --algorithms '填写实际使用的模型、分类器和降维算法' --device cpu
```

`completed/failed/cancelled` 是执行状态；`positive/negative/mixed/inconclusive/not_evaluated` 是科学结果。失败的计算不能标记为负结果。冻结后保存逐文件校验表，研究入口拒绝再写入。改变配置或重新执行时创建新运行；只有已支持恢复的步骤才可继续未冻结的原运行。

## 代码和实验卡

- `main` 收纳通过检查、可以复现的实现，正负结果均可合入；推荐模型由实验卡记录。
- 分支格式：`experiment/e010-topic`、`fix/topic`、`chore/topic`。独立代码修改使用 `worktrees/`；配置变体不复制整个阶段目录。
- 实验卡记录问题、对照、配置、输入资产、上游实验、协议、运行和解释边界。A—K、X 子实验编号保留在 E005/E006 下。
- 每完成一个实验，更新卡片的状态、结论与代表性运行，然后运行 `python scripts/experiment.py index` 生成总表。`list` 会另外发现所有本地运行。
- 正式运行绑定提交版本；报告提交与训练提交可以不同，不能用报告提交冒充训练提交。历史缺失来源记为 `unknown`。
- 提交前运行相关测试和 `python scripts/verify_public_tree.py --tracked`。将待纳入文件加入暂存区后再检查；不要把未跟踪文件遗漏在审查之外。

## 数据、路径与依赖

`asset://名称` 指向 `workspace.local.json` 中登记的输入；`run://子目录` 指向本次运行；`repo://路径` 指向代码。配置中的逻辑名称可以提交，机器上的绝对位置只保存在本地设置和运行记录中。工作区可通过 `GEOCHEM_WORKSPACE` 指定，也可从代码目录向上发现。

输入数据按版本冻结。Stage 1 的重新清洗写入当前运行的 `prepared_data/`，不会替换 `pyrite-v1`。需要使用新数据时，在 `data/` 下登记一个新版本、记录文件 SHA-256 和准备协议，再为新实验配置相应资产。黄铁矿实验使用 processed 表；全岩实验使用自己的数据库。

跨实验输入引用确定的资产与上游运行，不能依赖 `latest`。新运行保存数据、词表和权重校验值，`exec` 启动前重新校验。旧缓存的原始配置与哈希不改写；本次迁移只登记旧路径到新路径的映射。新配置签名使用逻辑资产标识，不把本机路径作为科学变量。

代码、配置、汇总指标、精选静态图和说明进入 Git。数据、样本级结果、拟合池、投影基、缓存、新增权重、本机路径均留在仓库外。权利声明和已有授权边界保持原样。本次没有远程同步或发布动作。

## 结果保留

历史运行和完成的运行永久保留 manifest、科学配置、环境、主要指标、结论以及必要权重。历史缓存没有自动删除策略；清理必须单独指定并确认可重建依据。

汇报成品放在 `deliverables/`，每份成品的 provenance 文件列出引用运行。只将精选汇总复制到 Git 下的 `reports/`。原始报告与完整图表仍属于对应冻结运行。

研究解释沿用原协议：Stage 3 主结果使用 `label_masked`；V2/RoPE 主地质评估使用 chemistry-only 或明确的 all-label-masked 模式；全量预训练后的随机划分不能称为外部泛化；来源、missingness 与随机模型/随机投影对照必须保留。缺少可选依赖时要记录实际使用的算法，不能把替代算法记成 XGBoost 或 UMAP。

## Agent 开始和结束工作

开始：阅读工作区与仓库 `AGENTS.md`，检查 Git 状态，读取实验卡及相应协议，确认输入资产和运行目录。结束：检查相关测试与内容审计，写明变更和验证结果，更新实验卡后生成总表；记录尚未完成的步骤，不把技术失败当研究负结果。

新增实验时复制相近实验卡的结构，分配尚未使用的永久编号（当前下一个为 E010），填写研究问题、对照、模块、配置、资产、上游、协议、状态和结论。编号不回收。新增运行会由 `list` 自动发现；精选/正式运行再写入实验卡的 `runs` 列表。

并行修改示例（从主仓库执行）：

```bash
git worktree add ../worktrees/e010-topic -b experiment/e010-topic main
```

工作副本仍使用同一工作区的数据与运行目录。将来移动整个工作区后，在 `repo/` 执行 `git worktree repair ../worktrees/*` 修复 Git 自身的绝对关联；科学配置无需改写。本机设置模板见 `docs/workspace.local.example.json`。

运行工具拒绝同一 run 的并行写入。显式命令中的输入/权重覆盖会记录额外校验值；命令参数与配置快照共同构成完整执行定义。只改动输出文件、manifest 或配置以绕过校验不属于续跑。

## 环境与检查范围

本次检查使用 Python 3.11 的 Element2Vec 环境。数据准备的 `openpyxl` 与可选的 xgboost、umap-learn、plotly、shap、hdbscan 当前缺失；需要对应能力时按 `environment.yml` 建立环境，不能把未执行的阶段称为通过。已有 processed 数据、权重与完成的结果可以直接读取。本次没有执行长训练、全量评估或幻灯片重建。

E009 的 20 项现有回归测试读取 `asset://latent_reference` 的历史缓存；是只读验收，不能用空运行替代这些测试输入。其他测试与工作区管理测试的命令见仓库 AGENTS。
