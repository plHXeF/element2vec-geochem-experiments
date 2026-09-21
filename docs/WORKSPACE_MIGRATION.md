# 本地工作区迁移记录

迁移：2026-09-20 开始，2026-09-21 完成整理。没有推送远程，没有重训模型，没有重算完整 benchmark。

## 保留与整合

- 原主仓库完整迁入工作区 `repo/`；原 Git 历史和 `v0.1.0-stage3.6` 保留。
- 原 `experiment/value-rope-z` 的历史保留。geo15 未提交实现与汇总证据保存为 `experiment/e008-rope-geo-mask15` 的独立提交，再纳入整理分支。
- 原 RoPE 工作副本迁入 `worktrees/e008-rope-geo-mask15`，Git 关联已修复。
- 统一评价代码进入主仓库；它的样本缓存、拟合统计量和完整产物进入 E009 历史运行。
- 外层重复阶段源码、旧总说明、旧 Windows 环境及空 Git 目录归入 `archive/pre-reorg-20260920`。
- 原仓库 13,289 个未追踪的旧产物（252,826,956 字节）转入 `archive/pre-reorg-20260920/repo-untracked/`，逐文件映射保留。
- 原汇报交付包整体保留在 `deliverables/`。迁移不改变图表和科学结论。

## 本地证据

以下路径相对于工作区，仅保存在本地：

- `archive/pre-reorg-20260920/history.bundle`：迁移前所有 Git 引用的备份；未提交工作另有 patch 和完整源码快照。
- `archive/pre-reorg-20260920/files-before.jsonl`：原始文件清单及 SHA-256。
- `archive/pre-reorg-20260920/path-mapping.json`：旧位置到新位置的映射。
- `archive/pre-reorg-20260920/verification/migration.json`：168,130 个文件、25,696,297,469 字节的核验记录，零丢失、零内容不匹配。代码比较使用迁移前源码快照，数据和结果比较使用迁移后的实际文件。
- `archive/pre-reorg-20260920/verification/model-equivalence.json`：固定 24 个样本、CPU 推理，V2、RoPE-z、RoPE_geo 的 768 维输出最大绝对差均为 0。

原始配置、样本 ID、拟合/评价划分、checkpoint、旧缓存哈希均保留。历史执行 commit 未独立确认时记为 `unknown`；新建的保存提交不是历史训练提交。

## 回退

先停止新运行并检查 Git 状态。依据路径映射查找原位置及当前保存位置；从源码快照或 Git bundle 恢复需要的实现。保留新旧运行的独立编号，不用旧文件覆盖新结果。当前归档和历史运行均不自动清理。

## 使用与验收范围

主仓库纳入 E001—E009 实验卡、运行工具和统一路径入口。日常规则同时写入工作区及仓库 AGENTS，旧总说明改为导航入口。11 份历史运行均保留来源未知的明确标记、原配置、权重及逐文件校验；两份汇报包保存 provenance。

58 项入口/语法检查验证了缺少运行目录时停止，并避免旧脚本把 `--help` 当作实际执行。工作区测试覆盖并行隔离、路径迁移、配置/数据/权重变化、源代码快照、运行锁与冻结保护。模型数值验收是固定小样本 CPU 推理，不声称完成重新训练或全量 benchmark。

本机可选环境依赖及原 PowerPoint 模板未随整理补装。汇报的 15 份旧生成源码已转为可登记本机依赖的模板，成品保持原样；本次未重新构建汇报。相关缺项由准备入口明确报告。

本次最终相关测试共 67 项通过（工作区 11、Stage 3.5 25、Stage 3.6 11、latent 20），另有 58 项入口/语法检查。Git 内容检查在暂存后执行；明细位于本地 `verification/validation.json`。
