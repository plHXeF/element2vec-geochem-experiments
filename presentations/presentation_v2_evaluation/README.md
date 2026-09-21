# V2 汇报生成源

`templates/` 收录原始版本及 77 页修订版的 15 份生成、汇总、媒体和验证源码。原稿完整保存在工作区 `deliverables/presentation_v2_evaluation/`；这里只把机器专属绝对路径替换成占位符。图表、讲稿和科学结论没有改写。

模板不是直接运行入口。需要继续编辑汇报时，以 E009 新建运行并指定本目录 `config.json`，再通过 `scripts/experiment.py exec --run-dir RUN -- presentations/presentation_v2_evaluation/prepare_sources.py` 准备独立副本。生成器、工作文件和成品都位于本次运行的 `presentation_workspace/`；输入链接指向冻结的明确历史运行。不要编辑交付包内的历史脚本。

使用原 PowerPoint 生成流程前，在 `workspace.local.json` 的 `presentation_runtime` 中登记 `artifact_tool_module`、`node_modules`、`skill_directory`、`python`。这些可选依赖只影响汇报生成，不影响实验运行。模板原件未在本次目录中找到，若需重新执行原版导入流程，请在同一配置中登记 `template` 的真实位置；准备工具会明确列出缺项，不会替换模板或猜测路径。

材料准备工具只复制源码和创建输入链接，不重训、不重算评价、不重新生成幻灯片。维护性修改先写回本仓库模板并提交，再准备新运行副本；运行已保存的生成器前仍需使用该运行的环境并确认可写。定稿后复制到新的 deliverables 子目录，保留 provenance 的实验与运行编号。
