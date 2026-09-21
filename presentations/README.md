# 汇报代码与文字

- `presentation_html/build_presentation.py`：以 `asset://stage2_run` 和 `asset://stage3_run` 的固定汇总生成 HTML。先 `new-run E003 --config presentations/presentation_html/config.json`，再通过 `exec --run-dir` 调用；结果在当前运行的 `presentation/`。
- `presentation_v2_evaluation/`：原版和修订版 PowerPoint 生成源码模板、讲稿与准备入口；细节见该目录 README。

定稿在工作区 `deliverables/`，附 provenance；新修订保存到新目录。历史交付不覆盖。模板、渲染依赖缺失时先报告具体缺项，不臆造已重新构建的结果。
