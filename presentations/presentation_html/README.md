# HTML 汇报页说明

本目录是黄铁矿主/微量元素地球化学 representation benchmark 的本地交互式 HTML 汇报页。

## 如何打开

双击 `index.html`，或用浏览器打开：

```text
presentation_html/index.html
```

如果交互图或 iframe 加载失败，打开静态备份版：

```text
presentation_html/static_backup.html
```

## 如何翻页

- 方向键左/右；
- 空格键进入下一页；
- 页面右下角按钮；
- 支持 `?slide=5` 和 `#slide-5` 定位。

## 交互图

已有 Plotly/UMAP3D HTML 没有 inline 到主页面，而是复制到了：

```text
presentation_html/interactive/
```

主要交互图包括：

- Stage2 balance UMAP3D；
- Stage3 sample encoded token UMAP3D；
- Stage3 log robust / label-masked latent 对照图；
- Stage3 raw 476D token embedding sanity-check 图（若存在）。

## 引用的文件来源

- `stage2_report`: `assets/tables/stage2_report.md`
- `stage3_report`: `assets/tables/stage3_report.md`
- `stage2_unsupervised`: `assets/tables/unsupervised_metrics.csv`
- `stage2_xgb`: `assets/tables/xgb_classification_metrics.csv`
- `stage2_umap_metrics`: `assets/tables/umap3d_metrics.csv`
- `stage3_summary`: `assets/tables/all_space_benchmark_summary.csv`
- `stage2_raw`: `interactive/copied_stage2_umap3d_htmls/raw_robust_z_14_umap3d_interactive.html`
- `stage2_balance`: `interactive/copied_stage2_umap3d_htmls/geo_balance_12_umap3d_interactive.html`
- `stage2_log`: `interactive/copied_stage2_umap3d_htmls/log_robust_z_14_umap3d_interactive.html`
- `stage2_rank`: `interactive/copied_stage2_umap3d_htmls/rank_uniform_z_14_umap3d_interactive.html`
- `stage2_ratio`: `interactive/copied_stage2_umap3d_htmls/geo_ratio_12_umap3d_interactive.html`
- `stage2_info`: `interactive/copied_stage2_umap3d_htmls/geo_info_12_umap3d_interactive.html`
- `stage3_baseline`: `interactive/copied_stage3_umap3d_htmls/previous_baseline_space_3d.html`
- `stage3_encoded`: `interactive/copied_stage3_umap3d_htmls/sample_encoded_token_flat_pca12_space_3d.html`
- `stage3_input`: `interactive/copied_stage3_umap3d_htmls/sample_input_token_flat_pca12_space_3d.html`
- `stage3_latent`: `interactive/copied_stage3_umap3d_htmls/transformer_latent_label_masked_space_3d.html`
- `stage3_log`: `interactive/copied_stage3_umap3d_htmls/log_robust_space_3d.html`
- `stage3_raw476`: `interactive/copied_stage3_umap3d_htmls/sample_raw_element_token_flat_476_umap3d.html`
- `stage2_balance_png`: `assets/figures/geo_balance_12_UMAP1_UMAP2.png`
- `stage3_encoded_png`: `assets/figures/sample_encoded_token_flat_pca12_space_3d.png`
- `stage3_report_capability`: `assets/figures/capability_heatmap.png`
- `stage3_training`: `assets/figures/training_loss_and_lr.png`

同时生成或复制了汇报图：

- `assets/figures/report_logic_flow.svg`
- `assets/figures/six_dimension_framework.svg`
- `assets/figures/stage2_ability_profile.svg`
- `assets/figures/stage3_space_comparison.svg`
- `assets/figures/element2vec_architecture.svg`

## 注意事项

本汇报为 exploratory report。Stage3 的 Transformer/Element2Vec 结果是全量训练后的 representation sanity check，不是严格泛化估计。

后续需要补充：

- GroupKFold / 按文献或矿区分组验证；
- 外部数据验证；
- 对元素 embedding、attention、元素共现网络的地质解释；
- 在更大全岩地球化学数据上训练通用元素表示并迁移到黄铁矿数据。
