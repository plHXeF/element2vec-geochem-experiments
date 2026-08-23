# Element2Vec V2 子空间表示审计

## 已读取的上一轮证据

- 主报告：`stage3_6_element2vec_space_exploration/reports/element2vec_exploration_report.md`。
- 实验 registry：50 行，覆盖 A–K 与 X；所有原实验配置、指标、图和观察目录均存在。
- 共享缓存：12,000 个样品，34,000 个 contextual/projected token，随机模型样品 3,000 个；随机种子为 42。
- 旧 `pca_sweep` 已覆盖 256/128/64/32/16/8/5/3/2D，但属于“把旧实验函数置于投影 context 后重放”的敏感性扫描；本任务不把它单独当作低秩地球化学语义证据，而只作为 paired replay 数据源，并新增 RP、whitening、drop-PC、PC attribution、energy、bootstrap 与混杂控制。
- 历史样品 ID 限制：50 个旧 `samples.csv` 中只有 1 个非空。其余实验通过固定代码路径、固定 seed=42、固定端元/扰动参数重放；无法声称旧文件保存了逐行样品 ID。

## 原生维度、拟合秩与上游/最终表示

| representation | native D | fitting rows | estimable rank | provenance |
| --- | ---: | ---: | ---: | --- |
| static_trained_numeric | 509 | 35 | 34 | upstream |
| static_random_numeric | 509 | 35 | 34 | upstream |
| static_trained_pure | 509 | 96 | 95 | upstream |
| static_random_pure | 509 | 96 | 95 | upstream |
| mlp_projected_numeric_token | 768 | 20000 | 116 | model_output |
| contextual_numeric_token | 768 | 20000 | 768 | model_output |
| sample_projected_mean_numeric | 768 | 8000 | 118 | model_output |
| sample_contextual_mean_numeric | 768 | 8000 | 768 | model_output |
| sample_cls | 768 | 8000 | 768 | model_output |
| random_sample_projected_mean_numeric | 768 | 3000 | 82 | model_output |
| random_sample_contextual_mean_numeric | 768 | 3000 | 407 | model_output |
| random_sample_cls | 768 | 3000 | 571 | model_output |
| pure_label_projected | 768 | 96 | 95 | model_output |
| random_pure_label_projected | 768 | 96 | 95 | model_output |
| cross_joint_contextual_label | 768 | 8096 | 768 | model_output |

`static_*` 的 509D 是 embedding lookup 的上游输出；MLP 后 token 与 Transformer/pooled latent 的最终坐标均为 768D。35 个 numeric tags 虽然原生为 509D，但样本秩至多 34，所以 PCA64 不可估计；PCA32 可估计但已接近最大秩，必须谨慎解释稳定性。本审计不会补零或复制维度。96 个纯标签允许 PCA64，但其小词表稳定性必须由 bootstrap 检查。

## PCA fitting/evaluation pool

- PCA fitting pool：8000 个固定样品 ID，不使用目标标签。
- evaluation pool：4000 个互斥固定样品 ID。
- token fitting pool：20000 个来自 fitting 样品的固定 token。
- 纯标签 PCA：在完整可用词表上无监督拟合；跨空间实验使用 `cross_joint_contextual_label` 的单一共享 basis，禁止分别 PCA 后直接比较。
- 旧实验的监督 probe 仍保留原定义以保证 paired replay；新增 S 系列监督/混杂指标仅在固定 evaluation pool 上评价，并把 PCA basis 固定在 reference pool。

## 完整重放兼容性

- FULL/PCA64/PCA32/PCA3/PCA2 可运行实验数：{'FULL': 44, 'PCA64': 41, 'PCA32': 44, 'PCA3': 44, 'PCA2': 44}。
- 6 个覆盖或技术失败实验保持 invariant unavailable/failed，不能因投影而变成正结果。
- A 家族受 35 个 tag 的拟合秩限制；H、I4、X3 属于跨空间实验，必须使用共享 basis 或训练外 alignment。
- 严格 CIPW 仍未实现；I1–I4 中可运行部分只能称为 NNLS normative proxy。

## 禁止项

- 未使用补零、复制维度或结果驱动的标准化。
- PCA 不使用实验目标标签。
- UMAP 不参与向量差、余弦、类比或状态判定。
