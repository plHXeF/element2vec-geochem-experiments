# Experiment status

> **当前工作入口（2026-09 整理后）：** 请从 [E001—E009 实验总表](experiments/README.md)、[管理规则](docs/WORKSPACE_MANAGEMENT.md) 和 [运行命令](docs/EXPERIMENT_COMMANDS.md) 开始。数据和产物位于仓库外，每次执行创建独立运行。以下研究报告与旧路径用于历史阅读；写入型历史命令须按新运行流程执行，不能直接覆盖历史输出。

Snapshot date: 2026-08-24

## Value RoPE Z follow-up — completed 2026-09-12

The 50-epoch `experiment/value-rope-z` run completed at step 13,200. The September 11–12 paired audit uses the original 12,000 samples and 8,000/4,000 projection pools, both trained models, and seed-42 random RoPE/MLP controls.

Final fixed training-probe loss improves from 0.35137 to 0.30026. Small-step tangent stability and joint-intervention additivity improve, but chemistry-only rock-neighbor purity falls from 46.98% to 38.35%, while reference/source purity increases. Label-masked and CLS controls confirm the regression. Random models are smoother than trained models. Common-z suppression is not concentration-ratio invariance; wide out-of-support SiO2 sweeps show near-collisions.

Retain this model as an architectural comparison, not a replacement for V2. See the [complete paired report](stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPORT.md), [11-item registry](stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/experiment_registry.csv), and [reproduction commands](stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPRODUCE.md). Checkpoints, sample-level outputs and fitted bases remain private and ignored.

## Overall status

The repository is a publishable research snapshot, not a finished scientific claim or production package. Code, configurations, aggregate evidence, vocabularies, final static embeddings, training logs, and checkpoints are preserved. Private datasets and sample-level derived data are excluded.

## Stage 1 — Pyrite representation benchmark

Status: completed exploratory baseline.

- Data scope: 7,778 pyrite samples, 14 trace elements, 6 standardized deposit types.
- `rank_uniform_z` and `log_robust_z` are the strongest value-only baselines.
- `raw_robust_z` remains dominated by extreme concentrations and is retained as a negative control.
- `observed_mask_z` and `log_plus_mask` expose strong reporting/missingness structure. Their high metrics must not be interpreted as purely geological.
- PCA and UMAP dimension sweeps were completed. Two- or three-dimensional plots are visual summaries, not sufficient-space estimates.

Open problem: replace random stratified validation with deposit/district/region-aware grouped validation when reliable grouping fields are available.

## Stage 2 — Geology-informed representations

Status: completed exploratory benchmark; generalization claim not established.

- Ratio, balance, and information-theoretic 12D representations were compared with 14D value baselines.
- Stratified XGBoost macro-F1 reached about 0.93 for the strongest baselines, but literature and deposit leakage can inflate it.
- Citation GroupKFold was incomplete because some folds lacked classes.
- Geo-informed 12D features remain useful for interpretation and fusion experiments, but did not beat the strongest value baselines overall.

Open problem: create valid locality/suite/deposit grouped folds and audit whether important features proxy analytical source.

## Stage 3 — Pyrite element-token Transformer prototype

Status: prototype completed.

- Demonstrated sample-level input-token and contextual-token representations.
- The principal sample latent is `label_masked`; `label_visible` is an auxiliary leakage control only.
- Contextual token matrices showed stronger structure than input-token matrices.
- The bundled small checkpoint is suitable for reproducing the prototype, not for a production or out-of-domain claim.

Open problem: evaluate with fold-specific pretraining and realistic missing-panel perturbations.

## Stage 3.5 — Element2Vec V2

Status: 50-epoch whole-rock training completed; final checkpoint published.

- Input vocabulary: 34 numeric fields and 96 pure-label tokens.
- Static label embedding: 509D; unified token input: 512D; projected/contextual width: 768D.
- Model: 12 layers, 12 heads, 3,072D FFN, no positional embedding.
- Training ended at optimizer step 13,200.
- Final logged training total loss: approximately 0.302.
- The fixed-probe total loss was approximately 0.353 at step 13,000.
- The lowest instantaneous training total loss was approximately 0.174 at step 11,227; it should not be treated as a selected validation checkpoint.
- Only the final full checkpoint and same-step component exports are available.

The training objective includes numeric-value reconstruction, numeric-tag prediction, and rock group/origin/type prediction. Consequently, geological label structure can reflect supervision and cannot by itself demonstrate emergent geochemical semantics.

## Stage 3.5 — Value-RoPE Z architecture experiment

Status: implementation and local verification complete; full 50-epoch training not yet run.

- Experiment ID and branch: `value_rope_z` / `experiment/value-rope-z`, based on commit `5650034`.
- Frozen inputs: the same 34 numeric fields, 96 pure-label tokens, and V2 robust-z transform statistics.
- Token input: direct 768D tag/label embedding with no input fusion MLP.
- Numeric injection: standardized z rotates all 64 dimensions of every attention head's Q and K; V receives no direct numeric input.
- Masked numeric values use position zero plus one shared learned 768D mask-state vector.
- Backbone and training budget remain 12 layers, 12 heads, 3,072D FFN, batch 512 with accumulation 2, 50 epochs, and 13,200 optimizer steps.
- Parameter count: 85,255,040, compared with 86,798,583 for the released V2 model.
- Generated checkpoints and sample-level outputs are local-only and excluded from Git.

The experiment is an architecture comparison, not yet evidence of improved representation quality. Stage 3.6 comparisons require a completed checkpoint and revised interpretation of the direct input-embedding level.

## Stage 3.6 — Breadth-first representation-space audit

Status: 50 registered experiments completed or explicitly retained as unavailable/failed.

Original registry status counts:

- 13 strong positive;
- 4 moderate positive;
- 17 weak/ambiguous;
- 13 negative;
- 1 invalid because of leakage/confounding;
- 2 technically failed/unavailable.

Main observations:

- Contextual tokens strongly encode the surrounding sample chemistry.
- Single- and multi-element perturbations produce partly stable local directions.
- Synthetic mixing, differentiation, melting, weathering, and alteration paths are often continuous, but continuity alone is not geological validation.
- Volcanic–plutonic label relations have some structure, while direct sample-to-label alignment remains weak.
- Static element embeddings show weak periodic, REE, and broad affinity structure.
- Source and missingness structure are strong; a mask-only source-neighborhood purity reached approximately 0.94 in one control.
- Token permutation is effectively invariant, while random 30% token removal gives a mean latent cosine of only about 0.329.
- Strict CIPW is unavailable; the existing normative experiments are NNLS-based proxies.

## Stage 3.6 — Multiscale subspace replay

Status: completed and verified.

- All 50 original experiments entered the compatibility audit.
- 2,041 projection-registry rows were generated, including 1,760 random-projection controls.
- The core experiment-dimension matrix contains 355 rows.
- Seventy-four figures were indexed and 25 verification checks passed.
- FULL, PCA64, PCA32, PCA3, and PCA2 are the required primary comparisons. Additional random projections, whitening, and component-removal tests are controls.

Strict low-rank recovery was found only for `F2_single_element_jacobian`:

| Space | Metric | FULL | Delta vs FULL | Interpretation |
|---|---:|---:|---:|---|
| PCA64 | 0.443 | 0.395 | +0.048 | Stable low-rank recovery |
| PCA32 | 0.457 | 0.395 | +0.062 | Stable low-rank recovery |

The bootstrap delta intervals were approximately `[0.033, 0.064]` for PCA64 and `[0.035, 0.089]` for PCA32. Other apparent improvements in perturbation or label-relation experiments were projection-sensitive or preservation-under-compression rather than strict recovery.

The contextual-mean neighborhood audit shows the trade-off directly:

| Space | Geology purity | Source purity | Missingness purity | Geology − source |
|---|---:|---:|---:|---:|
| FULL | 0.647 | 0.363 | 0.606 | 0.284 |
| PCA64 | 0.646 | 0.363 | 0.604 | 0.283 |
| PCA32 | 0.642 | 0.355 | 0.594 | 0.287 |
| PCA3 | 0.537 | 0.189 | 0.457 | 0.349 |
| PCA2 | 0.495 | 0.156 | 0.389 | 0.340 |

PCA2/PCA3 reduce source information but also discard substantial geological neighborhood structure. They are visualization/compression spaces, not recovered universal semantics.

Direction-energy analysis further shows that confounding is more top-heavy than local chemical response: missingness directions place about 83% of their energy in the first three PCs and nearly all in the first 32, whereas single-element perturbations retain only about 34% in the first 32 and 53% in the first 64. Some useful chemical directions therefore lie outside the highest-variance nuisance subspace.

## Current blockers and risks

1. No clean external or locality-held-out evaluation exists for the V2 model.
2. V2 label supervision and full-data training contaminate label readability and probe results.
3. Publication, source, missingness, token count, and analytical-panel differences are entangled with geology.
4. The vocabulary is too small for a strong REE/HFSE/LILE semantic test.
5. Final-only checkpoints prevent temporal stability and checkpoint-selection analysis.
6. Several synthetic geological processes are simplified and not mass-balanced or petrologically calibrated.
7. Sample-label spaces do not naturally align; separately fitted PCA bases must never be compared directly.
8. The project has scripts but no formal test suite or CI yet.
9. Exact environment reproduction across macOS/MPS, Linux/CUDA, and CPU remains unverified.

## RoPE_geo + 15% numeric mask (2026-09-12)

Independent implementation on experiment/value-rope-z; user-launched training completed
13,200 steps and paired analysis completed 2026-09-13. Only numeric coordinates and numeric-task mask density change;
the 85,255,040-parameter RoPE backbone and prior training budget remain fixed.
The original RoPE-z reports and conclusions are preserved.

- [Experiment, formulas and commands](stage3_5_element2vec_v2/experiments/rope_geo_mask15/README.md)
- [Independent report and current verification](stage3_5_element2vec_v2/experiments/rope_geo_mask15/REPORT.md)
- [Paired evaluation protocol](stage3_5_element2vec_v2/experiments/rope_geo_mask15/PAIRED_PROTOCOL.md)

Original 12,000 paired IDs and 8,000/4,000 analysis pools are frozen locally.
The new combination improves several small-step geometric metrics but fails the
global-representation goal: chemistry-only FULL k=15 rock purity is 0.3065
(V2 0.4698; RoPE-z 0.3835), and source purity increases to 0.6293
(RoPE-z 0.6063). Random controls show why smoothness alone is insufficient.
There is no held-out pretraining validation. Private outputs are Git-ignored;
only new aggregate tables and figures accompany the independent report.

## Recommended next experiments

1. Retrain with publication/locality/suite-aware partitions and evaluate strictly outside the pretraining fold.
2. Separate nuisance and geology subspaces using explicit adversarial or residualization controls, with held-out validation.
3. Map local Jacobian/tangent-space rank by rock group and realistic compositional perturbations.
4. Replace random token dropout with real analytical-panel ablations and explicit censoring/detection-limit states.
5. Expand the element vocabulary before repeating static element-semantic claims.
6. Add intermediate checkpoints in the next training run.
7. Validate process experiments against documented petrological forward models and natural suites.
