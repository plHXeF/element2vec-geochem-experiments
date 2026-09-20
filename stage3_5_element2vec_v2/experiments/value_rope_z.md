# Experiment: value_rope_z

## Question

Can Element2Vec use numeric concentration only through continuous Q/K rotary position while preserving the V2 label vocabulary, Transformer capacity, mask tasks, and optimization protocol?

## Frozen comparison

- Baseline commit: `5650034`.
- Baseline experiment: completed Element2Vec V2 50-epoch run.
- Dataset view: `v_high_quality_dataset`, historically 270,241 samples.
- Vocabulary and value transforms: tracked V2 artifacts under `outputs/vocab/`.
- Training: seed 42; micro-batch 512; accumulation 2; 50 epochs; 13,200 optimizer steps; AdamW at `2.5e-4`; 500-step warmup; cosine minimum `3e-6`.

## Systematic change

The 512-to-768 token projector is removed. Numeric-field and pure-label embeddings are direct 768-dimensional inputs. Visible robust-z values generate RoPE angles for Q and K in all 12 layers. The rotary dimension is the complete 64-dimensional attention head, with `base=10000` and `scale=1`. V does not receive a direct numeric signal.

Masked values use a zero rotary position and add one shared learned 768-dimensional state to the field embedding. Pure labels, CLS, padding, and masked values have zero rotary position. Masked numeric tags retain their visible value position.

## Outputs and interpretation

The model has 85,255,040 parameters. Outputs are local under `outputs/value_rope_z/` and are excluded from Git. The fixed probe remains a training-set diagnostic. No quality result should be recorded until the full run finishes and is compared with V2 using the same downstream controls. Stage 3.6 input-level analyses must describe `projected_tokens` as a compatibility alias for direct label embeddings.

Training prints an explicit progress record for every micro-batch by default. The record includes epoch, batch index and total, completed optimizer steps, actual batch size, and the three mask-task counts. `logging.batch_log_every` or `--batch-log-every` controls its frequency.

## Completed training and paired audit

Training completed at 13,200 steps / 50 epochs. The September 11–12 audit confirms the budget, vocabulary and transformation statistics, and evaluates four models on identical inputs with MPS/CPU consistency checks. Final training-probe total loss is 0.30026 versus V2's 0.35137; local tangent stability improves, while rock-neighbor purity and source confounding worsen. This model is retained as a comparison rather than promoted as a better overall latent representation.

See the [analysis report](../../stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPORT.md) and [reproduction instructions](../../stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPRODUCE.md). The private audit directory is `outputs/value_rope_z/audit/`; published outputs contain only aggregates.
