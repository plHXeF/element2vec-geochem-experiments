# Reproducibility and continuation guide

## Current local workspace workflow

Use [workspace management](WORKSPACE_MANAGEMENT.md) and [per-experiment commands](EXPERIMENT_COMMANDS.md) for execution. Create a run, execute each step through `scripts/experiment.py exec --run-dir ...`, check and freeze it. Stage directories contain code and selected historical reports; new artifacts go only to `runs/`. All old output paths below describe the original snapshot, not current writable destinations.

The usable local Python is selected in `workspace.local.json`. Tests exist under `tests/`, `stage3_5_element2vec_v2/tests`, `stage3_6_element2vec_space_exploration/tests`, and `latent_evaluation_benchmark/tests`. The original Windows environment is archived. Optional dependencies may be absent; record the actual fallback, versions and device in the run. `scripts/doctor.py --check-schema` checks local inputs without exporting rows.


## September 2026 paired value-RoPE audit

The completed follow-up compares the final V2 and value-RoPE checkpoints, both seed-42 random architectures, and raw/missingness controls on the original Stage 3.6 samples. See [exact commands and data requirements](../stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPRODUCE.md) and the [aggregate report](../stage3_6_element2vec_space_exploration/reports/value_rope_z_20260911/REPORT.md). Sample-level outputs and checkpoints stay in the ignored value-RoPE output directory. New replay directories preserve the historical audit.

## Reproduction levels

The repository supports three distinct levels. State which one was used in any report.

1. **Artifact inspection:** clone, download checkpoints, and inspect existing aggregate reports. No dataset is needed.
2. **Representation replay:** supply the private database/workbook, use the released checkpoint, regenerate caches/latents, and rerun evaluation.
3. **Training replay:** supply data and retrain from initialization. This is the most hardware- and software-sensitive level.

## Baseline environment

Create the environment from `environment.yml`. The released V2 training was performed with PyTorch on Apple hardware using the available MPS backend. Exact floating-point identity is not expected across MPS, CUDA, and CPU. The project fixes pseudo-random seeds where practical, but low-level kernels may remain nondeterministic.

Record at minimum:

```bash
python --version
python -m pip freeze
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), torch.cuda.is_available())"
git rev-parse HEAD
```

Also record the private input hash and SQLite view row count without adding either to Git if the hash itself is considered sensitive.

## V2 training audit

Before a long run:

```bash
python -m stage3_5_element2vec_v2.analysis.scripts.p1a_verify_architecture
python -m stage3_5_element2vec_v2.analysis.scripts.p1b_verify_masking
python -m stage3_5_element2vec_v2.analysis.scripts.p1c_verify_data_pipeline
```

Then run the two-step smoke test from the root README. Long training should be started manually and monitored. Check:

- training and fixed-probe losses;
- task balance and per-task losses;
- finite gradients and gradient clipping;
- device memory after probes and checkpoints;
- checkpoint hashes and ability to reload on CPU.

## Value-RoPE Z continuation experiment

The Value-RoPE configuration is `stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml`. It intentionally reads the frozen V2 vocabulary and robust-z statistics from the frozen `asset://v2_vocab`; if any required file is missing, training fails instead of fitting replacement statistics.

Before training, run:

```bash
python -m unittest discover -s stage3_5_element2vec_v2/tests -p 'test_*.py'
python scripts/verify_public_tree.py --tracked
```

The formal run keeps the V2 seed, tasks, optimizer, learning-rate schedule, probe protocol, and 13,200-step budget. The checkpoint is a fresh initialization and is incompatible with the released MLP-fusion V2 checkpoint. Training and extraction accept explicit `--sqlite-path`, `--outputs-dir`, and `--device` overrides; training also accepts batch and probe-size overrides for isolated smoke tests.

All new Value-RoPE outputs are written below the explicit run directory and remain outside Git. The full checkpoint includes the model architecture, both frozen vocabularies, and value-transform records. Extraction validates these records before loading weights.

## Stage 3.6 replay order

The normal dependency order is:

```text
audit_project -> extract_cache -> run_experiments -> build_reports -> verify_outputs
                                      |
                                      +-> run_pca_sweep -> build/verify PCA report
                                      +-> run_subspace_replay -> build/verify subspace report
```

Caches, fitting pools, projection bases, and sample-level outputs are intentionally absent from the public repository. They are reproducible local artifacts and must remain ignored.

## Frozen subspace protocol

- Fit PCA only on the declared reference/fitting pool.
- Transform evaluation samples; do not use test labels to fit or select PCA.
- Use the same maximum-rank PCA basis and truncate it for nested dimensions where the implementation specifies this.
- Required primary spaces: FULL, PCA64, PCA32, PCA3, PCA2 when native rank permits.
- Use ten fixed random orthogonal projections per target dimension for the main random baseline.
- Apply L2 normalization only where the original cosine experiment used it.
- Do not manufacture unavailable dimensions by zero padding or repeated coordinates.
- For cross-space experiments, use one joint basis or a training-only alignment map with held-out validation.

## Historical outputs versus regenerated outputs

Published CSV/Markdown/figures are a frozen snapshot. A fresh run may differ because of package versions, data revisions, nondeterministic kernels, or a changed research view. Write regenerated outputs to a newly registered run directory, compare aggregate metrics, and commit only a deliberate new snapshot after reviewing data exposure.

## Independent RoPE_geo + 15% mask run

Use the new Stage 3.5 config v2_value_rope_geo_mask15.yaml and the existing frozen
vocabularies. Geo references/rules are versioned in the config and validated
against the checkpoint. No geo per-element statistics are fitted.
User-launched full training and paired analysis completed on 2026-09-13.
Tested MPS smoke outputs remain isolated from the formal run.
See [formulas, environment, training and extraction commands](../stage3_5_element2vec_v2/experiments/rope_geo_mask15/README.md)
and the [paired analysis protocol](../stage3_5_element2vec_v2/experiments/rope_geo_mask15/PAIRED_PROTOCOL.md).
The new 12,000-ID pairing is frozen locally; do not replace the prior audit's
raw stimuli with geo coordinates or overwrite any published RoPE-z result.
Post-training commands and legacy-cache replay verification are documented in
[ANALYSIS_REPRODUCE.md](../stage3_5_element2vec_v2/experiments/rope_geo_mask15/ANALYSIS_REPRODUCE.md).

## Verification before publication

```bash
python -m compileall -q pyrite_benchmark stage2_geo_representation_xgb \
  stage3_element_token_model stage3_5_element2vec_v2 \
  stage3_6_element2vec_space_exploration
python scripts/doctor.py
python scripts/verify_public_tree.py --tracked
```

The project currently has no formal unit-test suite or CI. Stage-specific verification scripts and public-tree auditing are the minimum release gate.
