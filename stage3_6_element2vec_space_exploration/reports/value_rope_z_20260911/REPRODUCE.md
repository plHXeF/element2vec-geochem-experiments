# Reproduce the paired value-RoPE audit

Run from the experiment repository root in the existing `Element2Vec` conda environment. Exact pairing requires the original private Stage 3.6 cache and saved fitting/evaluation pools. The audit fails rather than silently drawing replacement samples.

Set paths to your private files. Use a new output directory under the ignored experiment outputs:

```bash
conda activate Element2Vec
export LEGACY_AUDIT=/path/to/original/stage3_6_element2vec_space_exploration
export V2_CHECKPOINT=/path/to/original/stage3_5_element2vec_v2/outputs/checkpoints/full_model.pt
export GEOCHEM_DB=/path/to/geochem.db
export AUDIT_OUT=stage3_5_element2vec_v2/outputs/value_rope_z/audit_replay
export MPLCONFIGDIR=/tmp/value-rope-mpl
```

RoPE weights default to `stage3_5_element2vec_v2/outputs/value_rope_z/checkpoints/full_model.pt`. Override `--rope-checkpoint` consistently if needed. Both trained checkpoints are step 13,200. These commands do not start training.

```bash
python -m stage3_6_element2vec_space_exploration.src.value_rope_audit prepare \
  --legacy "$LEGACY_AUDIT" --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT"

python -m stage3_6_element2vec_space_exploration.src.value_rope_audit extract \
  --legacy "$LEGACY_AUDIT" --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT" \
  --device mps --batch-size 128

python -m stage3_6_element2vec_space_exploration.src.value_rope_audit probe \
  --legacy "$LEGACY_AUDIT" --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT" \
  --sqlite-path "$GEOCHEM_DB" --device mps

python -m stage3_6_element2vec_space_exploration.src.value_rope_audit analyze \
  --legacy "$LEGACY_AUDIT" --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT"

python -m stage3_6_element2vec_space_exploration.src.value_rope_supplement extract \
  --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT" --device mps

python -m stage3_6_element2vec_space_exploration.src.value_rope_supplement controlled \
  --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT" --device mps

python -m stage3_6_element2vec_space_exploration.src.value_rope_supplement analyze \
  --v2-checkpoint "$V2_CHECKPOINT" --output "$AUDIT_OUT"

python -m stage3_6_element2vec_space_exploration.src.value_rope_diagnostics \
  --sqlite-path "$GEOCHEM_DB" --output "$AUDIT_OUT"
```

MPS needs access to the host GPU; restricted processes may report it unavailable. Use `--device cpu` when necessary. The four models run sequentially. Hardware differences remain possible; one seed is not multiple independent training replicates.

```bash
python -m unittest discover -s stage3_5_element2vec_v2/tests -p 'test_*.py'
python -m unittest discover -s stage3_6_element2vec_space_exploration/tests -p 'test_*.py'
python -m stage3_6_element2vec_space_exploration.src.verify_value_rope_audit --output "$AUDIT_OUT"
python scripts/verify_public_tree.py --tracked
```

Publish only aggregate results through the explicit allowlist; choose a new report directory for a replay:

```bash
python -m stage3_6_element2vec_space_exploration.src.value_rope_publish_summary \
  --output "$AUDIT_OUT" \
  --report-dir stage3_6_element2vec_space_exploration/reports/value_rope_z_replay
```

Private files include `*_global.npz`, `*_masked_full.npz`, `*_stimuli.npz`, `*_controlled.npy`, sample metadata, perturbation inputs, probe predictions, fingerprints and fitted PCA bases. They stay under the ignored output directory. Main latent arrays have `mean`, `cls`, and pooled `input` keys; stimulus arrays also include mean layers 1/3/6/9 and queried final tokens. RoPE adds queried first-layer rotated Q/K; final layer mean is `mean`.

The analyzed pool contains 12,000 samples, not the entire database. Stage 3.5's existing `extract_embeddings` command remains available for a full all-label-masked CSV export.

Thresholds are frozen in `REPORT.md`: finite-difference reference h=0.025; fine control h=0.005; linearity ≤10% error and ≥0.99 cosine; distant pairs separated by ≥4 z with distance below one median 0.25-z step; 1,000 paired cluster bootstrap resamples; ten random-projection seeds per dimension. Constant paths have undefined geometry rather than perfect scores.
