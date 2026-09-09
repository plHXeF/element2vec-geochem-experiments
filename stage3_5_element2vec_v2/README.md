# Stage 3.5: Element2Vec V2

Element2Vec V2 is an exploratory representation model for whole-rock geochemistry. It is independent from Stage1, Stage2, and Stage3, and writes all artifacts under `stage3_5_element2vec_v2/outputs/`.

## Design Goal

V2 removes the fixed tabular panel assumption. Each sample is represented as:

```text
[CLS] + observed numeric-label tokens + available pure-label tokens
```

Missing fields do not create tokens. Token order is randomly shuffled during training, while `[CLS]` stays first. The Transformer uses no ordinary positional embedding, so it should not learn a fixed oxide/element order.

## Token Types

Numeric-label tokens contain a measured value:

```text
SiO2 = 72.3
Rb_ppm = 180
Sr_ppm = 250
```

Pure-label tokens contain categorical geology labels:

```text
rock_group = igneous
rock_origin = plutonic
rock_type = granite
```

Pure labels preserve field identity. The token is `rock_type:granite`, not just `granite`.

## Unified Token Input

Every non-CLS token is built as a 512-dimensional vector:

```text
509d label embedding
+ 1d normalized_value
+ 1d mask_tag_flag
+ 1d mask_value_flag
= 512d token input
```

Numeric tokens use field embeddings such as `SiO2` or `Rb_ppm`. Pure-label tokens use full field-value embeddings such as `rock_type:granite`. Pure-label values are always `0`, and their `mask_value_flag` is always `0`.

All non-CLS tokens pass through the same `TokenProjector`:

```text
Linear(512 -> 768) -> GELU -> LayerNorm -> Dropout
Linear(768 -> 768) -> GELU -> LayerNorm -> Dropout
Linear(768 -> 768)
```

`[CLS]` is a direct 768-dimensional learnable parameter.

## Mask Tasks

Each training sample receives exactly one random mask task:

1. Mask a numeric value, e.g. `Rb_ppm = [MASK_NUMBER]`.
2. Mask a numeric tag, e.g. `[MASK_NUMERIC_TAG] = 72.3`.
3. Mask a pure label, e.g. `rock_type:[MASK]`.

The default probabilities are:

```yaml
mask_numeric_value: 0.45
mask_numeric_tag: 0.15
mask_pure_label: 0.40
```

Pure-label masking defaults to `rock_type` first, then `rock_origin`, then `rock_group`.

## Model

The base config uses:

- `d_model = 768`
- 12 Transformer encoder layers
- 12 attention heads
- FFN dimension 3072
- pre-LN Transformer blocks
- no positional embedding

Decoder heads are separated:

- numeric value regression head, Huber loss
- numeric tag classification head
- `rock_group` classification head
- `rock_origin` classification head
- `rock_type` classification head

## Current Training Defaults

The default config is tuned for a longer whole-rock run:

- batch size: `1024`
- gradient accumulation steps: `1`
- learning rate: `2.5e-4`
- minimum learning rate: `3.0e-6`
- warmup: `500` steps
- epochs: `50`
- scheduler: warmup + cosine decay with minimum LR
- checkpoint save interval: once per epoch
- task-balanced batch masking: enabled
- fixed train probe: enabled, 8192 fixed samples and fixed masks, evaluated every 500 steps

For `v_high_quality_dataset` with 270,241 samples and batch size 1024, one epoch is:

```text
ceil(270241 / 1024) = 264 optimizer steps
```

So the default 50-epoch run is:

```text
264 * 50 = 13200 optimizer steps
```

To force a different run length, pass `--max-steps`; this overrides `epochs`.

## Value Normalization

Numeric fields are standardized per field. Major oxides use robust z-score directly. Trace elements use:

```text
log10(value + eps) -> robust_z
```

Transform metadata is saved to:

```text
stage3_5_element2vec_v2/outputs/vocab/value_transform_stats.csv
```

## Smoke Test

Build vocab and stats:

```bash
python -m stage3_5_element2vec_v2.src.data.build_vocab \
  --config stage3_5_element2vec_v2/configs/v2_base.yaml
```

Run a short shape/gradient smoke test only:

```bash
python -m stage3_5_element2vec_v2.src.training.train_v2 \
  --config stage3_5_element2vec_v2/configs/v2_base.yaml \
  --max-samples 256 \
  --max-steps 2
```

## Manual Training

Do not let the agent run the full training loop unattended. Start training manually and monitor loss, speed, memory, and MPS behavior:

```bash
conda activate Element2Vec
python -m stage3_5_element2vec_v2.src.training.train_v2 \
  --config stage3_5_element2vec_v2/configs/v2_base.yaml
```

Checkpoints are saved as:

```text
outputs/checkpoints/token_projector.pt
outputs/checkpoints/transformer_encoder.pt
outputs/checkpoints/decoders.pt
outputs/checkpoints/full_model.pt
```

Training logs are saved to:

```text
outputs/logs/train_log.csv
outputs/logs/probe_log.csv
```

`train_log.csv` contains per-step losses plus an EMA window. `probe_log.csv` uses fixed samples and fixed masks, so it is the cleaner curve for checking whether the model is actually improving rather than just seeing random mask noise.

## Value-RoPE Z Experiment

`configs/v2_value_rope_z.yaml` is a controlled V2 architecture experiment. Numeric-field and pure-label IDs are embedded directly in 768 dimensions; the input fusion MLP is removed. A numeric value enters only as a continuous rotary position on every attention layer's Q and K tensors. The experiment uses the frozen V2 vocabulary and robust-z statistics from `outputs/vocab/`.

For each attention head, all 64 dimensions are rotated in adjacent pairs with `base=10000` and `scale=1`. CLS, pure-label, padding, and masked-value positions are zero. A shared learned 768-dimensional state distinguishes a masked numeric value from a real standardized value of zero. The model has 85,255,040 parameters and retains the V2 12-layer, 12-head, 3072-dimensional FFN backbone and five training objectives.

Run the tests before training:

```bash
python -m unittest discover \
  -s stage3_5_element2vec_v2/tests \
  -p 'test_*.py'
```

Run a two-step CPU smoke test against the private database:

```bash
python -m stage3_5_element2vec_v2.src.training.train_v2 \
  --config stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml \
  --sqlite-path ../legacy_whole_rock_geochem/geochem.db \
  --outputs-dir stage3_5_element2vec_v2/outputs/value_rope_z/smoke \
  --device cpu --batch-size 2 --max-samples 8 --max-steps 2 \
  --probe-size 8 --probe-batch-size 2
```

Start the complete 50-epoch run from scratch with:

```bash
python -m stage3_5_element2vec_v2.src.training.train_v2 \
  --config stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml \
  --sqlite-path ../legacy_whole_rock_geochem/geochem.db
```

The full state is written to `outputs/value_rope_z/checkpoints/full_model.pt`. All files under `outputs/value_rope_z/` and all newly generated model-weight suffixes are ignored by Git. Value-RoPE checkpoints store their architecture, frozen vocabularies, and value-transform records; extraction rejects mismatched inputs.

The Value-RoPE config prints one explicit `[batch]` line per micro-batch. Each line reports the epoch, micro-batch position, completed optimizer steps, actual batch size, and numeric-value/numeric-tag/pure-label mask counts. Pass `--batch-log-every N` to print every N micro-batches, or `--batch-log-every 0` to disable these lines.

Extract label-masked 768-dimensional CLS and mean-numeric latents after training:

```bash
python -m stage3_5_element2vec_v2.src.evaluation.extract_embeddings \
  --config stage3_5_element2vec_v2/configs/v2_value_rope_z.yaml \
  --sqlite-path ../legacy_whole_rock_geochem/geochem.db \
  --checkpoint stage3_5_element2vec_v2/outputs/value_rope_z/checkpoints/full_model.pt
```

## Embedding Export

The training script exports initial embeddings before optimization:

```text
outputs/embeddings/initial_numeric_tag_embeddings.csv
outputs/embeddings/initial_pure_label_embeddings.csv
```

After training, it exports:

```text
outputs/embeddings/final_numeric_tag_embeddings.csv
outputs/embeddings/final_pure_label_embeddings.csv
```

Inspect embedding neighbors:

```bash
python -m stage3_5_element2vec_v2.src.evaluation.inspect_embeddings \
  --config stage3_5_element2vec_v2/configs/v2_base.yaml \
  --prefix final
```

## Latent Export

After training:

```bash
python -m stage3_5_element2vec_v2.src.evaluation.extract_embeddings \
  --config stage3_5_element2vec_v2/configs/v2_base.yaml \
  --checkpoint stage3_5_element2vec_v2/outputs/checkpoints/full_model.pt
```

This writes:

```text
outputs/latents/sample_cls_latent.csv
outputs/latents/sample_mean_numeric_latent.csv
```

During latent export, all pure labels are masked by default to prevent label leakage.

## Benchmark

Run after latent export:

```bash
python -m stage3_5_element2vec_v2.src.evaluation.benchmark_latents \
  --config stage3_5_element2vec_v2/configs/v2_base.yaml \
  --latent stage3_5_element2vec_v2/outputs/latents/sample_cls_latent.csv
```

The summary is written to:

```text
outputs/metrics/v2_benchmark_summary.csv
```

Metrics include downstream classification for `rock_group`, `rock_origin`, and `rock_type`, kNN purity@15/@30, KMeans ARI/NMI, and a UMAP-3D or PCA-3D projection table.

## Current Limits

- Does not use balance, ratio, or information-theoretic handcrafted features.
- Does not use a fixed tabular panel.
- Does not use ordinary positional embedding.
- Does not implement a V3 sequence model.
- Does not automatically run long training from the agent.
- Leakage-safe benchmark should use latents exported with pure labels masked.
