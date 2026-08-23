# Stage 3: Pyrite Element Token Transformer

This directory contains an exploratory prototype for a pyrite geochemistry element-token Transformer.

It is intentionally independent from previous stages. It reuses the cleaned pyrite processed data and writes all new outputs under `stage3_element_token_model/outputs/`.

## 1. Purpose

The goal is to test whether an element-token Transformer representation preserves deposit-type structure, local neighborhood structure, clustering structure, and visualization structure.

This stage is not a strict generalization benchmark. The model is trained on the full dataset and then representations are extracted from the same dataset for exploratory sanity checks.

## 2. Data Format

The input is the already processed pyrite table:

- `../pyrite_benchmark/data/processed/pyrite_metadata.csv`
- `../pyrite_benchmark/data/processed/pyrite_element_values.csv`
- `../pyrite_benchmark/data/processed/pyrite_numeric_mask.csv`

Each row is one sample. The default columns are:

- sample id column: `sample_id`
- label column: `type`
- element columns: `As, Ni, Cu, Pb, Co, Au, Ag, Bi, Sb, Zn, Se, Te, Tl, Mn`

These are configured in:

`configs/pyrite_token_transformer.yaml`

## 3. Log + Robust Normalization

For each element abundance `c_i`, preprocessing computes:

```text
log10(c_i + eps)
```

Then each element column is robust-normalized using the column median and IQR:

```text
x_i = (log10(c_i + eps) - median_i) / IQR_i
```

Missing numeric values are imputed to `0` after robust normalization, which corresponds to the element median in normalized space. Missingness is still preserved through `visible_mask_i`.

## 4. Element Token Construction

Each element is a token. For each sample and each element:

```text
token_input_i = [x_i, visible_mask_i, element_embedding_i]
T_i = MLP(token_input_i)
```

The learned `element_embedding_i` represents element identity only. It does not contain sample concentration by itself.

## 5. Deposit Type Token

Deposit type is categorical. It is never fed as a continuous number.

Each deposit type has a learned embedding. The model also has a special `[MASK_DEPOSIT]` embedding.

If the label is visible:

```text
T_deposit = MLP([0, 1, deposit_type_embedding])
```

If the label is masked:

```text
T_deposit = MLP([0, 0, MASK_DEPOSIT_embedding])
```

## 6. Training Tasks

Training uses full data and masked modeling:

1. masked element reconstruction
   - random element abundance values are masked;
   - the model predicts normalized log abundance;
   - default loss: Huber loss.

2. masked deposit type prediction
   - deposit token is randomly masked;
   - the model predicts deposit type from element tokens through self-attention;
   - loss: CrossEntropyLoss.

Total loss:

```text
loss = lambda_elem * loss_elem + lambda_type * loss_type
```

Training loss and masked deposit-token accuracy are sanity checks only. They are not strict validation metrics.

The current default run uses `epochs: 1000` with cosine learning-rate annealing from `1e-3` to `1e-5`.

## 7. Why Main Latent Uses `label_masked`

All main latent benchmark results must use `label_masked` mode. In this mode the deposit type token is forced to `[MASK_DEPOSIT]` during extraction.

This avoids label leakage. If `label_visible` is used, the model can carry target-label information through the deposit token, so any clustering/classification result may be artificially inflated.

`label_visible` latent is exported only for auxiliary inspection.

## 8. Embedding Outputs

`outputs/embeddings/element_embedding.csv`

- one row per element;
- first column: `element`;
- remaining columns: `emb_0 ... emb_31`;
- this is the learned element identity embedding;
- it is the same for every sample and must not be flattened alone for sample-level benchmark.

`outputs/embeddings/deposit_embedding.csv`

- one row per deposit type;
- includes `[MASK_DEPOSIT]`;
- first column: `deposit_type`;
- remaining columns: `emb_0 ... emb_31`.

## 9. Sample Token Matrix Outputs

Do not use element embeddings alone as sample-level features. Element embeddings are constant across samples and contain no sample concentration information.

This stage exports sample-specific token matrices:

`outputs/embeddings/sample_input_token_flat.csv`

- projected element token matrix before the Transformer;
- represents `abundance + visible mask + element identity` after MLP projection;
- flattened per sample;
- generated in label-masked mode.

`outputs/embeddings/sample_encoded_token_flat.csv`

- Transformer-encoded element token matrix after self-attention;
- flattened per sample;
- generated in label-masked mode.

`outputs/embeddings/sample_token_matrix_flat.csv`

- compatibility alias of `sample_encoded_token_flat.csv`.

## 10. Latent Outputs

Main latent:

`outputs/latents/transformer_latent_label_masked_mean_element.csv`

Auxiliary latent:

`outputs/latents/transformer_latent_label_visible_mean_element.csv`

Each row contains:

- `sample_id`
- `deposit_type`
- `latent_0 ... latent_n`

Supported pooling modes:

- `mean_element`: mean pooling over element tokens;
- `deposit`: deposit token output;
- `mean_all`: mean pooling over all tokens.

## 11. Benchmark Spaces

`src/build_spaces.py` creates:

1. `previous_baseline_space`
   - copied from Stage 2 baseline if available.

2. `log_robust_space`
   - raw log + robust normalized 14-element table.

3. `log_robust_pca12_space`
   - PCA-12 of `log_robust_space`.

4. `sample_input_token_flat_pca12_space`
   - PCA-12 of `sample_input_token_flat.csv`.
   - evaluates the projected element-token matrix before self-attention.

5. `sample_encoded_token_flat_pca12_space`
   - PCA-12 of `sample_encoded_token_flat.csv`.
   - evaluates the Transformer-contextualized token matrix.

6. `transformer_latent_label_masked_space`
   - label-masked mean-element Transformer latent.
   - main Transformer representation.

## 12. Benchmark Metrics

`src/evaluate_spaces.py` runs the same benchmark for every space:

- XGBoost classification with Stratified 5-fold CV;
- fallback to sklearn HistGradientBoostingClassifier if XGBoost is unavailable;
- KNN-15 classification with Stratified 5-fold CV;
- KNN-15 neighborhood purity in full space;
- KMeans ARI/NMI/homogeneity/completeness/silhouette;
- UMAP-3D visualization;
- fallback to PCA-3D if `umap-learn` is unavailable.

Main summary:

`outputs/metrics/all_space_benchmark_summary.csv`

JSON summary:

`outputs/metrics/all_space_benchmark_summary.json`

## 13. Run Commands

The project `geochem-benchmark` environment does not currently include PyTorch. On this machine, the existing `datasafe` conda environment has CUDA PyTorch and can run the Transformer training/extraction steps without downloading large packages.

Run from this directory:

```powershell
cd E:\Geochem\stage3_element_token_model
```

Training and extraction:

```powershell
C:\Users\17599\.conda\envs\datasafe\python.exe src/preprocess.py --config configs/pyrite_token_transformer.yaml
C:\Users\17599\.conda\envs\datasafe\python.exe src/train.py --config configs/pyrite_token_transformer.yaml
C:\Users\17599\.conda\envs\datasafe\python.exe src/extract_embeddings.py --config configs/pyrite_token_transformer.yaml
C:\Users\17599\.conda\envs\datasafe\python.exe src/extract_latent.py --config configs/pyrite_token_transformer.yaml --mode label_masked --pooling mean_element
C:\Users\17599\.conda\envs\datasafe\python.exe src/extract_latent.py --config configs/pyrite_token_transformer.yaml --mode label_visible --pooling mean_element
C:\Users\17599\.conda\envs\datasafe\python.exe src/build_spaces.py --config configs/pyrite_token_transformer.yaml
```

Final evaluation can be run with the project `geochem-benchmark` environment, because this step does not require PyTorch and that environment has XGBoost, UMAP, and Plotly:

```powershell
E:\Geochem\.conda\envs\geochem-benchmark\python.exe src/evaluate_spaces.py --config configs/pyrite_token_transformer.yaml
E:\Geochem\.conda\envs\geochem-benchmark\python.exe src/report_figures.py --config configs/pyrite_token_transformer.yaml
```

If evaluation is run with `datasafe`, scripts still complete, but XGBoost falls back to sklearn HistGradientBoosting and UMAP falls back to PCA-3D because those optional packages are not installed there.

If a future environment has both PyTorch and XGBoost/UMAP/Plotly installed, the whole sequence can be run in that single environment.

## 14. Viewing Results

Open:

`outputs/metrics/all_space_benchmark_summary.csv`

for the main comparison table.

Open:

`outputs/stage3_report.md`

for a compact report.

Open:

`outputs/report_fig/`

for report-ready capability plots, including the heatmap, radar chart, classification comparison, structure metrics, token gains relative to log robust, and loss/LR curve.

Open:

`outputs/figures/*_3d.html`

for 3D visualization. If the environment lacked Plotly, HTML files use a Plotly CDN fallback.
