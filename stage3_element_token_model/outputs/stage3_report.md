# Stage 3 Element Token Transformer Report

## Scope

This stage trains a pyrite geochemistry element-token Transformer on the full processed dataset. It is an exploratory benchmark and sanity check, not a strict generalization estimate.

Main benchmark latent is `label_masked + mean_element`, so the deposit type token is forced to `[MASK_DEPOSIT]` during latent extraction.

Training used 1000 full-data epochs with cosine learning-rate annealing from `1e-3` to approximately `1e-5`. The final training log reached epoch 1000 with total loss about `0.143`; this remains a training sanity check, not a generalization metric.

## Spaces Evaluated

| space_name | n_features | knn15_neighborhood_purity | xgb_macro_f1_mean | knn15_macro_f1_mean | kmeans_ari | kmeans_nmi | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| previous_baseline_space | 14 | 0.9176480671980801 | 0.9310962583559956 | 0.8980699651921691 | 0.11400529985004731 | 0.25939887351619045 | ok |
| log_robust_space | 14 | 0.9182394788720324 | 0.9310962583559956 | 0.8985636942115329 | 0.11029593179988952 | 0.25769196294111335 | ok |
| log_robust_pca12_space | 12 | 0.9127710636838948 | 0.8675469075784962 | 0.9003120149286872 | 0.13659673294136943 | 0.29061984160482895 | ok |
| sample_input_token_flat_pca12_space | 12 | 0.9310362561069683 | 0.8943886122591991 | 0.9036315400597983 | 0.21826033287185273 | 0.4591923417329892 | ok |
| sample_encoded_token_flat_pca12_space | 12 | 0.9435073283620469 | 0.9213085606798013 | 0.9302469636892547 | 0.2996341916595803 | 0.5363223608837021 | ok |
| transformer_latent_label_masked_space | 64 | 0.9495928687751779 | 0.9457878290116059 | 0.945723144794206 | 0.3270571742210068 | 0.517296123379027 | ok |


## Best Local Neighborhood Structure

| space_name | knn15_neighborhood_purity | kmeans_ari | kmeans_nmi | notes |
| --- | --- | --- | --- | --- |
| transformer_latent_label_masked_space | 0.9495928687751779 | 0.3270571742210068 | 0.517296123379027 | ok |
| sample_encoded_token_flat_pca12_space | 0.9435073283620469 | 0.2996341916595803 | 0.5363223608837021 | ok |
| sample_input_token_flat_pca12_space | 0.9310362561069683 | 0.21826033287185273 | 0.4591923417329892 | ok |
| log_robust_space | 0.9182394788720324 | 0.11029593179988952 | 0.25769196294111335 | ok |
| previous_baseline_space | 0.9176480671980801 | 0.11400529985004731 | 0.25939887351619045 | ok |
| log_robust_pca12_space | 0.9127710636838948 | 0.13659673294136943 | 0.29061984160482895 | ok |


## Classification Sanity Check

The classification metrics use Stratified 5-fold CV. Because the Transformer was trained on all samples, these scores are not a clean generalization claim. They are only used to compare representation spaces.

| space_name | xgb_accuracy_mean | xgb_macro_f1_mean | xgb_weighted_f1_mean | knn15_accuracy_mean | knn15_macro_f1_mean |
| --- | --- | --- | --- | --- | --- |
| transformer_latent_label_masked_space | 0.9686288529414195 | 0.9457878290116059 | 0.9684511735622536 | 0.969015200985295 | 0.945723144794206 |
| previous_baseline_space | 0.9607869134312566 | 0.9310962583559956 | 0.9604669964023225 | 0.9384171633093347 | 0.8980699651921691 |
| log_robust_space | 0.9607869134312566 | 0.9310962583559956 | 0.9604669964023225 | 0.9381601765595681 | 0.8985636942115329 |
| sample_encoded_token_flat_pca12_space | 0.9607871614081782 | 0.9213085606798013 | 0.9605254879730427 | 0.9611722695674455 | 0.9302469636892547 |
| sample_input_token_flat_pca12_space | 0.9435594607328545 | 0.8943886122591991 | 0.9428098703081478 | 0.9487023367691915 | 0.9036315400597983 |
| log_robust_pca12_space | 0.9188739367989486 | 0.8675469075784962 | 0.9165914042891451 | 0.9372588631084733 | 0.9003120149286872 |


## Outputs

- `outputs/embeddings/element_embedding.csv`: learned element identity embeddings.
- `outputs/embeddings/deposit_embedding.csv`: learned deposit type embeddings plus `[MASK_DEPOSIT]`.
- `outputs/embeddings/sample_input_token_flat.csv`: sample-specific projected element token matrix, flattened.
- `outputs/embeddings/sample_encoded_token_flat.csv`: sample-specific Transformer-encoded element token matrix, flattened.
- `outputs/latents/transformer_latent_label_masked_mean_element.csv`: main label-masked sample latent.
- `outputs/metrics/all_space_benchmark_summary.csv`: main summary table.
- `outputs/figures/*_3d.html`: 3D visualizations. If Plotly package was unavailable, HTML uses a CDN fallback.
- `outputs/report_fig/*`: report-ready capability plots and summary charts.

## Notes

- If `xgboost` is unavailable in the active environment, evaluation falls back to `HistGradientBoostingClassifier`.
- If `umap-learn` is unavailable, 3D visualization falls back to PCA-3D and records this in `notes`.
- Do not use `label_visible` latent as the main result, because it can leak the target label through the deposit token.

## PCA Dimension Sweep: 8-32D

Additional PCA dimension sweep outputs are saved under:

`outputs/pca_dim_sweep/`

The scan covers:

- `sample_input_token_flat`
- `sample_encoded_token_flat`
- `transformer_latent_label_masked_mean_element`

Tracked metrics are cumulative variance, kNN15 label purity, KMeans ARI, and KMeans NMI. The figures mark 12D with a vertical dashed line.

Short reading:

- `sample_input_token_flat`: 12D is mostly sufficient for kNN15 and KMeans metrics. kNN15 improves from `0.9284` at 12D to `0.9348` at 32D, but ARI/NMI do not improve meaningfully.
- `sample_encoded_token_flat`: 12D is also mostly sufficient. kNN15 improves from `0.9456` at 12D to `0.9556` at 32D, while KMeans ARI/NMI stay nearly flat.
- `transformer_latent_label_masked_mean_element`: 12D is good for kNN15, but clustering benefits from more dimensions. KMeans ARI rises from `0.2985` at 12D to a best value of `0.3460` at 24D; NMI rises from `0.4992` at 12D to `0.5232` at 24D.

Conclusion: PCA12 is not disastrously too small, but for the pooled Transformer latent, 24D looks like a better clustering-diagnostic dimension than 12D.
