# Latent Dimension Sweep Report

## Scope

- PCA and UMAP latent dimensions were swept independently for each representation.
- Main 14-feature representations were scanned from 1 to 12 dimensions.
- `log_plus_mask` has 28 features and was scanned from 1 to 20 dimensions as a control.
- UMAP parameters were fixed at `n_neighbors=30`, `min_dist=0.1`, `random_state=42`.
- kNN15 means: for each sample, find 15 nearest-neighbor samples in the current latent space, compute the fraction with the same `type`, then average over samples.
- Plateau rule: first dimension `d` where gains from `d -> d+1` and the following two gains are all `< 0.005`.

## Plateau Summary

- `raw_robust_z` / `pca`: plateau_dim=`not found`, plateau_knn15=`NA`; max=`0.743` at dim `10`.
- `raw_robust_z` / `umap`: plateau_dim=`3`, plateau_knn15=`0.708`; max=`0.714` at dim `11`.
- `log_robust_z` / `pca`: plateau_dim=`9`, plateau_knn15=`0.894`; max=`0.904` at dim `12`.
- `log_robust_z` / `umap`: plateau_dim=`3`, plateau_knn15=`0.896`; max=`0.900` at dim `12`.
- `winsor_log_robust_z` / `pca`: plateau_dim=`not found`, plateau_knn15=`NA`; max=`0.904` at dim `12`.
- `winsor_log_robust_z` / `umap`: plateau_dim=`3`, plateau_knn15=`0.897`; max=`0.901` at dim `7`.
- `rank_uniform_z` / `pca`: plateau_dim=`not found`, plateau_knn15=`NA`; max=`0.914` at dim `12`.
- `rank_uniform_z` / `umap`: plateau_dim=`3`, plateau_knn15=`0.908`; max=`0.911` at dim `8`.
- `observed_mask_z` / `pca`: plateau_dim=`1`, plateau_knn15=`0.617`; max=`0.617` at dim `1`.
- `observed_mask_z` / `umap`: plateau_dim=`1`, plateau_knn15=`0.663`; max=`0.665` at dim `2`.
- `log_plus_mask` / `pca`: plateau_dim=`8`, plateau_knn15=`0.859`; max=`0.929` at dim `20`.
- `log_plus_mask` / `umap`: plateau_dim=`3`, plateau_knn15=`0.925`; max=`0.931` at dim `9`.

## Short Interpretation Checklist

- Low-dimensional PCA plateaus: log=`9`, rank=`not found`, winsor=`not found`.
- Low-dimensional UMAP plateaus: log=`3`, rank=`3`, winsor=`3`.
- PCA: `log_robust_z` reaches a plateau around 9D; `rank_uniform_z` and `winsor_log_robust_z` keep improving through 12D, so their PCA minimum-sufficient dimension is not resolved by this scan.
- UMAP: log/rank/winsor representations all enter a local-neighborhood plateau around 3D; this suggests 2D is useful for visualization, but 3D is a stronger minimum-sufficient UMAP latent space for quantitative neighborhood metrics.
- `raw_robust_z` remains substantially worse than log/rank/winsor even as dimensions increase, supporting its role as a baseline/negative control.
- `observed_mask_z` plateaus immediately because it mostly captures reporting/missingness patterns, not concentration structure.
- `log_plus_mask` is a control: it gives the highest neighborhood purity, but part of that gain likely comes from explicit reporting-pattern information.

## Outputs

- `dimension_sweep_metrics.csv`
- `dimension_sweep_metrics_with_marginal_gain.csv`
- `plateau_dimension_summary.csv`
- `pca_cumulative_variance.csv`
- `figures/*_metric_vs_dim.png/.svg`
- `figures/*_representation_comparison.png/.svg`
