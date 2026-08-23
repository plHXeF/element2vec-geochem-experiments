# Pyrite Representation Benchmark

## Scope

- Samples: `7778`
- Grouping label: `type`
- Selected elements: `14` with numeric coverage >= `20%`
- Elements: `As`, `Ni`, `Cu`, `Pb`, `Co`, `Au`, `Ag`, `Bi`, `Sb`, `Zn`, `Se`, `Te`, `Tl`, `Mn`
- PCA was computed with `sklearn.decomposition.PCA`.
- UMAP was computed with `n_neighbors=30`, `min_dist=0.1`.

## Representation Metrics

| representation | features | PC1+PC2 | input kNN15 | PCA kNN15 | UMAP kNN15 | UMAP trust | kmeans ARI | kmeans NMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `log_plus_mask` | 28 | 0.229 | 0.937 | 0.797 | 0.918 | 0.989 | 0.216 | 0.374 |
| `rank_uniform_z` | 14 | 0.428 | 0.917 | 0.599 | 0.902 | 0.977 | 0.100 | 0.277 |
| `log_robust_z` | 14 | 0.506 | 0.908 | 0.600 | 0.885 | 0.964 | 0.133 | 0.246 |
| `winsor_log_robust_z` | 14 | 0.507 | 0.908 | 0.598 | 0.882 | 0.964 | 0.115 | 0.254 |
| `raw_robust_z` | 14 | 0.756 | 0.757 | 0.567 | 0.697 | 0.972 | -0.003 | 0.005 |
| `observed_mask_z` | 14 | 0.295 | 0.617 | 0.617 | 0.665 | 0.877 | 0.026 | 0.277 |

## First Reading

- `input kNN15` asks whether local neighborhoods in the full representation space preserve `type` labels.
- `PCA kNN15` and `UMAP kNN15` ask the same question after 2D projection.
- `trustworthiness` measures how well the 2D embedding preserves local neighborhoods from the full representation.
- `kmeans ARI/NMI` asks whether unsupervised clusters resemble the known `type` labels.
- The observation-mask representation is included because missingness and censoring can themselves create structure.

## Outputs

- `representation_metrics_latest.csv`: benchmark metrics
- `selected_elements.csv`: elements used in this first pass
- `representation_clusters.csv`: sample clusters in the full representation space
- `embeddings/*_pca.csv`: sample-level PCA coordinates and clusters
- `embeddings/*_umap.csv`: sample-level UMAP coordinates and clusters
- `figures/*_pca.png` and `figures/*_pca.svg`: PCA scatter plots colored by `type`
- `figures/*_umap.png` and `figures/*_umap.svg`: UMAP scatter plots colored by `type`
- `figures/raw_robust_z_pca_central_zoom.*`: percentile zoom for the raw PCA plot dominated by extreme values
