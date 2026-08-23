# Stage 2: Geo Representation XGB Benchmark

This directory is an independent experiment stage. It does not overwrite any
previous pyrite benchmark outputs.

## Data input choice

The stage reuses the processed pyrite files from:

- `../pyrite_benchmark/data/processed/pyrite_metadata.csv`
- `../pyrite_benchmark/data/processed/pyrite_element_values.csv`
- `../pyrite_benchmark/data/processed/pyrite_observed_mask.csv`
- `../pyrite_benchmark/data/processed/pyrite_numeric_mask.csv`

These files are preferred because they are the cleaned Stage 1 outputs. The raw
Excel file is not read by this stage.

The selected 14 core elements are loaded from
`configs/stage2_config.json`, matching the Stage 1 selected element file:
`../pyrite_benchmark/dimension_sweep/selected_elements.csv`.

## Run

Use the project conda environment:

```powershell
E:\Geochem\.conda\envs\geochem-benchmark\python.exe E:\Geochem\stage2_geo_representation_xgb\run_stage2.py
```

## Outputs

All outputs are written under `stage2_geo_representation_xgb/outputs/`.

Main files:

- `outputs/representations/representation_qc.csv`
- `outputs/unsupervised/unsupervised_metrics.csv`
- `outputs/pca_sweep/pca_dimension_sweep_metrics.csv`
- `outputs/umap3d/umap3d_coordinates.csv`
- `outputs/xgb/xgb_classification_metrics.csv`
- `outputs/xgb/feature_importance_permutation.csv`
- `outputs/xgb/feature_importance_xgb_gain.csv`
- `outputs/stage2_report.md`
