# Stage 2 Geo Representation XGB Report

## 1. Data input and reuse

Processed data directory: `E:\Geochem\pyrite_benchmark\data\processed`

Files reused:

- `pyrite_metadata.csv`
- `pyrite_element_values.csv`
- `pyrite_observed_mask.csv`
- `pyrite_numeric_mask.csv`

The raw Excel file was not re-cleaned. This stage uses the cleaned Stage 1
processed tables, because they already contain normalized labels and numeric
element matrices.

Samples: 7778

Target label: `type`

Elements (14): As, Ni, Cu, Pb, Co, Au, Ag, Bi, Sb, Zn, Se, Te, Tl, Mn

Class counts:

| label | count |
| --- | --- |
| orogenic | 4799 |
| sedimentary | 1421 |
| porphyry | 658 |
| carlin | 487 |
| ni-cu-pge | 263 |
| vms | 150 |


## 2. Representation definitions

Representation QC:

| representation | n_features | raw_missing_rate | final_missing_rate |
| --- | --- | --- | --- |
| raw_robust_z_14 | 14 | 0.20261359879513646 | 0.0 |
| log_robust_z_14 | 14 | 0.20261359879513646 | 0.0 |
| rank_uniform_z_14 | 14 | 0.0 | 0.0 |
| geo_ratio_12 | 12 | 0.24470729407731207 | 0.0 |
| geo_balance_12 | 12 | 0.018374475014999572 | 0.0 |
| geo_info_12 | 12 | 0.0 | 0.0 |


All log transforms use element-specific eps values equal to half of the minimum
positive concentration for that element, with `1e-9` as fallback. Constructed
features are median-imputed and robust-scaled before benchmarking. Rank baseline
uses percentile ranks and standard scaling, following Stage 1.

Feature formulas and group definitions are saved to:
`outputs/representations/feature_definitions.csv`

## 3. Unsupervised input-space benchmark

Main table: `outputs/unsupervised/unsupervised_metrics.csv`

Best input kNN15:

- rank_uniform_z_14: input_knn15=0.9172
- log_robust_z_14: input_knn15=0.9084
- geo_ratio_12: input_knn15=0.8631
- geo_balance_12: input_knn15=0.8545
- raw_robust_z_14: input_knn15=0.7573
- geo_info_12: input_knn15=0.7520

Best input kNN30:

- rank_uniform_z_14: input_knn30=0.8923
- log_robust_z_14: input_knn30=0.8773
- geo_ratio_12: input_knn30=0.8318
- geo_balance_12: input_knn30=0.8201
- raw_robust_z_14: input_knn30=0.7198
- geo_info_12: input_knn30=0.7184

KMeans/GMM ARI and NMI remain diagnostic rather than final truth, because
deposit-type labels can overlap geochemically and class sizes are imbalanced.
Silhouette is computed using the true `type` label; Davies-Bouldin and
Calinski-Harabasz are computed on KMeans clusters.

HDBSCAN notes:

| representation | hdbscan_note |
| --- | --- |
| raw_robust_z_14 | ok |
| log_robust_z_14 | ok |
| rank_uniform_z_14 | ok |
| geo_ratio_12 | ok |
| geo_balance_12 | ok |
| geo_info_12 | ok |


## 4. PCA dimension sweep

PCA is used only as a linear diagnostic. Cumulative explained variance is not
equivalent to geologic structure quality.

Main table: `outputs/pca_sweep/pca_dimension_sweep_metrics.csv`

Approximate kNN15 plateau dimensions:

| representation | pca_knn15_plateau_dim |
| --- | --- |
| raw_robust_z_14 | not found |
| log_robust_z_14 | 9 |
| rank_uniform_z_14 | not found |
| geo_ratio_12 | 9 |
| geo_balance_12 | 8 |
| geo_info_12 | 9 |


## 5. UMAP-3D visualization

Coordinates: `outputs/umap3d/umap3d_coordinates.csv`

Metrics: `outputs/umap3d/umap3d_metrics.csv`

UMAP trustworthiness and neighborhood purity:

| representation | trustworthiness | umap_knn15 | umap_knn30 | note |
| --- | --- | --- | --- | --- |
| raw_robust_z_14 | 0.9801482820302951 | 0.7078426330676266 | 0.683106196965801 | ok |
| log_robust_z_14 | 0.9809521297457265 | 0.8958858318333762 | 0.8712822490785977 | ok |
| rank_uniform_z_14 | 0.9879434735133569 | 0.9076540670266564 | 0.8895345847261507 | ok |
| geo_ratio_12 | 0.9769240447786045 | 0.8533470472272221 | 0.8291505956972658 | ok |
| geo_balance_12 | 0.9893503998808103 | 0.8274363589611725 | 0.8032570497985771 | ok |
| geo_info_12 | 0.9946601094624707 | 0.7105682694780149 | 0.6899888574612154 | ok |

Interactive Plotly status:

- ok: raw_robust_z_14_umap3d_interactive.html
- ok: log_robust_z_14_umap3d_interactive.html
- ok: rank_uniform_z_14_umap3d_interactive.html
- ok: geo_ratio_12_umap3d_interactive.html
- ok: geo_balance_12_umap3d_interactive.html
- ok: geo_info_12_umap3d_interactive.html

UMAP-3D coordinates are not ternary data. They are unconstrained latent
coordinates, so no ternary plot is produced for them.

## 6. XGBoost classification

Metrics: `outputs/xgb/xgb_classification_metrics.csv`

Per-class F1: `outputs/xgb/per_class_f1.csv`

Confusion matrices in long form: `outputs/xgb/confusion_matrix_long.csv`

Stratified CV ranking by macro-F1:

| representation | model | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | note |
| --- | --- | --- | --- | --- | --- | --- |
| raw_robust_z_14 | xgboost | 0.9607868346618668 | 0.9212604987994292 | 0.931277096436641 | 0.9605017920349597 |  |
| log_robust_z_14 | xgboost | 0.9607868346618668 | 0.9212604987994292 | 0.931277096436641 | 0.9605017920349597 |  |
| rank_uniform_z_14 | xgboost | 0.9609154024170738 | 0.9209973663390715 | 0.9309978076970208 | 0.9606391970611695 |  |
| geo_ratio_12 | xgboost | 0.9213165338133196 | 0.8152966291361695 | 0.845983419057058 | 0.919725100574056 |  |
| geo_balance_12 | xgboost | 0.8864746721522242 | 0.7476299855705791 | 0.7930422388644799 | 0.8800330178918706 |  |
| geo_info_12 | xgboost | 0.8400617125224994 | 0.6957989910088442 | 0.7356695268426576 | 0.8346822723939917 |  |

Class filtering:

- minimum class count: 30
- samples used: 7778
- group column tried: citation

GroupKFold by citation/source is stricter than random StratifiedKFold and is
included when a usable metadata column is available. It is a first guard against
literature-source leakage, but later stages should add deposit- or district-level
grouping if those metadata become available.

## 7. Feature importance

Permutation importance: `outputs/xgb/feature_importance_permutation.csv`

XGBoost gain: `outputs/xgb/feature_importance_xgb_gain.csv`

SHAP summary: `outputs/xgb/shap_values_summary.csv` when available.

Top permutation features:

- raw_robust_z_14: Au (0.171), Ni (0.117), Sb (0.082), Se (0.070), Te (0.067), Bi (0.066)
- log_robust_z_14: Au (0.171), Ni (0.117), Sb (0.082), Se (0.070), Te (0.067), Bi (0.066)
- rank_uniform_z_14: Au (0.172), Ni (0.114), Sb (0.083), Se (0.069), Te (0.066), Bi (0.065)
- geo_ratio_12: log_Cu_Pb (0.145), log_Ag_Au (0.121), log_Sb_Tl (0.120), log_As_Se (0.119), log_Co_Ni (0.116), log_Au_As (0.112)
- geo_balance_12: Au_enrichment_balance (0.206), semimetal_vs_lattice (0.178), lateLowT_vs_highT (0.151), SeTeBi_vs_AsSbTl (0.148), AsSbTl_vs_CoNi (0.139), highT_vs_gold (0.115)
- geo_info_12: highT_lattice_info_share (0.254), gold_pathfinder_info_share (0.229), base_metal_info_share (0.229), rarity_mean (0.228), rarity_max (0.118), semimetal_telluride_info_share (0.073)

Importance notes:

- no warnings

Geologic reading should focus on stable signals that are both important in the
supervised task and reasonable under unsupervised structure. Features that mostly
separate source/citation groups in GroupKFold should be treated cautiously.

## 8. Next-stage `geo_fusion_12` suggestion

Candidate features selected from Stage 2 importance and unsupervised behavior:

- log_Cu_Pb
- log_Ag_Au
- log_Sb_Tl
- log_As_Se
- Au_enrichment_balance
- semimetal_vs_lattice
- lateLowT_vs_highT
- SeTeBi_vs_AsSbTl
- highT_lattice_info_share
- gold_pathfinder_info_share
- base_metal_info_share
- rarity_mean

This is only a candidate list. It should be validated in a new experiment stage
without overwriting this stage.
