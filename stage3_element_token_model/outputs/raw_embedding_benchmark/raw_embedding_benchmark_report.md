# Raw Element Token Embedding Benchmark

## What Was Added

This benchmark evaluates the unprojected sample-level token matrix:

`[x_i, visible_mask_i, learned_element_embedding_i]`

for each of the 14 elements. The resulting feature dimension is:

`14 * (1 + 1 + 32) = 476`

This is different from the previous `sample_input_token_flat.csv`, which used the projected Transformer input token:

`14 * d_model = 14 * 64 = 896`

## Benchmark Summary

| space_name | n_features | xgb_accuracy_mean | xgb_accuracy_std | xgb_macro_f1_mean | xgb_macro_f1_std | xgb_weighted_f1_mean | xgb_weighted_f1_std | knn15_accuracy_mean | knn15_accuracy_std | knn15_macro_f1_mean | knn15_macro_f1_std | knn15_weighted_f1_mean | knn15_weighted_f1_std | knn15_neighborhood_purity | kmeans_ari | kmeans_nmi | kmeans_homogeneity | kmeans_completeness | silhouette_score | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_raw_element_token_flat_476 | 476 | 0.9638724902669059 | 0.0038127258521919714 | 0.9353234468431886 | 0.010780356702221628 | 0.9637131830562626 | 0.00387368499325644 | 0.9502453318344507 | 0.004590100988188564 | 0.9149605794975606 | 0.01377362121489652 | 0.9499572576370221 | 0.004685229705107682 | 0.9375246421530814 | 0.03641742762331844 | 0.28392165474307146 | 0.3256570588609912 | 0.2516684481790305 | 0.1416938751935959 | ok |


## Outputs

- `embeddings/sample_raw_element_token_flat.csv`
- `embeddings/deposit_type_embedding_nomask.csv`
- `metrics/raw_element_token_benchmark_summary.csv`
- `figures/sample_raw_element_token_flat_476_umap3d.html`
- `figures/deposit_type_embedding_nomask_umap3d.html`

## Embedding Learning Audit

# Embedding Learning Audit

## What The Code Does

- `model.py` defines `self.element_embedding = nn.Embedding(n_elements, element_embedding_dim)`.
- `model.py` defines `self.deposit_embedding = nn.Embedding(n_deposit_types + 1, deposit_embedding_dim)`.
- No fixed `_weight` is passed into either embedding layer, so PyTorch initializes these lookup tables randomly during model construction.
- `train.py` builds the model after `set_seed(random_seed)`, then optimizes `model.parameters()` with AdamW. This includes `element_embedding.weight` and `deposit_embedding.weight`.
- In `make_tokens`, element embeddings are concatenated with abundance and visible mask before projection:

```python
elem_input = torch.cat([x.unsqueeze(-1), visible_mask.unsqueeze(-1), elem_emb], dim=-1)
```

## Learned vs Deterministic Initial Weights

To verify the embeddings changed during training, this audit reconstructs a fresh model with the same seed and architecture, then compares its initial embedding weights with the checkpoint weights.

               parameter shape  requires_grad  l2_delta_from_deterministic_initialization  mean_abs_delta  max_abs_delta  current_mean  current_std  initial_mean  initial_std
element_embedding.weight 14x32           True                                    3.490182        0.130326       0.562063      0.037407     0.847983      0.040124     0.963305
deposit_embedding.weight  7x32           True                                    3.038790        0.157887       0.611239     -0.035862     1.003785     -0.037259     1.002907

Interpretation: non-zero deltas confirm the embedding lookup matrices were randomly initialized and then updated by training.
