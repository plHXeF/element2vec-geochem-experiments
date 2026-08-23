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
