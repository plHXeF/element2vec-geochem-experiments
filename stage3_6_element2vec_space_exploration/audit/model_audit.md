# Element2Vec V2 model audit

## Source artifacts

- Model config: `stage3_5_element2vec_v2/configs/v2_base.yaml`
- Selected checkpoint: `stage3_5_element2vec_v2/outputs/checkpoints/full_model.pt`
- Available checkpoints:

| file | size_mb |
| --- | --- |
| stage3_5_element2vec_v2/outputs/checkpoints/decoders.pt | 0.379659 |
| stage3_5_element2vec_v2/outputs/checkpoints/full_model.pt | 993.545 |
| stage3_5_element2vec_v2/outputs/checkpoints/token_projector.pt | 6.02466 |
| stage3_5_element2vec_v2/outputs/checkpoints/transformer_encoder.pt | 324.507 |

## Tokenization and architecture

- Numeric tags: 34 observed-only fields plus one masked-tag index.
- Pure-label tokens: 96 field-value tokens across rock_group, rock_origin, rock_type.
- Raw token: 509 label dimensions + value + tag-mask flag + value-mask flag = 512 dimensions.
- Token projector: shared MLP from 512 to 768 dimensions.
- Transformer: 12 pre-LN layers, 12 heads, FFN 3072, no positional embedding.
- CLS: direct learned 768-dimensional parameter.
- Training masking: one masked target per sample; numeric value, numeric tag, or one pure label.

## Extractable intermediate representations

The trained model exposes static embeddings, MLP-projected input tokens,
Transformer contextual tokens, CLS, contextual mean-numeric pooling, and an
additional pre-Transformer mean-numeric control. These layers are not treated
as interchangeable coordinate systems.

## Existing validation and visualization

- `train_log.csv` and fixed-train `probe_log.csv` exist.
- Stage 3.5 contains nearest-neighbor inspection and a latent benchmark script.
- Existing latent/metrics directories are empty; no completed V2 latent audit exists.
- Existing probes are not held-out validation and do not establish generalization.

## Leakage-relevant behavior

The checkpoint was trained on all samples with pure labels available during
most numeric masking tasks and as prediction targets during label masking.
Masking or removing labels at inference prevents direct input leakage but does
not make downstream fixed-latent cross-validation an independent generalization
test. All such results must be described as representation decodability.
