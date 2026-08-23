# Checkpoints

## Distribution

The small Stage 3 prototype checkpoint is tracked in Git:

```text
stage3_element_token_model/outputs/checkpoints/token_transformer_last.pt
```

Element2Vec V2 checkpoints are attached to GitHub Release `v0.1.0-stage3.6` and are downloaded with:

```bash
./scripts/fetch_checkpoints.sh
```

Destination:

```text
stage3_5_element2vec_v2/outputs/checkpoints/
```

The two small files are direct Release assets. The full model and Transformer encoder are stored as ordered 64 MiB parts because large single-request uploads were unreliable on the publication connection. The fetch script downloads the parts and reconstructs the original files automatically.

## Files

- `full_model.pt`: complete step-13,200 training state, including model, optimizer, scheduler, config, and vocabulary state. Use this for resuming or exact model reconstruction.
- `transformer_encoder.pt`: same-step Transformer encoder state.
- `token_projector.pt`: same-step shared token-projector state.
- `decoders.pt`: same-step task-head state.

The component files are convenience exports, not four independently trained models.

## Integrity

Checksums, byte sizes, and ordered asset names are recorded in `docs/checkpoint_manifest.json`. The fetch script verifies each reconstructed file before moving it into place. If an existing file has the wrong checksum, the script stops unless `--force` is supplied.

PyTorch checkpoints use Python/pickle-based serialization. Load only the files published by this repository and verify checksums first. Prefer `map_location="cpu"` when auditing on an unknown machine.

## Reproduction boundary

The release provides the final state only. There are no intermediate epoch checkpoints, so the following cannot be reconstructed from the release:

- checkpoint-selection curves;
- temporal stability of semantic relations;
- whether a relation appeared and later disappeared during training.

The next training run should retain a sparse series of intermediate checkpoints and attach dataset/view hashes to each run manifest.
