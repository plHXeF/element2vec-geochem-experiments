# Element2Vec V2 checkpoints

The checkpoint binaries are GitHub Release assets, not Git-tracked files. From the repository root run:

```bash
./scripts/fetch_checkpoints.sh
```

The script downloads and verifies `full_model.pt`, `transformer_encoder.pt`, `token_projector.pt`, and `decoders.pt` into this directory.
