# Stage 3.6: Element2Vec V2 Representation-Space Exploration

This directory is an independent, breadth-first audit of the trained
`stage3_5_element2vec_v2` checkpoint. It reuses the V2 model and vocabulary but
does not write into or modify Stage 3.5.

The exploration follows four rules:

1. high-dimensional metrics before low-dimensional pictures;
2. chemistry-only inference for geological label checks;
3. positive, negative, confounded, and technically failed experiments are all
   retained in the registry;
4. the existing model is analyzed as-is; no large retraining is performed.

## Run order

Use the `Element2Vec` conda environment from the repository root:

```bash
/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.audit_project \
  --config stage3_6_element2vec_space_exploration/configs/exploration.yaml

/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.extract_cache \
  --config stage3_6_element2vec_space_exploration/configs/exploration.yaml

/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.run_experiments \
  --config stage3_6_element2vec_space_exploration/configs/exploration.yaml

/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.build_reports \
  --config stage3_6_element2vec_space_exploration/configs/exploration.yaml

/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.verify_outputs \
  --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
```

All generated files remain under this directory:

```text
audit/       model/data audit and available-representation manifest
cache/       reproducible shared embedding caches
results/     experiment registry and per-experiment evidence
reports/     final synthesis and supporting indexes
```

## Nested PCA dimension sweep

The full Stage 3.6 registry can be recomputed in nested 256/128/64/32/16/8/5/3/2D
PCA spaces without rerunning the Transformer for every dimension:

```bash
/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.run_pca_sweep \
  --config stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml

/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.build_pca_sweep_report \
  --config stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml

/opt/anaconda3/envs/Element2Vec/bin/python \
  -m stage3_6_element2vec_space_exploration.src.verify_pca_sweep \
  --config stage3_6_element2vec_space_exploration/configs/pca_sweep.yaml
```

Outputs are isolated under `pca_sweep/`. PCA bases are fit once at maximum rank
and truncated, so all requested subspaces are nested and resumable.
