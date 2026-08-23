# Reproducibility

- Environment: `/opt/anaconda3/envs/Element2Vec/bin/python`.
- Seed: 42; random projection seeds: 1–10.
- Main PCA spaces are nested prefixes of a single fixed basis per representation.
- PCA fitting IDs, evaluation IDs, and token fitting rows are saved under `projections/fitting_pools/`.
- Historical limitation: only D1 had non-empty saved sample IDs; all other legacy experiments are deterministic code replays using the original seed and fixed definitions.
- No zero padding or duplicated dimensions. Native/rank failures are `Not applicable`.
- UMAP is unavailable and was not used; no vector algebra is performed in nonlinear visualization coordinates.
- Strict CIPW remains unavailable; NNLS normative proxy is not renamed.

Commands:

```bash
python -m stage3_6_element2vec_space_exploration.src.subspace_audit --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml
python -m stage3_6_element2vec_space_exploration.src.run_subspace_replay --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml --phase main
python -m stage3_6_element2vec_space_exploration.src.run_subspace_replay --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml --phase random
python -m stage3_6_element2vec_space_exploration.src.run_subspace_replay --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml --phase aux
python -m stage3_6_element2vec_space_exploration.src.subspace_specialty --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml
python -m stage3_6_element2vec_space_exploration.src.build_subspace_report --config stage3_6_element2vec_space_exploration/configs/subspace_audit.yaml
```
