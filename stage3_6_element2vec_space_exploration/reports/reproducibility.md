# Reproducibility

## Source identity

```json
{
  "sample_count": 12000,
  "sample_id_sha256": "0c2d1e0466dbb141af4204b9e8c71826ccc4191608e7dddd0041f38a108a4ae7",
  "seed": 42,
  "checkpoint": "stage3_5_element2vec_v2/outputs/checkpoints/full_model.pt",
  "checkpoint_sha256": "1d9210c0db29f243193df23bd434e98d169a1fe112e87c8809b4b3716397bd1e",
  "checkpoint_step": 13200,
  "device": "cpu",
  "batch_size": 128,
  "token_reservoir_rows": 34000,
  "numeric_fields": [
    "SiO2",
    "TiO2",
    "Al2O3",
    "Fe2O3",
    "FeO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
    "MnO",
    "LOI",
    "CO2",
    "Rb_ppm",
    "Sr_ppm",
    "Ba_ppm",
    "Zr_ppm",
    "Nb_ppm",
    "Y_ppm",
    "V_ppm",
    "Cr_ppm",
    "Ni_ppm",
    "Cu_ppm",
    "Zn_ppm",
    "La_ppm",
    "Ce_ppm",
    "Nd_ppm",
    "Th_ppm",
    "U_ppm",
    "Pb_ppm",
    "Sc_ppm",
    "Co_ppm",
    "Ga_ppm",
    "Yb_ppm"
  ],
  "elapsed_seconds": 249.56974649999756,
  "files": [
    "latents_masked_labels.npz",
    "latents_numeric_only.npz",
    "latents_random_numeric_only.npz",
    "latents_visible_labels_control.npz",
    "numeric_embedding_names.csv",
    "pure_embedding_names.csv",
    "sample_metadata.csv",
    "static_embeddings_random.npz",
    "static_embeddings_trained.npz",
    "token_reservoir_metadata.csv",
    "token_reservoir_numeric_only.npz"
  ]
}
```

## Environment

- Python: `/opt/anaconda3/envs/Element2Vec/bin/python`
- Primary packages: numpy, pandas, scipy, scikit-learn, torch, matplotlib
- UMAP/Plotly/Parquet dependencies were not required.
- Main inference used `cpu` and completed in 249.6 seconds.
- Random seed: 42

## Commands

```bash
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.audit_project --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.extract_cache --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.run_experiments --config stage3_6_element2vec_space_exploration/configs/exploration.yaml --batch all
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.build_reports --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.verify_outputs --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
```

Each experiment records its effective configuration, samples, metrics,
observations, figures, and errors. The registry is upserted by experiment ID, so
reruns replace one experiment rather than duplicate it.
