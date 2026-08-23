# AGENTS.md — Element2Vec geochemistry research snapshot

## Mission

This repository preserves a staged experimental history and enables continuation on a new computer once the private datasets are supplied. It is not a package, a clean benchmark dataset, or a finished paper. Preserve negative results and confounder analyses as first-class evidence.

## Mandatory reading order

1. `README.md`
2. `EXPERIMENT_STATUS.md`
3. `docs/DATASET_SETUP.md`
4. `docs/REPRODUCIBILITY.md`
5. `stage3_6_element2vec_space_exploration/reports/element2vec_exploration_report.md`
6. `stage3_6_element2vec_space_exploration/subspace_audit/reports/subspace_exploration_report.md`
7. Relevant stage source/config files before changing an experiment.

The older reports are historical records. If a number conflicts with `EXPERIMENT_STATUS.md` or the final subspace report, prefer the later aggregate table and document the discrepancy rather than silently rewriting history.

## Data boundary — never commit

No original or processed dataset may be committed. Never add:

- `legacy_whole_rock_geochem/geochem.db` or any SQLite/database file;
- `pyrite_benchmark/data/raw/pyrite.xlsx` or another raw workbook;
- `pyrite_benchmark/data/processed/*`;
- sample metadata, sample-level values, embeddings, latents, coordinates, spaces, or caches;
- PCA fitting/evaluation pools, serialized projection bases, token reservoirs, or per-sample neighbor tables;
- secrets, credentials, local environment directories, or machine-specific absolute paths.

Before every push run:

```bash
python scripts/verify_public_tree.py --tracked
```

If a result artifact may reconstruct or expose sample records, exclude it even when it is technically a model output. Aggregate metrics, small vocabularies, static tag/label embeddings, training curves, and non-sample experiment registries are acceptable.

## Rights boundary

The repository has no open-source license. Do not add a `LICENSE` file or replace `RIGHTS.md` unless the owner explicitly requests it. Do not describe the code or weights as open source.

## Stage architecture

The stages exchange files and do not import each other as a unified package:

```text
Pyrite workbook -> Stage 1 processed tables -> Stage 1/2/3
Whole-rock SQLite view -> Stage 3.5 V2 -> checkpoint -> Stage 3.6 cache/results
```

- `pyrite_benchmark`: Stage 1 preprocessing, baseline representations, anomaly diagnostics, and dimension sweep.
- `stage2_geo_representation_xgb`: Stage 2 geology-informed features and supervised/unsupervised probes.
- `stage3_element_token_model`: Stage 3 small pyrite token-Transformer prototype.
- `stage3_5_element2vec_v2`: whole-rock Element2Vec V2 training and embedding extraction.
- `stage3_6_element2vec_space_exploration`: breadth-first token/latent audit, PCA sweep, and strict subspace replay.
- `legacy_whole_rock_geochem`: schema/view helper only; the database is private and absent.

Run all commands from the repository root. Preserve relative paths because configs depend on them.

## Environment and bootstrap

Preferred baseline is Python 3.11 from `environment.yml`. PyTorch is required for Stage 3/3.5/3.6 extraction and training. XGBoost, UMAP, Plotly, SHAP, and HDBSCAN are used by older evaluations; some scripts fall back to simpler methods when optional packages are absent.

```bash
conda env create -f environment.yml
conda activate element2vec-geochem
./scripts/fetch_checkpoints.sh
python scripts/doctor.py
```

Do not commit a local conda environment.

## Checkpoints

- The small Stage 3 checkpoint is tracked at `stage3_element_token_model/outputs/checkpoints/token_transformer_last.pt`.
- The four V2 checkpoint files are release assets downloaded by `scripts/fetch_checkpoints.sh` into `stage3_5_element2vec_v2/outputs/checkpoints/`.
- `full_model.pt` is the resumable training state and contains model, optimizer, scheduler, config, and vocab state. The component files are same-step exports, not independent training runs.
- Verify SHA-256 against `docs/checkpoint_manifest.json`.
- Do not commit V2 `.pt` files to Git or Git LFS.

## Scientific interpretation constraints

1. V2 was trained on the full current dataset with rock labels. Label probes are decodability tests, not held-out generalization.
2. For Stage 3 pyrite latents, `label_masked` is the main result. `label_visible` is leakage inspection only.
3. For V2 geological tests, use chemistry-only or explicitly all-label-masked inference unless the experiment is a declared label-space control.
4. Source, publication, token panel, and missingness are strong confounders. Always report them beside geology metrics.
5. A smooth synthetic trajectory can result from network continuity. Require natural-sample correspondence or a matched random-model/projection advantage before claiming geological semantics.
6. Two- and three-dimensional PCA plots are not evidence by themselves. Quantitative FULL/PCA/RP comparisons are mandatory.
7. Never compare vectors from separately fitted PCA spaces. Cross-space work needs one shared basis or a training-only alignment map with held-out classes.
8. Static numeric tags number only 35 vectors including special handling, so their PCA rank is at most 34; PCA64 is not applicable.
9. UMAP may visualize neighborhoods but must not be used for vector algebra, analogies, direction cosines, or process operators.
10. Existing I-series normative calculations are NNLS/CIPW-like proxies. Strict CIPW is explicitly unimplemented.

## Authoritative Stage 3.6 artifacts

- Original registry: `stage3_6_element2vec_space_exploration/reports/original_experiment_registry.csv`
- Original synthesis: `stage3_6_element2vec_space_exploration/reports/element2vec_exploration_report.md`
- Subspace compatibility: `stage3_6_element2vec_space_exploration/subspace_audit/audit/subspace_experiment_compatibility.csv`
- Dimension matrix: `stage3_6_element2vec_space_exploration/subspace_audit/reports/experiment_dimension_matrix.csv`
- Final subspace synthesis: `stage3_6_element2vec_space_exploration/subspace_audit/reports/subspace_exploration_report.md`
- Verification record: `stage3_6_element2vec_space_exploration/subspace_audit/reports/verification.json`

The released repository intentionally omits caches and per-experiment sample artifacts. Regenerate them locally from the private database.

## Experiment invariants

When replaying the subspace work, retain the original sample IDs, fitting/evaluation split, seeds, targets, endpoints, perturbation sizes, metrics, and decision thresholds. Projection is the systematic variable. Do not change a failed experiment definition after inspecting its low-dimensional result.

Required main spaces are FULL, PCA64, PCA32, PCA3, and PCA2 when the native rank allows them. Random orthogonal projections use at least ten fixed seeds. Whitening and dropped-PC analyses are explanatory controls only.

Every claimed positive result must answer:

- Is it better than FULL?
- Is it better than a matched random projection?
- Is it better than an untrained model or raw-chemistry baseline where applicable?
- Did source or missingness increase at the same time?
- Is it stable under bootstrap/resampling?
- Does the quantitative metric support the 2D/3D visual pattern?

## Current research conclusion

Do not summarize the repository as proof of a universal low-rank geochemical language. The supported conclusion is narrower: contextualization and local chemical perturbation geometry exist; one finite-difference family shows stable PCA32/PCA64 recovery; different representation levels have different effective dimensions; and nuisance structure remains substantial.

## Safe workflow for changes

1. Inspect `git status` and do not overwrite unrelated work.
2. Add a new experiment ID or preserve the original one plus a systematic suffix.
3. Keep fitting and evaluation pools separate and persist their IDs only locally.
4. Write generated sample-level output only to ignored directories.
5. Update the registry, aggregate report, `EXPERIMENT_STATUS.md`, and reproducibility notes together.
6. Run syntax/import smoke checks and the stage-specific verifier.
7. Run `python scripts/verify_public_tree.py --tracked` before commit.
8. Never push a dataset or a newly generated large checkpoint without an explicit artifact plan.

## High-priority continuation work

1. Fold-specific or locality-held-out pretraining/evaluation.
2. Nuisance-subspace attribution and removal with held-out controls.
3. Local Jacobian rank, curvature, and tangent stability across geological backgrounds.
4. Realistic analytical-panel dropout and censoring tokens.
5. Expanded REE/HFSE/LILE vocabulary.
6. Intermediate checkpoint retention and training-time semantic audits.
