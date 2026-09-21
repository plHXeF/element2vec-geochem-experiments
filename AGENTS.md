# AGENTS.md — Element2Vec geochemistry research snapshot

## Mission

This repository preserves a staged experimental history and enables continuation on a new computer once the private datasets are supplied. It is not a package, a clean benchmark dataset, or a finished paper. Preserve negative results and confounder analyses as first-class evidence.

## Mandatory reading order

1. `README.md` and `docs/WORKSPACE_MANAGEMENT.md`.
2. `experiments/README.md` and the target `experiments/E00N/experiment.json`.
3. `docs/EXPERIMENT_COMMANDS.md`, `docs/DATASET_SETUP.md`, `docs/REPRODUCIBILITY.md`.
4. The experiment card's reports, protocol and source/config files.

Experiment cards are the current authority. `EXPERIMENT_STATUS.md` and the stage reports preserve historical evidence. Preserve conflicting evidence and document discrepancies; never rewrite historical conclusions to fit new results.

## Required daily workflow

Register experiment → modify and commit code → create run → execute/check → update experiment card → merge reproducible implementation → freeze run. Status and scientific outcome are separate; computation failure is not a scientific negative result.

- One permanent E ID per research question. Keep A–K/X sub-experiments. Stage directories remain modules, not variant backups.
- Use `experiment/eNNN-topic`, `fix/topic`, `chore/topic` branches and separate worktrees for concurrent implementations. A parameter/seed change needs a run, not a copied stage or a branch.
- Run `python scripts/experiment.py list` to discover registered runs. Edit the authoritative card and regenerate `index`; do not hand-edit the derived total table.
- Formal runs must start from clean committed code. Use `new-run E00N --scratch` for debug work; it preserves the patch and untracked source. Never backfill a historical run with today's commit.
- All new outputs belong to the workspace's `runs/E00N/<UTC-seed-random>/`. Every writing entry point requires `--run-dir`. Use `scripts/experiment.py exec` for command logs and status; all steps share the same directory.
- Configs use `asset://`, `run://`, `repo://`; local paths belong only in `workspace.local.json`. Pin explicit upstream runs, data versions and checkpoint checksums. `latest` is navigation only.
- Use `check --rehash`, write a result/limitation summary, then `freeze` with actual algorithms and device. Frozen runs cannot be resumed. A scientific configuration change needs a new run; resume only supported, validated steps in a mutable run.
- Keep original cache configurations/hashes; migration maps locations instead of rewriting history. Unknown provenance stays `unknown`.
- `data/`, frozen runs and `archive/` are immutable inputs. New preprocessing writes `prepared_data/` in its run; promote reviewed output as a new data version. Pyrite stages 2/3 read processed Stage 1 tables.
- Preserve the same scientific controls when relocating paths. Optional fallback implementations must be named as actually used.
- Deliverables record source run IDs in provenance. Keep generation source in `presentations/`; create intermediates in a new run and a new final delivery directory.
- Check the Git status before edits. Stage only intended source/aggregate changes, run relevant tests and the content audit before commit. Do not auto-push, publish, clear historical results, or launch long training.

## Data boundary — never commit

No original or processed dataset may be committed. Never add:

- `legacy_whole_rock_geochem/geochem.db` or any SQLite/database file;
- `pyrite_benchmark/data/raw/pyrite.xlsx` or another raw workbook;
- `pyrite_benchmark/data/processed/*`;
- sample metadata, sample-level values, embeddings, latents, coordinates, spaces, or caches;
- PCA fitting/evaluation pools, serialized projection bases, token reservoirs, or per-sample neighbor tables;
- secrets, credentials, local environment directories, or machine-specific absolute paths.

After staging and before every commit or push run:

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

Run commands from the repository/worktree root. Use logical asset IDs and the shared run context; never write new results into historical stage output folders. `latent_evaluation_benchmark` is E009. `experiments/`, `reports/`, `presentations/`, `docs/`, and `scripts/` are shared management modules.

## Environment, checks and checkpoints

Python 3.11 is specified in `environment.yml`. The local usable interpreter is recorded in workspace settings. Archived Windows binaries are not an environment. PyTorch is required for model work; optional package fallbacks must be reported with their real algorithm names. Do not commit conda environments or local absolute paths.

Relevant checks (there are tests):

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s stage3_5_element2vec_v2/tests -p 'test_*.py'
python -m unittest discover -s stage3_6_element2vec_space_exploration/tests -p 'test_*.py'
python -m unittest discover -s latent_evaluation_benchmark/tests -p 'test_*.py'
python scripts/verify_public_tree.py --tracked
```

The original small Stage 3 checkpoint remains tracked as history. New weights, including V2/RoPE, are local run artifacts. `asset://v2_checkpoint`, `asset://rope_z_checkpoint`, and `asset://geo15_checkpoint` resolve frozen inputs. Component exports from the same training step are not separate runs. `docs/checkpoint_manifest.json` preserves released hashes. Checkpoint downloads require an explicit mutable run and are never automatic.

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

## Changes to evidence

Keep fitting/evaluation pools separate, persist IDs only locally, and preserve seeds, targets and controls. Record new evidence in the experiment card and a new report, regenerate the total index, and cite the new run. Existing snapshot reports remain unchanged. Run relevant syntax/import checks, tests, and stage verifiers before merging.

## High-priority continuation work

1. Fold-specific or locality-held-out pretraining/evaluation.
2. Nuisance-subspace attribution and removal with held-out controls.
3. Local Jacobian rank, curvature, and tangent stability across geological backgrounds.
4. Realistic analytical-panel dropout and censoring tokens.
5. Expanded REE/HFSE/LILE vocabulary.
6. Intermediate checkpoint retention and training-time semantic audits.
