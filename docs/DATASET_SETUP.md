# Dataset setup

No dataset is included in this repository. A fresh clone can inspect the code, reports, vocabularies, static embeddings, and released checkpoints without data. Recomputing sample representations requires the corresponding private input.

## Whole-rock database for Element2Vec V2 and Stage 3.6

Expected path:

```text
asset://whole_rock -> <workspace>/data/whole_rock/whole-rock-v1/geochem.db
```

Expected SQLite view:

```text
v_high_quality_dataset
```

The default V2 configuration reads 34 numerical columns plus `sample_id`, `rock_group`, `rock_origin`, and `rock_type`. The exact mapping is in `stage3_5_element2vec_v2/configs/v2_base.yaml`.

Check the database without exporting rows:

```bash
python scripts/doctor.py --check-schema
```

If the research view is missing, read `legacy_whole_rock_geochem/research_views_documentation.md`. `create_research_views.py` copies the registered database into the explicit run, then creates views in that copy. Register a new data version before adopting it; never mutate a frozen input.

## Pyrite workbook for Stages 1–3

Expected path:

```text
asset://pyrite_raw -> <workspace>/data/pyrite/pyrite-v1/raw/pyrite.xlsx
```

Generate the processed inputs once:

```bash
python scripts/experiment.py new-run E001
python scripts/experiment.py exec --run-dir /path/returned/above -- pyrite_benchmark/scripts/prepare_pyrite_dataset.py
```

This writes sample-level CSV files to the run’s `prepared_data/`. To adopt a revision, register a new version in `data/pyrite/`, with hashes and the preparation run ID, then map `asset://pyrite_processed` in `workspace.local.json`. Current stages read the frozen processed input rather than reparsing the workbook. Local paths are not committed; logical asset names in configurations are stable across machines.

## Expected historical scale

These counts are checks, not filters:

- Pyrite line: 7,778 samples, 14 selected elements, 6 standardized deposit types.
- Whole-rock V2 training view: approximately 270,241 samples at the time of the released run.
- V2 vocabulary: 34 numeric tags and 96 pure-label tokens in the released checkpoint.

Counts may differ if the private source is revised. Record database/workbook hashes, view counts, schema differences, and the date of access before rerunning. Never alter code merely to force a new dataset to match the historical count.

## Data governance

- Confirm access rights and citation requirements before use.
- Keep raw and processed files outside Git history.
- Do not publish sample IDs, coordinates, source rows, or reconstructable sample-level embeddings.
- If sharing aggregate results, run `python scripts/verify_public_tree.py --tracked` before pushing.
