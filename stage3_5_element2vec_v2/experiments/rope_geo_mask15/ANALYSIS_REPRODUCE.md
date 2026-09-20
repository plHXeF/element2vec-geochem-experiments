# Reproduce the geo15 post-training audit

Run from the repository root in the Element2Vec environment. These commands do
not train. Required private inputs: the finished geo checkpoint, original V2 and
RoPE-z checkpoints, the original RoPE-z audit cache and protocol, the whole-rock
database, and original Stage 3.6 raw mixing/melting CSVs.

The paired audit directory was frozen before training. On a new machine, prepare
it once with the README's prepare_geo_comparison command; never overwrite its IDs
or fitting/evaluation pools. The baseline's four older extraction records predate
per-file provenance fields. Verify frozen checkpoint hashes and directly replay
24 global, 24 masked-label, 28 stimulus, 21 controlled and 12 ratio rows per model
before using them. Verification also snapshots private cache hashes.

    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_baseline_verify \
      --v2-checkpoint /path/to/v2/checkpoints/full_model.pt

    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_audit global \
      --sqlite-path /path/to/geochem.db

    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_audit quality
    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_audit geometry_extract
    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_audit geometry
    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_audit numerical
    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_checks

Reconstruct exact raw process endpoints, including carbonate mixtures whose
original z coordinates were clipped. The legacy directory here contains the
original Stage 3.6 results subdirectory:

    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_checks \
      --legacy /path/to/private/stage3_6_element2vec_space_exploration

    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_probe \
      --sqlite-path /path/to/geochem.db

    python -m stage3_6_element2vec_space_exploration.src.geo_mask15_publish

Inference defaults to MPS; pass --device cpu where needed. Model arrays, ID lists,
projection bases and per-case metrics remain under the ignored
stage3_5_element2vec_v2/outputs/value_rope_geo_mask15/audit directory.
The publisher writes only aggregates and raster figures to this independent
experiment directory. It does not generate or overwrite the editorial REPORT.md.
Use --tables-only to add supplemental tables without redrawing figures.

The principal geometry figure uses the strict admissible within-regime path
subset. raw_process_summary.csv additionally covers exact raw mixing endpoints;
closure-switch paths are a separate row, not silently combined with the
within-regime smoothness result. In stimulus_coverage.csv, a “closure crossing”
count means a case contains both statuses; contextual cases intentionally contain
multiple backgrounds, so this column is not a path-jump count for context probes.

Verification:

    python -m unittest discover -s stage3_5_element2vec_v2/tests -p 'test_*.py'
    python -m unittest discover -s stage3_6_element2vec_space_exploration/tests -p 'test_*.py'
    python scripts/verify_public_tree.py --tracked

Local verification passed 25 Stage 3.5 and 10 Stage 3.6 tests, finite trained
weights/outputs, checkpoint-coordinate identity, baseline cache subset replay,
and CPU/MPS fine-derivative checks. Full-data training and label supervision
remain limitations; source-grouped downstream probes do not undo pretraining
exposure.
