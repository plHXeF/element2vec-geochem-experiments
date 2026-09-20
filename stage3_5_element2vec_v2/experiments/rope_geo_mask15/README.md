# RoPE_geo + 15% numeric mask

Independent experiment on experiment/value-rope-z, based on commit 53bb70f.
Implementation and smoke verification: 2026-09-12. User-launched training completed
13,200 steps; paired analysis completed 2026-09-13. Local geometry improved on several
small-step tests, while chemistry-only global rock neighborhoods worsened and source
clustering increased. See REPORT.md. Previous V2 and RoPE-z outputs/reports remain frozen.

## Fixed conditions

Config: configs/v2_value_rope_geo_mask15.yaml (relative to Stage 3.5).
Same whole-rock view, frozen 34 numeric tags and 96 rock-label embedding entries,
12-layer / hidden 768 / 12-head / head dimension 64 Transformer, 3072 FFN,
85,255,040 parameters, Q/K-only full-head continuous RoPE, base 10000, scale 1.
No input MLP, absolute-value channel, new decoder, or attention change.

Seed 42, full-data training on 270,241 rows, batch 512, accumulation 2,
50 nominal epochs / 13,200 optimizer updates, AdamW LR 2.5e-4 to 3e-6,
warmup 500, weight decay .01, clipping 1, no mixed precision.
The fixed 8192-sample probe remains a training diagnostic, not held-out validation.
The 8000/4000 analysis split is for projection fitting/evaluation; both pools
were exposed to pretraining.

## Coordinates and provenance

All reference values, rules, epsilon values and family scales are stored in the
versioned YAML under data.value_transform. No statistics are fitted for geo v1.
The original V2 statistics are retained as frozen vocabulary/checkpoint provenance;
the geo forward preprocessing does not use their medians or IQRs.

Major oxide panel: SiO2, TiO2, Al2O3, Fe2O3, FeO, MgO, CaO, Na2O, K2O, P2O5, MnO.
All 11 must be present and nonnegative, with total T in [90,105] wt%.
For eligible records, C_i* = 100 C_i / T (anhydrous normalization), p_i = C_i*/100.
For other records, p_i = C_i/100 for each valid observed oxide. Missing/negative
oxides stay absent, never zero-filled. FeO and Fe2O3 remain the database's separate
fields; outside-vocabulary total-iron fields do not enter the denominator.
LOI and CO2 are excluded from the closure panel.

    major coordinate = s_major * (sqrt(p_i + epsilon_fraction) - sqrt(epsilon_fraction))
    s_major = 1; epsilon_fraction = 1e-12

This square-root mass-fraction coordinate and the complete-panel/90–105 rule are
experimental design choices. They are not attributed as a published normalization
standard. The normalization formula is consistent with
[USGS igneous indices documentation](https://pubs.usgs.gov/of/2016/1191/ofr20161191_meta/IgneousIndices/igneous_rock_geochem.faq.html).
We do not adopt that source's zero-filling of missing oxides.
The rule conservatively covers 161,270 / 270,241 records (59.6764%).
Completeness/total thresholds create boundaries; continuity tests must separate
within-regime perturbations from crossings and report closure coverage by source.
Closure status is diagnostic metadata, never a model input.

For traces (input concentrations in ppm):

    trace coordinate = s_trace * log10(max(C_i, epsilon_ppm) / PM_i)
    s_trace = 1; epsilon_ppm = 1e-9; C_i <= 0 is absent

PM references use one coherent set: McDonough & Sun (1995), The composition of
the Earth, Chemical Geology 120, 223–253, Table 5, DOI
[10.1016/0009-2541(94)00140-4](https://doi.org/10.1016/0009-2541(94)00140-4).
Values are transcribed from the Primitive Mantle entries in
[EarthRef GERM](https://www.earthref.org/GERMRD/c:3/), converting ppb to ppm.
Do not mix this set with Sun & McDonough (1989).

| Element | PM ppm | Element | PM ppm | Element | PM ppm |
|---|---:|---|---:|---|---:|
| Rb | 0.6 | Sr | 19.9 | Ba | 6.6 |
| Zr | 10.5 | Nb | 0.658 | Y | 4.3 |
| V | 82 | Cr | 2625 | Ni | 1960 |
| Cu | 30 | Zn | 55 | La | 0.648 |
| Ce | 1.675 | Nd | 1.25 | Th | 0.0795 |
| U | 0.0203 | Pb | 0.150 | Sc | 16.2 |
| Co | 105 | Ga | 4 | Yb | 0.441 |

LOI/CO2 remain observed numeric tokens to preserve the vocabulary. Their auxiliary
coordinate is s_aux * sign(C) * (sqrt(abs(C)/100 + eps) - sqrt(eps)), with scale 1
and eps 1e-12. Signed LOI is allowed; negative CO2 is absent. This auxiliary
coordinate is not an anhydrous composition. There is no per-element z/IQR scale
or coordinate clipping in any geo family.

## Multi-target masking and data flow

User-confirmed scope: only numeric-value-task samples use 15% masking.
Task proportions remain numeric value .45 / numeric tag .15 / rock label .40;
rock type/origin/group proportions remain .60/.25/.15.
Label tasks use exactly the old single-label logic, without additional numeric masks.

Given n observed numeric tokens, select k = floor(.15*n) plus a Bernoulli draw for
the fractional remainder, clamp to [1,n], then sample k distinct positions uniformly.
Small panels can exceed 15% because of the minimum one mask. No missing field is
invented. The new config requires scope=numeric_value_task and stochastic rounding.

    raw record -> sample-local coordinates -> observed numeric tokens -> task draw
    numeric-value task -> k masks -> zero their visible values + shared mask state
    shuffled labels [B,L] -> embedding [B,L,768] -> prepend CLS [B,L+1,768]
    visible coordinates [B,L] -> RoPE positions [B,L+1] -> existing Q/K attention
    encoded numeric positions [B,L,768] -> shared existing 768->1 decoder [B,L]
    numeric_prediction_mask [B,L] selects all k targets -> mean Huber loss

The decoder targets are the same scaled coordinates used by RoPE. Other task
heads/weights are unchanged. Each micro-batch averages over its masked numeric
tokens; gradient accumulation retains the baseline's equal micro-batch weighting.
Probe numeric Huber is aggregated by total masked-token count across probe batches.
Label probe aggregation remains the baseline behavior. Console and CSV logs show
actual numeric masked/eligible counts; the CSV also records their ratio.

All masked positions are zeroed again inside the model's position constructor;
targets never enter attention. A real zero coordinate and a masked zero remain
distinguishable through the shared mask embedding. Closure is performed before
masking: compositional constraints can make a single missing major coordinate
reconstructible from the others. Thus lower numeric loss is not independent evidence
of improved geology. Coordinate units also differ from V2/z, so absolute Huber values
across versions are not directly comparable.

## Run from the repository root

Use the existing Element2Vec environment (PyTorch required).

    python -m unittest discover -s stage3_5_element2vec_v2/tests -p 'test_*.py'
    python -m unittest discover -s stage3_6_element2vec_space_exploration/tests -p 'test_*.py'

Optional full-model two-step smoke (uses a fresh, ignored subdirectory):

    python -u -m stage3_5_element2vec_v2.src.training.smoke_geo_mask15 \
      --sqlite-path /path/to/geochem.db --device mps \
      --outputs-dir stage3_5_element2vec_v2/outputs/value_rope_geo_mask15/smoke_new

Formal training:

    python -u -m stage3_5_element2vec_v2.src.training.train_v2 \
      --config stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml \
      --sqlite-path /path/to/geochem.db --device mps

Use cuda or cpu if appropriate. Default device auto retains the original preference.
No resume behavior is added: an existing training log/full checkpoint in the chosen
geo output directory is rejected. Choose a fresh run subdirectory for a new run.
Latest full model/optimizer/scheduler/config/vocabulary/coordinate metadata is at
outputs/value_rope_geo_mask15/checkpoints/full_model.pt (relative to Stage 3.5).
Checkpoint exports overwrite only the current run's latest state, as in the baseline.
The entire new output tree and all weight extensions are Git-ignored.

After training, default extraction keeps all rock-label tokens masked:

    python -m stage3_5_element2vec_v2.src.evaluation.extract_embeddings \
      --config stage3_5_element2vec_v2/configs/v2_value_rope_geo_mask15.yaml \
      --sqlite-path /path/to/geochem.db \
      --checkpoint stage3_5_element2vec_v2/outputs/value_rope_geo_mask15/checkpoints/full_model.pt

For the main chemistry-only paired global analysis, append:

    --label-mode chemistry_only \
    --sample-ids stage3_5_element2vec_v2/outputs/value_rope_geo_mask15/audit/paired_sample_ids.csv

Chemistry-only removes pure-label tokens completely; labels remain metadata only.
CLS and mean-numeric latents both have dimension 768; mean pooling excludes
CLS, labels and padding. Exact-ID extraction checks missing/duplicate IDs and preserves
the supplied order. Masked and chemistry-only outputs have different filenames.
Checkpoint loading rejects geo/z mismatch and changes to reference values/scales/rules.

Before paired extraction, prepare the frozen analysis inputs once:

    python -m stage3_5_element2vec_v2.src.evaluation.prepare_geo_comparison \
      --sqlite-path /path/to/geochem.db

This already passed locally. On another machine the original private RoPE-z audit
metadata/protocol/global_inputs are required. The command refuses to replace its
frozen audit directory; it validates the original metadata hash and current database
coordinates, copies exact IDs and disjoint pools, and saves aggregate QC locally.

See [independent report](REPORT.md) and [paired evaluation protocol](PAIRED_PROTOCOL.md).
