# Element2Vec V2 data audit

## Data source

- SQLite: `legacy_whole_rock_geochem/geochem.db`
- View: `v_high_quality_dataset`
- Samples: 270,241
- View columns: 107
- Model numeric inputs: 34

The view is built from canonicalized references and a cross-publication
deduplication decision table. It requires coordinates and a complete core major
oxide panel. This improves baseline quality but does not eliminate publication,
locality, or suite leakage.

## Model-input coverage and transforms

| token | unit | transform | coverage |
| --- | --- | --- | --- |
| SiO2 | wt_pct | robust_z | 1 |
| TiO2 | wt_pct | robust_z | 1 |
| Al2O3 | wt_pct | robust_z | 1 |
| Fe2O3 | wt_pct | robust_z | 0.672126 |
| FeO | wt_pct | robust_z | 1 |
| MgO | wt_pct | robust_z | 1 |
| CaO | wt_pct | robust_z | 1 |
| Na2O | wt_pct | robust_z | 1 |
| K2O | wt_pct | robust_z | 1 |
| P2O5 | wt_pct | robust_z | 0.952653 |
| MnO | wt_pct | robust_z | 0.940153 |
| LOI | wt_pct | robust_z | 0.418493 |
| CO2 | wt_pct | robust_z | 0.327852 |
| Rb_ppm | ppm | log10 + robust_z | 0.603143 |
| Sr_ppm | ppm | log10 + robust_z | 0.701019 |
| Ba_ppm | ppm | log10 + robust_z | 0.663101 |
| Zr_ppm | ppm | log10 + robust_z | 0.67166 |
| Nb_ppm | ppm | log10 + robust_z | 0.541746 |
| Y_ppm | ppm | log10 + robust_z | 0.639411 |
| V_ppm | ppm | log10 + robust_z | 0.526386 |
| Cr_ppm | ppm | log10 + robust_z | 0.614133 |
| Ni_ppm | ppm | log10 + robust_z | 0.606666 |
| Cu_ppm | ppm | log10 + robust_z | 0.527525 |
| Zn_ppm | ppm | log10 + robust_z | 0.52166 |
| La_ppm | ppm | log10 + robust_z | 0.499821 |
| Ce_ppm | ppm | log10 + robust_z | 0.510393 |
| Nd_ppm | ppm | log10 + robust_z | 0.383979 |
| Th_ppm | ppm | log10 + robust_z | 0.43358 |
| U_ppm | ppm | log10 + robust_z | 0.345762 |
| Pb_ppm | ppm | log10 + robust_z | 0.439833 |
| Sc_ppm | ppm | log10 + robust_z | 0.449599 |
| Co_ppm | ppm | log10 + robust_z | 0.406878 |
| Ga_ppm | ppm | log10 + robust_z | 0.377186 |
| Yb_ppm | ppm | log10 + robust_z | 0.347838 |

## Label hierarchy

| field | classes_nonmissing | missing | largest_class | largest_class_n |
| --- | --- | --- | --- | --- |
| rock_group | 3 | 4292 | igneous | 234354 |
| rock_origin | 9 | 27368 | volcanic | 153121 |
| rock_type | 81 | 17735 | subalkalic basalt | 48622 |

`rock_group`, `rock_origin`, and `rock_type` are related hierarchical labels,
not independent supervision targets. In particular, `rock_type` comes from the
database computed table and is strongly coupled to major-element chemistry.

## Available external interpretation fields

`sample_id`, `canonical_ref_id`, `author`, `data_source`, `year`, `latitude`, `longitude`, `rock_name`, `protolith`, `material`, `rock_group`, `rock_origin`, `rock_facies`, `rock_type`, `age`, `mg_number`, `asi`, `mali`, `maficity`, `cia`, `wip`, `spar`, `qtzindex`

The database also exposes isotope columns, but they were not used as V2 tokens.
They may therefore be used as external interpretation targets where coverage is
sufficient. The current view does not expose explicit detection-limit states.
Non-positive trace values are discarded before log transformation, and extreme
robust-z values are clipped to ±12.0.

## Missingness and units

- Missing numeric fields create no token, so token-set composition directly
  carries reporting and analytical-coverage information.
- Major fields are in weight percent; trace fields are in ppm.
- Major fields use direct robust-z. Trace fields use log10 then robust-z.
- There is no explicit below-detection token or unit embedding.

## Known unavailable targets

The trained vocabulary cannot test a complete REE sequence, Zr–Hf, Nb–Ta,
Rb–Cs–K, Mo–W, or Y–Ho. Tectonic setting, alteration, province, suite, and
locality are also absent as trained pure-label fields. These limitations will be
registered as negative/unavailable experiments rather than silently omitted.
