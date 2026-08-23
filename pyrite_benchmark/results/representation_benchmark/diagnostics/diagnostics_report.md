# Representation Diagnostics

## Observed Mask

- Selected elements: As, Ni, Cu, Pb, Co, Au, Ag, Bi, Sb, Zn, Se, Te, Tl, Mn
- Unique observed/missing patterns: `34` across `7778` samples.
- Top pattern count: `3706`.
- `observed_mask_z` contains no concentration values; it only encodes reported/censored vs absent cells.

## Raw Robust PCA

- Raw robust PCA explained variance ratio PC1-PC4: 0.566, 0.191, 0.140, 0.049.
- This representation is robust-centered but not tail-clipped; very large concentrations can still dominate PCA.

## Diagnostic Files

- `observed_mask_patterns.csv`
- `observed_mask_top_patterns_by_label.csv`
- `observed_mask_pca_loadings.csv`
- `raw_robust_pca_loadings.csv`
- `raw_robust_extreme_pc1_samples.csv`
- `raw_robust_extreme_pc2_samples.csv`
- `raw_element_tail_diagnostics.csv`
