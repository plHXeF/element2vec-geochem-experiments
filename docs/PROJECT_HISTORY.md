# Project history

## Stage 1: encoding baselines

The project began as a pyrite trace-element representation benchmark. It established parsing, missingness masks, value transforms, PCA/UMAP diagnostics, clustering, neighborhood metrics, and dimension sweeps. This stage made the central confounding issue visible: analytical/reporting patterns can be easier to decode than geology.

## Stage 2: geology-informed features

Ratio, balance, and information-theoretic features were added and compared with value-only baselines. The work improved interpretability but also showed that high random-fold classification performance is not evidence of transportable geological information.

## Stage 3: token prototype

A small element-token Transformer was built for the pyrite panel. It demonstrated the distinction between static element identity, projected numeric tokens, contextual tokens, and pooled sample latents. Label masking became a mandatory evaluation constraint.

## Stage 3.5: Element2Vec V2

V2 moved to variable-length whole-rock samples. Observed numeric fields and available geological labels became unordered tokens. A shared MLP projected 512D inputs to a 768D, 12-layer Transformer. A 50-epoch run produced the released step-13,200 checkpoint.

## Stage 3.6: semantic-space audit

A breadth-first registry tested static element relations, numeric trajectories, contextual roles, sample neighborhoods, idealized geological processes, perturbation operators, label relation algebra, sample-label alignment, normative proxies, conditional probes, and robustness controls. Positive, negative, confounded, unavailable, and failed experiments were all retained.

## Multiscale subspace replay

The registry was replayed in FULL, PCA64, PCA32, PCA3, and PCA2 spaces with matched random projections, whitening, component removal, bootstrap stability, direction-energy analysis, and PC attribution. The result narrowed the claim: stable low-rank recovery is local and clearest for single-element finite-difference directions, while a universal low-dimensional geological semantic space is not supported.
