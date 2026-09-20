# Preregistered paired evaluation — awaiting full training

Compare V2 (MLP), trained RoPE-z, trained RoPE_geo+15%mask, random V2, random
RoPE-z and random RoPE-geo. Random models use the original seed 42; random geo
uses the same initial RoPE weights as random z, changing the coordinate transform.
Report trained-minus-random changes alongside architecture differences.
This combined experiment changes coordinates and numeric mask density together;
it cannot isolate their individual causal contributions without later ablations.

## Pairing and spaces

Freeze the original 12,000 sample IDs/order and the disjoint 8,000 fitting /
4,000 evaluation pool. Keep seeds, background identities (mafic/intermediate/felsic),
missing panels, labels, endpoints, process parameters, perturbation draws, k values
and bootstrap IDs paired. Do not replace unavailable cases with new samples.
The preparation manifest records old checkpoint hashes, coordinate spec, sample
hashes and pool hashes only locally. Missing files or mismatched data must fail.

Main spaces: FULL, PCA64, PCA32; retain PCA3/PCA2 visual controls and ten fixed
random orthogonal projections per tested dimension. Fit each model's PCA on its
same 8,000 fitting samples only. Compute geometric metrics within each space.
Never directly take a cosine between unrelated model vectors or independently
fitted PCA axes. Paired comparisons concern scalar metrics or within-model
direction relationships, with paired bootstrap intervals over original samples.
Rock/source metrics use chemistry-only mean-numeric latents primarily, CLS and
all-label-masked modes secondarily. Visible-label mode is not a main geology result.

## Physical inputs versus coordinate-native probes

The old z stimulus cache is not a geo input cache. Generate one physical
concentration stimulus and apply each model's own preprocessing. Preserve old
raw process endpoints/parameters; when old stimuli exist only as z, inverse-transform
unclipped supported values and mark clipped/noninvertible cases separately.
Also show native-coordinate perturbations as an explicitly separate diagnostic.
Equivalent numeric h values in z and sqrt-fraction/PM coordinates are not
equivalent physical perturbations.

Closure-eligible and fallback samples require separate summaries. Keep the closure
regime fixed for local derivative tests; test [90,105] threshold crossings and
panel-completeness changes separately. A discontinuity from a preprocessing switch
must not be attributed to RoPE. For masked-value diagnostics, retain the closure
constraint as an explicit possible reconstruction shortcut.

| Experiment | Required metrics / paired controls |
|---|---|
| Local single-element sweep | Speed CV, adjacent tangent cosine, turning/curvature, local linear prediction error, total displacement; scale-aware and relative metrics; whole-path straightness secondary |
| Multi-h finite difference/Jacobian | Original h grid .025/.05/.1/.25/.5/1 in declared units plus original fine-step probes; central-difference tangent agreement across h, numerical-floor exclusions, local Jacobian spectrum |
| Background dependence | Same element and raw value across original natural/synthetic mafic/intermediate/felsic backgrounds; within-background stability versus across-background direction cosine |
| Multi-element perturbation | Joint displacement versus sum of singles, normalized additivity error, direction agreement, original local-linearity thresholds and ranges |
| Process paths | Original mixing, differentiation, partial melting, weathering endpoints and parameter grids; uniform-parameter speed CV, tangent/curvature, displacement versus process parameter; note simplified process assumptions |
| Common/single/contrast shifts | Physically defined common multiplication, single-element and ratio/contrast changes; closed-major common-scale invariance separated from trace PM offsets and anchor effects |
| Contextual fixed-value probe | Fixed raw element concentration and fixed model-coordinate variants both reported; closure can alter the coordinate even at fixed raw concentration, so these are different controls |
| Global rock neighborhood | Original k=15/30 purity, accuracy, balanced accuracy; paired per-sample differences and per-class recall; FULL/PCA64/PCA32 gates |
| Source/reference neighborhood | Both source and reference purity beside geology metrics; condition on rock class, token panel and closure regime; missingness/token-count baselines |
| Token-drop robustness | Same sampled drops and seeds; latent displacement, kNN purity/accuracy changes; delete tokens after computing coordinates for baseline comparability; raw-panel-drop/recompute-closure is a separate diagnostic |
| Phase wrapping/distant collision | Per-family support ranges, full multiband rotary phase distance, nearest distant pairs in coordinate and physical units, trained latent distance relative to matched random controls; no inference from first-frequency period alone |
| Layer progression | Input labels, rotary Q/K, layers 1/3/6/9/12 and pooled latent; no value geometry expected in unrotated direct label embedding |

Report finite-difference failure/exclusion counts, tiny displacement denominators,
support violations, closure-regime crossings and unmatched cases. Do not turn
numerical noise or a near-constant representation into “stable tangents.”

## Decision rules and artifact policy

Primary questions: does geo retain RoPE-z's small-step local benefits; does
chemistry-only rock neighborhood recover toward V2; and does source/reference
clustering decrease without increased panel/closure shortcuts?
Present all three jointly. Smoothness alone does not establish geological meaning.
Training losses, full-data label probes and the analysis split do not establish
generalization to unseen regions/sources. Preserve unfavorable results.

After the user finishes training, validate the final 13,200-step state and logs,
then run the paired analyses. Write private inputs/caches/latents to the new ignored
output tree, aggregate tables/figures and conclusions to this independent experiment.
Update experiment_registry.csv and REPORT.md here, never the prior RoPE-z report.
No new-model metrics or plots are currently claimed.
