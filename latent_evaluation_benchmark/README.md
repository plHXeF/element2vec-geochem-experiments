# Latent Evaluation Benchmark v0.1

Independent, read-only checkpoint evaluation of V2, RoPE-z and RoPE_geo+15%mask against ZVector, GeoNorm and PCA-Geo. No encoder is trained or altered.

## Reproduce

From this directory, using Python 3.11 with the dependencies below:

```bash
python run.py --run-dir /path/to/registered/run all
```

Use the Python environment recorded in your local workspace settings. Data and checkpoints are logical asset IDs in `configs/benchmark.json`; their machine locations live in `workspace.local.json`. The managed entry point supplies `--run-dir`; see [workspace workflow](../docs/WORKSPACE_MANAGEMENT.md).

Dependencies: numpy, pandas, scipy, scikit-learn, torch, matplotlib, pyarrow; ffmpeg is preferred for MP4. The existing environment is reused; a local optional dependency directory can be registered in workspace settings. `outputs/run_manifest.json` records exact installed versions.

Stages may be resumed individually:

```bash
python run.py --run-dir /path/to/registered/run prepare
python run.py --run-dir /path/to/registered/run extract-real
python run.py --run-dir /path/to/registered/run bank
python run.py --run-dir /path/to/registered/run test
python run.py --run-dir /path/to/registered/run extract-bank
python run.py --run-dir /path/to/registered/run evaluate
python run.py --run-dir /path/to/registered/run render
python run.py --run-dir /path/to/registered/run verify
```

`evaluate` runs the paired 2,000-resample bootstrap and, if needed, grows only families failing precision from 2,000 to 4,000 to 8,000 episodes (single processes and composites are checked independently). Original checkpoint hashes and existing episode seeds are unchanged. All original model parameters remain frozen. Early interrupted extraction resumes from completed per-episode caches. Change scientific configuration only in a new run directory; changed configuration is rejected by the data manifest.

## Regenerate figures, tables and animations without the models

```bash
python run.py --run-dir /path/to/registered/run render
```

The rendering path uses saved metric tables, physical episodes, latent arrays and fixed visualization PCA bases; it does not construct an encoder or access the original dataset. Remove a generated video to regenerate that video; fixed automatic episode selection is in `outputs/animations/selection.json`. CSV, Parquet and JSON tables are emitted together by every metric stage. Cached latent metrics can be regenerated with `evaluate` when the precision requirement is already satisfied; that path uses no encoder unless it must append previously unevaluated Monte Carlo episodes.

## Definitions and interpretation

- Chemistry-only final-layer numeric-token mean for all three models, 768D, no label tokens.
- Same 12,000 raw records, common 34-field inputs and fixed sample-ID 70/15/15 probe split. All records were seen during original pretraining; this benchmark does **not** claim out-of-source generalization.
- Raw z uses train mean/std, missing standardized zero plus mask. GeoNorm uses observed-subcomposition closure/CLR, trace log robust scaling, explicit chondrite-normalized REE logs and missing mask. Every baseline fit and PCA fit uses train only.
- All target constituent fields are removed **before** any representation preprocessing. SigmaREE10 is the sum of the database's 10 available REE fields, not the full 14 REE total. Mg# uses molar MgO and FeO only. Eu anomaly is an external Sm/Eu/Gd target absent from all six encoder input schemas.
- Isotope/age recovery is explicitly external prediction: those fields were never in the encoders. Field support and excluded targets are recorded, not silently imputed.
- Geometry uses native Euclidean distances and one global train RMS scale only. PCA-Geo is an explicit baseline; no extra PCA/UMAP is applied before model metrics. The prescribed seven-state local rank metric uses eigenvalues locally, not a projected metric space.
- Shared synthetic/semi-synthetic EpisodeBank; synthetic modal fractions are mass fractions. Zero modal centers stay exactly zero (Dirichlet is applied to the positive support). Oxides are computed by chemical stoichiometry. No per-model clipping is allowed.
- Operators use split by initial-state ID, never by trajectory row. Whole trajectories stay together. Repeated semi-synthetic initial IDs share a split. Initial operator fits are frozen while Monte Carlo evaluation grows.
- Fractionation major removal is a constrained mineral mass-balance toy, with independently sampled synthetic trace D. Alteration is a sparse synthetic log-concentration law, not a natural named hydrothermal model. Heterogeneous composites exclude batch melting because its first melt is not an identity continuation of the source.
- Operator theta contains process parameters and the current process coordinate. Mixing additionally conditions on its initial latent, which is the previous predicted endpoint in composite rollout; the true intermediate raw-chemistry channel is zeroed. This avoids teacher-forcing the stage input; its training-only scaling is separate from global latent scaling. There is no latent whitening.
- Missing or degenerate metrics are NA, never rewarded as perfect. Illegal reverse-order episodes are excluded with coverage counts. A diverged learned rollout is reported explicitly.
- Paired CIs are clustered on natural initial-state IDs. Individual exploratory CIs are not familywise multiple-comparison discoveries. No weighted total score.

See [the frozen aggregate report](../reports/E009/REPORT.md), `outputs/audit/`, `outputs/run_manifest.json`, `outputs/verification.json` and the original request preserved in `USER_REQUEST.md`.

The complete frozen interpretation protocol (target formulas, support thresholds, physical priors, operator controls and animation rules) is in `configs/evaluation_protocol.json`. REE references use [McDonough & Sun (1995), GERM compilation](https://www.earthref.org/GERMRD/c:3/).

`render` is cache-only and does not rehash or reload original weights. `verify` separately checks the original checkpoint files are unchanged; `all` performs this final verification automatically.
