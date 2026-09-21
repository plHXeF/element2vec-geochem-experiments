from __future__ import annotations

import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
_workspace_sys.path.insert(0, str(_WorkspacePath(__file__).resolve().parents[2]))
from geochem_workspace import require_run_dir as _require_run_dir, guard_args as _guard_args
if __name__ == "__main__":
    _require_run_dir()

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/element2vec-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import ensure_directories, load_json, load_yaml, markdown_table, resolve_path


STATUS_ORDER = [
    "Strong positive",
    "Moderate positive",
    "Weak/ambiguous",
    "Negative",
    "Invalid due to leakage/confounding",
    "Failed technically",
]

STATUS_COLORS = {
    "Strong positive": "#15803d",
    "Moderate positive": "#65a30d",
    "Weak/ambiguous": "#ca8a04",
    "Negative": "#b91c1c",
    "Invalid due to leakage/confounding": "#7e22ce",
    "Failed technically": "#475569",
}


def _parse_metrics(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return {}


def _scalar_metrics(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, bool):
            rows.append((key, float(value)))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            rows.append((key, float(value)))
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    return rows[:10]


def _ensure_experiment_figure(row: pd.Series, results_dir: Path, dpi: int) -> str:
    exp_dir = results_dir / row["experiment_id"]
    figures = exp_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "metric_summary.png"
    metrics = _scalar_metrics(_parse_metrics(row["quantitative_metrics"]))
    fig, ax = plt.subplots(figsize=(8, max(3.4, 0.38 * max(len(metrics), 4) + 1.5)))
    color = STATUS_COLORS.get(row["status"], "#475569")
    if metrics:
        names = [name.replace("_", " ") for name, _ in metrics][::-1]
        values = [value for _, value in metrics][::-1]
        ax.barh(range(len(values)), values, color=color, alpha=0.82)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlabel("Recorded scalar value (mixed metrics; see metrics.json)")
    else:
        ax.axis("off")
        ax.text(0.5, 0.58, row["status"], ha="center", va="center", fontsize=18, color=color, weight="bold")
        ax.text(0.5, 0.35, "No scalar metric available", ha="center", va="center", fontsize=10)
    ax.set_title(f"{row['experiment_id']}\n{row['experiment_name']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path.relative_to(resolve_path(".")))


def _status_figures(registry: pd.DataFrame, reports_dir: Path, dpi: int) -> list[Path]:
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    counts = registry["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(counts.index, counts.values, color=[STATUS_COLORS[s] for s in counts.index])
    ax.set_ylabel("Experiments")
    ax.set_title("Element2Vec breadth-first experiment outcomes")
    ax.tick_params(axis="x", rotation=30)
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.15, str(value), ha="center", fontsize=9)
    fig.tight_layout()
    status_path = fig_dir / "experiment_status_counts.png"
    fig.savefig(status_path, dpi=dpi)
    plt.close(fig)

    family = pd.crosstab(registry["family"], registry["status"]).reindex(columns=STATUS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(family))
    for status in STATUS_ORDER:
        values = family[status].to_numpy()
        ax.bar(family.index, values, bottom=bottom, label=status, color=STATUS_COLORS[status])
        bottom += values
    ax.set_ylabel("Experiments")
    ax.set_title("Experiment status by family")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    family_path = fig_dir / "status_by_family.png"
    fig.savefig(family_path, dpi=dpi)
    plt.close(fig)
    return [status_path, family_path]


def _candidate_table() -> pd.DataFrame:
    rows = [
        ("Context written into individual element tokens", "C1/X1", 4, 4, 3, 4, 3, 4, "Main-result candidate after stricter raw/group controls"),
        ("Background-dependent finite-difference directions", "F2/F3", 4, 3, 2, 4, 4, 4, "Main-result candidate; needs closed perturbations"),
        ("Conditional latent increment under grouped probes", "J1/J2", 4, 3, 2, 3, 3, 4, "Main-result candidate only after fold-specific pretraining"),
        ("Permutation invariance but token-dropout fragility", "K2/K3", 3, 4, 4, 4, 5, 4, "Strong capability-boundary result"),
        ("Source/missingness structure retained by random model", "K1/X4", 5, 5, 5, 5, 5, 5, "Essential risk/control result"),
        ("Volcanic–plutonic label relation geometry", "G2/G4/X3", 4, 3, 2, 3, 4, 4, "Supplement only; cross-space confirmation nonsignificant"),
        ("Mixing/simplex path continuity", "E1/E2", 3, 4, 2, 3, 4, 3, "Supplement; continuity is partly architectural"),
        ("Normative-mineral information", "I0–I4", 4, 2, 2, 3, 2, 3, "Failure/benchmark study; strict CIPW still needed"),
        ("Static element periodic/geochemical semantics", "A1–A3", 4, 1, 3, 4, 5, 3, "Study the failure; vocabulary coverage is limiting"),
    ]
    columns = ["direction", "evidence", "geological_meaning", "strength", "robustness", "novelty", "interpretability", "publication_potential", "recommendation"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["priority_score"] = frame[["geological_meaning", "strength", "robustness", "novelty", "interpretability", "publication_potential"]].mean(axis=1)
    return frame.sort_values("priority_score", ascending=False).reset_index(drop=True)


def build(config: dict) -> None:
    paths = ensure_directories(config)
    results_dir, reports_dir = paths["results"], paths["reports"]
    registry_path = results_dir / "experiment_registry.csv"
    registry = pd.read_csv(registry_path, low_memory=False)
    dpi = int(config["analysis"].get("figure_dpi", 180))
    figure_paths = []
    for index, row in registry.iterrows():
        generated = _ensure_experiment_figure(row, results_dir, dpi)
        existing = [item for item in str(row.get("figure_paths", "")).split(";") if item and item != "nan"]
        if generated not in existing:
            existing.append(generated)
        registry.at[index, "figure_paths"] = ";".join(existing)
        for path in existing:
            figure_paths.append((row["experiment_id"], row["experiment_name"], row["status"], path))
    registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
    summary_columns = [
        "experiment_id", "family", "representation_level", "hypothesis", "dataset", "sample_size",
        "metric_primary", "metric_value", "baseline_value", "effect_size", "stability", "status",
        "main_observation", "possible_confounder", "recommended_next_step",
    ]
    summary = registry.reindex(columns=summary_columns)
    summary.to_csv(reports_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    status_figures = _status_figures(registry, reports_dir, dpi)

    counts = registry["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    status_table = counts.rename_axis("status").reset_index(name="count")
    experiment_map = registry[["experiment_id", "family", "representation_level", "status", "metric_primary", "metric_value", "main_observation"]].copy()
    experiment_map["metric_value"] = pd.to_numeric(experiment_map["metric_value"], errors="coerce").round(4)
    candidate = _candidate_table()
    candidate.to_csv(reports_dir / "candidate_directions_scored.csv", index=False, encoding="utf-8-sig")

    positives = registry.loc[registry["status"].isin(["Strong positive", "Moderate positive"])][
        ["experiment_id", "status", "metric_primary", "metric_value", "baseline_value", "main_observation", "possible_confounder"]
    ]
    negatives = registry.loc[registry["status"].isin(["Negative", "Failed technically", "Invalid due to leakage/confounding"])][
        ["experiment_id", "status", "main_observation", "possible_confounder", "recommended_next_step"]
    ]
    ambiguous_failures = registry.loc[
        registry["experiment_id"].isin(["D3_global_axes", "I1_normative_readability", "I2_normative_assemblages", "K3_order_dropout_robustness", "X2_role_raw_control", "X3_cross_space_relation_followup"])
    ][["experiment_id", "status", "main_observation", "possible_confounder", "recommended_next_step"]]
    negative_full = pd.concat([negatives, ambiguous_failures], ignore_index=True).drop_duplicates("experiment_id")

    risk_rows = [
        ("Full-data label supervision", "The checkpoint used rock_group/origin/type both as visible inputs and prediction targets. Chemistry-only inference removes direct tokens but not weight-level target exposure.", "Treat fixed-latent label metrics as decodability; use fold-specific or disjoint pretraining for generalization."),
        ("Publication and analytical-panel structure", "Source kNN purity is about 0.94–0.96 even for mask-only and random encoders on the controlled subset.", "Every neighborhood result must report source/reference and mask-only controls."),
        ("Computed rock_type", "rock_type is derived from database chemistry and is hierarchically coupled to rock_origin/group.", "Do not interpret label separability as independent geological discovery."),
        ("Missingness semantics", "Missing fields create no token, so the set itself reveals analytical coverage. Random 30% token removal reduces encoded-mean cosine to about 0.33.", "Evaluate realistic analytical panels and explicitly model censoring/detection limits."),
        ("Sparse external geology", "Tectonic setting, alteration, locality, province, suite, metamorphic grade, and most isotope targets are absent or sparse.", "Acquire explicit group metadata before process-level claims."),
        ("Incomplete element vocabulary", "Only La, Ce, Nd, Yb among REE and no Hf, Ta, Cs, Pr, Eu, Ho, Mo, W tags.", "Static periodic/REE conclusions are coverage-limited."),
        ("Synthetic process assumptions", "Mixing, melting, weathering, and alteration scans use declared broad proxies, not calibrated petrological simulators.", "Use natural suites or documented physical models for second-round tests."),
        ("CIPW approximation", "The current norm is an NNLS stoichiometric proxy and can admit non-unique/mutually incompatible phases.", "Validate a strict CIPW implementation before mineral-specific claims."),
        ("Single final checkpoint", "No epoch checkpoints exist, so representation stability across training time cannot be tested.", "Save periodic checkpoints in future runs."),
        ("Low-dimensional methods", "PCA is auxiliary; UMAP is unavailable and SpectralEmbedding crashed in native code.", "Base claims on high-dimensional metrics and isolate future graph/UMAP environments."),
    ]
    risks = pd.DataFrame(risk_rows, columns=["risk", "evidence", "required_mitigation"])
    risks.to_csv(reports_dir / "data_and_leakage_risks.csv", index=False, encoding="utf-8-sig")

    report = f"""# Element2Vec V2 representation-space breadth-first exploration

## 1. Executive summary

This audit registered **{len(registry)} experiments** across families A–K plus
targeted automatic follow-ups (X). The outcome distribution is:

{markdown_table(status_table)}

![Experiment status counts](figures/experiment_status_counts.png)

The clearest capability is not a Word2Vec-like periodic table. Static numeric
tag embeddings show essentially no trained-over-random periodic or broad
geochemical-affinity signal. Instead, the Transformer writes whole-sample
chemical context into each contextual token: at nearly fixed target-element
value, mafic/intermediate/felsic background remains highly readable. This is
partly useful context representation and partly expected because the token can
attend directly to SiO₂ and the full panel.

The largest risk is analytical-source structure. On exactly matched samples,
source-neighbor purity is about 0.94–0.96 in raw values, observation masks,
random Transformer latents, and trained latents. Training adds rock-type
structure, but it does not remove the source/reporting structure. In addition,
random 30% token removal reduces encoded-mean cosine to roughly 0.33, so the
model is permutation invariant but not robust to arbitrary panel loss.

The most defensible next directions are contextual finite-difference geometry,
publication-controlled conditional value, and realistic missing-panel/source
audits. Label analogies, ideal process paths, and normative-mineral retrieval
remain supplementary until confirmed with leakage-safe training and stronger
geological metadata.

## 2. Model and data audit

Authoritative audit artifacts:

- [`model_audit.md`](../audit/model_audit.md)
- [`data_audit.md`](../audit/data_audit.md)
- [`available_representations.json`](../audit/available_representations.json)

The analyzed layers were kept separate: 509D static numeric/pure-label
embeddings, 768D MLP-projected tokens, 768D Transformer contextual tokens,
projected mean numeric tokens, contextual mean numeric latent, and CLS latent.
Main geological checks used chemistry-only inference. Masked and visible label
modes were retained only as controls.

The shared cache contains 12,000 stratified samples, exactly 1,000 token records
per numeric field, a 3,000-sample same-seed random-model control, and explicit
sample/checkpoint hashes.

## 3. Experiment map

Every experiment—including negative, unavailable, confounded, and failed
experiments—is listed below. Each row resolves to a directory containing
`config.json`, `metrics.json`, `samples.csv`, `figures/`, `observations.md`, and
`errors.log`.

{markdown_table(experiment_map)}

## 4. Main positive observations

{markdown_table(positives)}

Important qualification: a positive status means the prespecified metric was
present in this breadth-first scan. It does not automatically mean novelty,
causality, or held-out generalization. For example, monotonic synthetic paths
are expected from a continuous network, while label organization is exposed to
full-data label supervision.

## 5. Anomalous and negative results

The complete negative/failure discussion is in
[`negative_results.md`](negative_results.md). The most consequential findings
are:

- no convincing periodic-property or Goldschmidt structure in static numeric tags;
- only four trained REE tags, making a real REE sequence untestable;
- direct sample-to-label alignment and cross-space relation preservation are weak;
- global latent axes and normative-mineral readout do not beat raw chemistry;
- strict sequential CIPW is explicitly unimplemented; I1–I4 are only an NNLS
  stoichiometric proxy and are kept separate from the failed strict calculation;
- contextual process proxies do not beat full raw chemistry under grouped controls;
- exact permutation invariance coexists with severe random token-dropout sensitivity;
- SpectralEmbedding failed with native exit 139 and was not used for a claim.

## 6. Comparison across representation levels

| Level | What it expresses best | Main limitation |
| --- | --- | --- |
| Static numeric-tag embedding | Field identity and task-specific tag parameters | No clear periodic/geochemical family gain over random |
| MLP input token | Smooth, nearly monotonic per-field value trajectories | Shared z scale is not a shared physical/log-fold scale |
| Contextual token | Whole-sample chemical background and role proxies | Also absorbs source, missingness, and label-supervised structure |
| Contextual mean sample latent | Rock-type neighborhoods, conditional label information, process-path continuity | Full-data labels and analytical source prevent generalization claims |
| CLS latent | Similar broad structure to contextual mean | CLS had no direct pooling objective and is less interpretable |
| Pure-label embedding | Some volcanic–plutonic and sequence relation geometry | Weak natural alignment to chemistry-only sample space |

## 7. Comparison with traditional representations

- Encoded mean improves rock-type kNN purity over raw/log+mask, major CLR, and
  raw PCA on the controlled subset, but this label metric is contaminated by
  label-supervised pretraining.
- Raw chemistry generally equals or beats contextual tokens for algebraic
  indicators such as Mg#, MALI, total alkali, and Sr/Y.
- Raw major oxides outperform latent for the CIPW-like normative-mineral proxy,
  which is expected because the proxy is deterministic from those oxides.
- Global latent PCA axes are less aligned with the predeclared traditional
  gradients than raw PCA axes; the latent is not simply a cleaner global
  petrological coordinate system.
- The strongest potentially new behavior is local/contextual: the same target
  value receives different token geometry depending on the rest of the sample,
  and multi-element interventions show background-dependent nonlinear residuals.

## 8. Candidate directions

Scores use 1–5 ordinal ratings and are evidence-based triage, not a paper plan.

{markdown_table(candidate)}

Three categories follow:

1. **Main-result candidates:** contextual token geometry, controlled finite
   differences, and the source/missingness capability boundary.
2. **Supplementary experiments:** mixing/simplex continuity, volcanic–plutonic
   label relations, and idealized mineral retrieval.
3. **Failures worth studying:** absent static periodic structure, weak
   sample–label alignment, raw-superior global/normative structure, and token
   dropout fragility.

## 9. Recommendations

1. Build publication/locality/suite-aware splits and retrain or pretrain outside
   each evaluation fold before any predictive/generalization claim.
2. Replace uniform token dropout with realistic analytical-panel ablations and
   add explicit censoring/detection-limit state.
3. Deepen contextual finite-difference experiments for the fields with the
   lowest cross-background direction consistency, using closed compositions.
4. Expand the numeric vocabulary to a complete REE/HFSE/LILE panel before
   revisiting Word2Vec-style element semantics.
5. Validate strict CIPW and documented melting/fractionation models only after
   the representation controls above are stable.
6. Do not force a single global linear geological analogy; use local directions,
   curves, and branches where supported.
"""
    (reports_dir / "element2vec_exploration_report.md").write_text(report, encoding="utf-8")

    promising = f"""# Promising directions

{markdown_table(candidate)}

## Evidence threshold

Directions are promising only when they have a high-dimensional quantitative
effect, an appropriate raw/random/missingness control, and an explicit path to
publication/source-held-out validation. Synthetic monotonicity alone is not
treated as novelty.
"""
    (reports_dir / "promising_directions.md").write_text(promising, encoding="utf-8")

    negative_md = f"""# Negative, anomalous, confounded, and failed results

Negative evidence is retained with the same status as positive evidence.

{markdown_table(negative_full)}

## Interpretation rules

- `Negative` means the prespecified effect was absent or required variables were unavailable.
- `Invalid due to leakage/confounding` means the measured structure cannot support the intended geological claim.
- `Failed technically` is not converted into a negative geological conclusion.
- Selected ambiguous results are included above when the baseline outperformed
  Element2Vec or a robustness metric revealed an important limitation.
"""
    (reports_dir / "negative_results.md").write_text(negative_md, encoding="utf-8")

    risks_md = f"""# Data and leakage risks

{markdown_table(risks)}

The strongest quantitative warning is the high source-neighborhood purity of
the mask-only and random-model controls. Chemistry-only inference is necessary
but not sufficient for leakage-safe evaluation because the checkpoint weights
were trained on all samples and labels.
"""
    (reports_dir / "data_and_leakage_risks.md").write_text(risks_md, encoding="utf-8")

    figure_index = pd.DataFrame(figure_paths, columns=["experiment_id", "experiment_name", "status", "figure_path"])
    figure_index.to_csv(reports_dir / "figure_index.csv", index=False, encoding="utf-8-sig")
    figure_md = f"""# Figure index

All experiments have a standardized metric/status figure. Experiments with
specialized plots list those figures as additional rows.

{markdown_table(figure_index)}

Report-level figures:

- `figures/experiment_status_counts.png`
- `figures/status_by_family.png`
"""
    (reports_dir / "figure_index.md").write_text(figure_md, encoding="utf-8")

    manifest = load_json(paths["cache"] / "cache_manifest.json")
    reproducibility = f"""# Reproducibility

## Source identity

```json
{json.dumps(manifest, ensure_ascii=False, indent=2)}
```

## Environment

- Python: `/opt/anaconda3/envs/Element2Vec/bin/python`
- Primary packages: numpy, pandas, scipy, scikit-learn, torch, matplotlib
- UMAP/Plotly/Parquet dependencies were not required.
- Main inference used `{manifest['device']}` and completed in {manifest['elapsed_seconds']:.1f} seconds.
- Random seed: {manifest['seed']}

## Commands

```bash
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.audit_project --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.extract_cache --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.run_experiments --config stage3_6_element2vec_space_exploration/configs/exploration.yaml --batch all
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.build_reports --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
/opt/anaconda3/envs/Element2Vec/bin/python -m stage3_6_element2vec_space_exploration.src.verify_outputs --config stage3_6_element2vec_space_exploration/configs/exploration.yaml
```

Each experiment records its effective configuration, samples, metrics,
observations, figures, and errors. The registry is upserted by experiment ID, so
reruns replace one experiment rather than duplicate it.
"""
    (reports_dir / "reproducibility.md").write_text(reproducibility, encoding="utf-8")
    print(f"[reports] wrote {reports_dir / 'element2vec_exploration_report.md'}")
    print(f"[reports] experiments={len(registry)} figures={len(figure_index)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage3_6_element2vec_space_exploration/configs/exploration.yaml")
    args = _guard_args(parser.parse_args())
    _guard_args(args)
    build(load_yaml(args.config))


if __name__ == "__main__":
    main()
