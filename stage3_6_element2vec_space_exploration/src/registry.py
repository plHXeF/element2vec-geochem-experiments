from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .common import json_default, resolve_path, save_json


REGISTRY_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "family",
    "representation_level",
    "dataset",
    "input_fields",
    "target_geological_relation",
    "seed",
    "sample_size",
    "control_group",
    "quantitative_metrics",
    "figure_paths",
    "status",
    "conclusion_summary",
    "risks_and_confounders",
    "hypothesis",
    "metric_primary",
    "metric_value",
    "baseline_value",
    "effect_size",
    "stability",
    "main_observation",
    "possible_confounder",
    "recommended_next_step",
]


@dataclass
class ExperimentSpec:
    experiment_id: str
    experiment_name: str
    family: str
    representation_level: str
    dataset: str
    input_fields: str
    target_geological_relation: str
    seed: int
    sample_size: int
    control_group: str
    hypothesis: str
    config: dict[str, Any] = field(default_factory=dict)


class ExperimentRegistry:
    def __init__(self, results_dir: str | Path):
        self.results_dir = resolve_path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.results_dir / "experiment_registry.csv"

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.results_dir / experiment_id
        (path / "figures").mkdir(parents=True, exist_ok=True)
        return path

    def run(
        self,
        spec: ExperimentSpec,
        function: Callable[[Path, ExperimentSpec], dict[str, Any]],
        samples: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        out = self.experiment_dir(spec.experiment_id)
        save_json(asdict(spec), out / "config.json")
        if samples is None:
            pd.DataFrame(columns=["sample_id"]).to_csv(out / "samples.csv", index=False)
        else:
            samples.to_csv(out / "samples.csv", index=False, encoding="utf-8-sig")
        error_path = out / "errors.log"
        error_path.write_text("", encoding="utf-8")
        try:
            result = function(out, spec)
            result.setdefault("status", "Weak/ambiguous")
            result.setdefault("quantitative_metrics", {})
            result.setdefault("figure_paths", [])
            result.setdefault("conclusion_summary", "No conclusion supplied.")
            result.setdefault("risks_and_confounders", "None recorded.")
        except Exception as exc:  # failures are evidence and must be registered
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            result = {
                "status": "Failed technically",
                "quantitative_metrics": {},
                "figure_paths": [],
                "conclusion_summary": f"{type(exc).__name__}: {exc}",
                "risks_and_confounders": "Technical failure; inspect errors.log.",
                "metric_primary": "technical_success",
                "metric_value": 0.0,
                "baseline_value": None,
                "effect_size": None,
                "stability": "not assessed",
                "main_observation": f"Experiment failed: {type(exc).__name__}",
                "possible_confounder": "technical failure",
                "recommended_next_step": "Inspect errors.log and rerun after fixing the recorded failure.",
            }
        save_json(result.get("quantitative_metrics", {}), out / "metrics.json")
        observations = result.get("observations_md") or self._default_observations(spec, result)
        (out / "observations.md").write_text(observations, encoding="utf-8")
        row = self._registry_row(spec, result)
        self._upsert(row)
        return result

    def register_unavailable(
        self,
        spec: ExperimentSpec,
        reason: str,
        recommended_next_step: str,
    ) -> None:
        def unavailable(_out: Path, _spec: ExperimentSpec) -> dict[str, Any]:
            return {
                "status": "Negative",
                "quantitative_metrics": {"available": False},
                "figure_paths": [],
                "conclusion_summary": reason,
                "risks_and_confounders": "The requested variables or labels are absent from the trained vocabulary/data view.",
                "metric_primary": "available",
                "metric_value": 0.0,
                "baseline_value": None,
                "effect_size": None,
                "stability": "not applicable",
                "main_observation": reason,
                "possible_confounder": "coverage limitation",
                "recommended_next_step": recommended_next_step,
            }

        self.run(spec, unavailable)

    def _registry_row(self, spec: ExperimentSpec, result: dict[str, Any]) -> dict[str, Any]:
        metrics = result.get("quantitative_metrics", {})
        figures = result.get("figure_paths", [])
        return {
            "experiment_id": spec.experiment_id,
            "experiment_name": spec.experiment_name,
            "family": spec.family,
            "representation_level": spec.representation_level,
            "dataset": spec.dataset,
            "input_fields": spec.input_fields,
            "target_geological_relation": spec.target_geological_relation,
            "seed": spec.seed,
            "sample_size": spec.sample_size,
            "control_group": spec.control_group,
            "quantitative_metrics": json.dumps(metrics, ensure_ascii=False, default=json_default),
            "figure_paths": ";".join(str(p) for p in figures),
            "status": result.get("status"),
            "conclusion_summary": result.get("conclusion_summary"),
            "risks_and_confounders": result.get("risks_and_confounders"),
            "hypothesis": spec.hypothesis,
            "metric_primary": result.get("metric_primary"),
            "metric_value": result.get("metric_value"),
            "baseline_value": result.get("baseline_value"),
            "effect_size": result.get("effect_size"),
            "stability": result.get("stability", "not assessed"),
            "main_observation": result.get("main_observation", result.get("conclusion_summary")),
            "possible_confounder": result.get("possible_confounder", result.get("risks_and_confounders")),
            "recommended_next_step": result.get("recommended_next_step", "No follow-up proposed."),
        }

    def _upsert(self, row: dict[str, Any]) -> None:
        if self.registry_path.exists():
            frame = pd.read_csv(self.registry_path)
            frame = frame.loc[frame["experiment_id"] != row["experiment_id"]]
        else:
            frame = pd.DataFrame(columns=REGISTRY_COLUMNS)
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        frame = frame.reindex(columns=REGISTRY_COLUMNS).sort_values("experiment_id")
        frame.to_csv(self.registry_path, index=False, encoding="utf-8-sig")

    @staticmethod
    def _default_observations(spec: ExperimentSpec, result: dict[str, Any]) -> str:
        return f"""# {spec.experiment_id}: {spec.experiment_name}

## Hypothesis

{spec.hypothesis}

## Design

- Representation: {spec.representation_level}
- Dataset: {spec.dataset}
- Input fields: {spec.input_fields}
- Sample size: {spec.sample_size}
- Control: {spec.control_group}
- Seed: {spec.seed}

## Quantitative result

```json
{json.dumps(result.get('quantitative_metrics', {}), ensure_ascii=False, indent=2, default=json_default)}
```

## Observation

{result.get('conclusion_summary', 'No conclusion supplied.')}

## Status

{result.get('status', 'Weak/ambiguous')}

## Risks and confounders

{result.get('risks_and_confounders', 'None recorded.')}

## Recommended next step

{result.get('recommended_next_step', 'No follow-up proposed.')}
"""
