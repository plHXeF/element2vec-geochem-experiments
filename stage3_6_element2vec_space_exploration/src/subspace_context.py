from __future__ import annotations

import copy
import gc
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .experiment_context import ExperimentContext
from .pca_sweep_context import SharedInferenceCache
from .registry import ExperimentRegistry, ExperimentSpec
from .subspace_projection import ProjectionManager, ProjectionSpec


CROSS_SPACE_IDS = {
    "H1_sample_label_retrieval", "H2_centroid_label_alignment", "H3_cross_space_relations",
    "I4_label_residual_norms", "X3_cross_space_relation_followup",
}


class SubspaceExperimentContext:
    """ExperimentContext-compatible view using one declared projection protocol."""

    def __init__(
        self,
        base: ExperimentContext,
        manager: ProjectionManager,
        inference_cache: SharedInferenceCache,
        spec: ProjectionSpec,
        fit_idx: np.ndarray,
        token_fit_idx: np.ndarray,
        cross_space: bool = False,
    ) -> None:
        self.base = base
        self.manager = manager
        self.inference_cache = inference_cache
        self.projection_spec = spec
        self.fit_idx = fit_idx
        self.token_fit_idx = token_fit_idx
        self.cross_space = cross_space
        self.config = copy.deepcopy(base.config)
        self.seed = base.seed
        self.dpi = base.dpi if spec.method != "RP" else 60
        self.meta = base.meta
        self.token_meta = base.token_meta
        self.vocab = base.vocab
        self.source_config = base.source_config
        self._latents: dict[str, dict[str, np.ndarray]] = {}
        self._tokens: dict[str, np.ndarray] | None = None
        self._static_trained: dict[str, np.ndarray] | None = None
        self._static_random: dict[str, np.ndarray] | None = None
        self._pure_trained_raw: np.ndarray | None = None
        self._pure_random_raw: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def numeric_tags(self) -> list[str]:
        return self.base.numeric_tags

    @property
    def pure_tokens(self) -> list[str]:
        return self.base.pure_tokens

    def _apply(self, name: str, reference: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.manager.transform(name, reference, x, self.projection_spec)

    @property
    def static_trained(self) -> dict[str, np.ndarray]:
        if self._static_trained is None:
            self._static_trained = {}
            for key, values in self.base.static_trained.items():
                if values.ndim == 2 and len(values) > 1:
                    try:
                        self._static_trained[key] = self._apply(f"static_trained_{key}", values, values)
                    except ValueError as exc:
                        if "not_applicable" not in str(exc):
                            raise
                        self._static_trained[key] = values
                else:
                    self._static_trained[key] = values
        return self._static_trained

    @property
    def static_random(self) -> dict[str, np.ndarray]:
        if self._static_random is None:
            self._static_random = {}
            for key, values in self.base.static_random.items():
                if values.ndim == 2 and len(values) > 1:
                    try:
                        self._static_random[key] = self._apply(f"static_random_{key}", values, values)
                    except ValueError as exc:
                        if "not_applicable" not in str(exc):
                            raise
                        self._static_random[key] = values
                else:
                    self._static_random[key] = values
        return self._static_random

    def _raw_pure(self, trained: bool) -> np.ndarray:
        if trained:
            if self._pure_trained_raw is None:
                self._pure_trained_raw = self.base.project_pure_labels(True)
            return self._pure_trained_raw
        if self._pure_random_raw is None:
            self._pure_random_raw = self.base.project_pure_labels(False)
        return self._pure_random_raw

    def _sample_reference(self, mode: str, key: str) -> tuple[str, np.ndarray]:
        random = mode == "random_numeric_only"
        base_mode = "random_numeric_only" if random else "numeric_only"
        values = self.base.latents(base_mode)[key]
        reference = values if random else values[self.fit_idx]
        key_name = {"encoded_mean_numeric": "contextual_mean_numeric"}.get(key, key)
        name = f"{'random_' if random else ''}sample_{key_name}"
        if self.cross_space and key == "encoded_mean_numeric":
            labels = self._raw_pure(not random)
            reference = np.concatenate([reference, labels], axis=0)
            name = f"{'random_' if random else ''}cross_joint_contextual_label"
        return name, reference

    def latents(self, mode: str = "numeric_only") -> dict[str, np.ndarray]:
        if mode not in self._latents:
            raw = self.base.latents(mode)
            output = {}
            for key, values in raw.items():
                name, reference = self._sample_reference(mode, key)
                output[key] = self._apply(name, reference, values)
            self._latents[mode] = output
        return self._latents[mode]

    def token_arrays(self) -> dict[str, np.ndarray]:
        if self._tokens is None:
            raw = self.base.token_arrays()
            names = {"projected": "mlp_projected_numeric_token", "contextual": "contextual_numeric_token"}
            self._tokens = {
                key: self._apply(names[key], values[self.token_fit_idx], values)
                for key, values in raw.items()
            }
        return self._tokens

    def project_numeric_grid(self, values: np.ndarray, trained: bool = True) -> np.ndarray:
        raw = self.base.project_numeric_grid(values, trained=trained)
        flat = raw.reshape(-1, raw.shape[-1])
        if trained:
            reference = self.base.token_arrays()["projected"][self.token_fit_idx]
            name = "mlp_projected_numeric_token"
        else:
            reference = self.base.project_numeric_grid(np.linspace(-3, 3, 25), trained=False).reshape(-1, raw.shape[-1])
            name = "random_mlp_projected_numeric_token"
        transformed = self._apply(name, reference, flat)
        return transformed.reshape(raw.shape[0], raw.shape[1], -1)

    def project_pure_labels(self, trained: bool = True) -> np.ndarray:
        raw = self._raw_pure(trained)
        name = "pure_label_projected" if trained else "random_pure_label_projected"
        return self._apply(name, raw, raw)

    def project_pure_labels_for_sample_space(self, trained: bool = True) -> np.ndarray:
        raw = self._raw_pure(trained)
        mode = "numeric_only" if trained else "random_numeric_only"
        name, reference = self._sample_reference(mode, "encoded_mean_numeric")
        return self._apply(name, reference, raw)

    def _transform_inference(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = {}
        for key, values in raw.items():
            name, reference = self._sample_reference("numeric_only", key)
            output[key] = self._apply(name, reference, values)
        return output

    def infer_transformed_frame(self, frame: pd.DataFrame, pure_mode: str = "numeric_only") -> dict[str, np.ndarray]:
        return self._transform_inference(self.inference_cache.infer("transformed", frame, pure_mode))

    def infer_raw_frame(self, frame: pd.DataFrame, pure_mode: str = "numeric_only") -> dict[str, np.ndarray]:
        return self._transform_inference(self.inference_cache.infer("raw", frame, pure_mode))

    def release(self) -> None:
        self._latents.clear()
        self._tokens = None
        self._static_trained = None
        self._static_random = None
        gc.collect()


class SubspaceRegistry(ExperimentRegistry):
    def __init__(self, results_dir: str | Path, projection: ProjectionSpec) -> None:
        super().__init__(results_dir)
        self.projection = projection

    def experiment_dir(self, experiment_id: str) -> Path:
        base_id = experiment_id.split("__", 1)[0]
        family = base_id[0]
        path = self.results_dir / family / experiment_id
        (path / "figures").mkdir(parents=True, exist_ok=True)
        return path

    def run(
        self,
        spec: ExperimentSpec,
        function: Callable[[Path, ExperimentSpec], dict[str, Any]],
        samples: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        tag = self.projection.name
        config = dict(spec.config)
        config["subspace_projection"] = {
            "name": tag,
            "method": self.projection.method,
            "target_dimension": self.projection.target_dimension,
            "seed": self.projection.seed,
            "whiten": self.projection.whiten,
            "drop_count": self.projection.drop_count,
        }
        tagged = replace(spec, experiment_id=f"{spec.experiment_id}__{tag}", config=config)

        def timed(out: Path, inner: ExperimentSpec) -> dict[str, Any]:
            started = time.perf_counter()
            result = function(out, inner)
            metrics = dict(result.get("quantitative_metrics", {}))
            metrics["projection_name"] = tag
            metrics["projection_method"] = self.projection.method
            metrics["target_dimension"] = self.projection.target_dimension
            metrics["projection_seed"] = self.projection.seed
            metrics["experiment_elapsed_seconds"] = time.perf_counter() - started
            result["quantitative_metrics"] = metrics
            return result

        return super().run(tagged, timed, samples=samples)
