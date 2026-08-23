from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .common import resolve_path, save_json
from .experiment_context import ExperimentContext
from .registry import ExperimentRegistry, ExperimentSpec


class PCABasisStore:
    """Fit each representation once at the maximum requested rank and reuse it."""

    def __init__(
        self,
        root: str | Path,
        seed: int,
        max_components: int = 256,
        max_fit_samples: int = 12000,
        iterated_power: int = 3,
    ) -> None:
        self.root = resolve_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        self.max_components = int(max_components)
        self.max_fit_samples = int(max_fit_samples)
        self.iterated_power = int(iterated_power)
        self._cache: dict[str, dict[str, Any]] = {}
        self._manifest_rows: dict[str, dict[str, Any]] = {}
        # Reconstruct the manifest from cached bases so partial/forced reruns do
        # not silently drop bases that were not touched in the latest process.
        for path in sorted(self.root.glob("*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    basis = {key: data[key] for key in data.files}
                basis["input_dim"] = int(basis["input_dim"])
                basis["fit_samples"] = int(basis["fit_samples"])
                self._record_manifest(path.stem, basis, path)
            except (KeyError, ValueError):
                continue

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.replace("/", "_").replace(":", "_")

    def _path(self, name: str) -> Path:
        return self.root / f"{self._safe_name(name)}.npz"

    def _sample_reference(self, x: np.ndarray, name: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if len(x) <= self.max_fit_samples:
            return x
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(self.seed + offset)
        idx = np.sort(rng.choice(len(x), size=self.max_fit_samples, replace=False))
        return x[idx]

    def basis(self, name: str, reference: np.ndarray) -> dict[str, Any]:
        if name in self._cache:
            return self._cache[name]
        path = self._path(name)
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                basis = {key: data[key] for key in data.files}
            basis["input_dim"] = int(basis["input_dim"])
            basis["fit_samples"] = int(basis["fit_samples"])
            self._cache[name] = basis
            self._record_manifest(name, basis, path)
            return basis

        sampled = self._sample_reference(reference, name)
        if sampled.ndim != 2 or sampled.shape[0] < 2 or sampled.shape[1] < 1:
            raise ValueError(f"Cannot fit PCA basis {name!r} on shape {sampled.shape}")
        n_components = min(self.max_components, sampled.shape[0] - 1, sampled.shape[1])
        solver = "full" if min(sampled.shape) <= 512 or n_components >= min(sampled.shape) * 0.8 else "randomized"
        started = time.perf_counter()
        model = PCA(
            n_components=n_components,
            svd_solver=solver,
            random_state=self.seed,
            iterated_power=self.iterated_power if solver == "randomized" else "auto",
        )
        model.fit(sampled)
        elapsed = time.perf_counter() - started
        basis = {
            "mean": model.mean_.astype(np.float32),
            "components": model.components_.astype(np.float32),
            "explained_variance_ratio": model.explained_variance_ratio_.astype(np.float64),
            "singular_values": model.singular_values_.astype(np.float64),
            "input_dim": np.asarray(sampled.shape[1], dtype=np.int64),
            "fit_samples": np.asarray(sampled.shape[0], dtype=np.int64),
            "elapsed_seconds": np.asarray(elapsed, dtype=np.float64),
        }
        np.savez_compressed(path, **basis)
        basis["input_dim"] = int(sampled.shape[1])
        basis["fit_samples"] = int(sampled.shape[0])
        self._cache[name] = basis
        self._record_manifest(name, basis, path)
        self.write_manifest()
        return basis

    def _record_manifest(self, name: str, basis: dict[str, Any], path: Path) -> None:
        evr = np.asarray(basis["explained_variance_ratio"], dtype=float)
        elapsed = float(np.asarray(basis.get("elapsed_seconds", np.nan)))
        self._manifest_rows[name] = {
            "basis_name": name,
            "path": str(path),
            "input_dimension": int(basis["input_dim"]),
            "fitted_components": int(np.asarray(basis["components"]).shape[0]),
            "fit_samples": int(basis["fit_samples"]),
            "variance_2d": float(evr[:2].sum()),
            "variance_8d": float(evr[:8].sum()),
            "variance_16d": float(evr[:16].sum()),
            "variance_32d": float(evr[:32].sum()),
            "variance_64d": float(evr[:64].sum()),
            "variance_128d": float(evr[:128].sum()),
            "variance_256d": float(evr[:256].sum()),
            "fit_elapsed_seconds": elapsed,
        }

    def transform(self, name: str, reference: np.ndarray, x: np.ndarray, dimension: int) -> np.ndarray:
        basis = self.basis(name, reference)
        components = np.asarray(basis["components"])
        effective = min(int(dimension), len(components))
        centered = np.asarray(x, dtype=np.float32) - np.asarray(basis["mean"], dtype=np.float32)
        return (centered @ components[:effective].T).astype(np.float32, copy=False)

    def dimension_info(self, name: str, reference: np.ndarray, dimension: int) -> dict[str, Any]:
        basis = self.basis(name, reference)
        effective = min(int(dimension), len(basis["components"]))
        evr = np.asarray(basis["explained_variance_ratio"], dtype=float)
        return {
            "basis_name": name,
            "requested_dimension": int(dimension),
            "effective_dimension": effective,
            "retained_variance": float(evr[:effective].sum()),
        }

    def write_manifest(self) -> None:
        if self._manifest_rows:
            pd.DataFrame(self._manifest_rows.values()).sort_values("basis_name").to_csv(
                self.root / "basis_manifest.csv", index=False, encoding="utf-8-sig"
            )


class SharedInferenceCache:
    """Memoize synthetic inference so nine PCA dimensions do not rerun the model."""

    def __init__(self, base: ExperimentContext) -> None:
        self.base = base
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _key(kind: str, frame: pd.DataFrame, pure_mode: str) -> str:
        hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype=np.uint64)
        digest = hashlib.sha256(hashed.tobytes() + "|".join(frame.columns).encode("utf-8")).hexdigest()
        return f"{kind}:{pure_mode}:{digest}"

    def infer(self, kind: str, frame: pd.DataFrame, pure_mode: str) -> dict[str, np.ndarray]:
        key = self._key(kind, frame, pure_mode)
        if key not in self._cache:
            if kind == "raw":
                self._cache[key] = self.base.infer_raw_frame(frame, pure_mode=pure_mode)
            elif kind == "transformed":
                self._cache[key] = self.base.infer_transformed_frame(frame, pure_mode=pure_mode)
            else:
                raise ValueError(kind)
        return self._cache[key]


class LowRankExperimentContext:
    """Drop-in ExperimentContext view whose learned vectors live in one PCA rank."""

    def __init__(
        self,
        base: ExperimentContext,
        basis_store: PCABasisStore,
        inference_cache: SharedInferenceCache,
        dimension: int,
        config: dict[str, Any],
    ) -> None:
        self.base = base
        self.basis_store = basis_store
        self.inference_cache = inference_cache
        self.dimension = int(dimension)
        self.config = config
        self.seed = base.seed
        self.dpi = base.dpi
        self.meta = base.meta
        self.token_meta = base.token_meta
        self.vocab = base.vocab
        self.source_config = base.source_config
        self._latents: dict[str, dict[str, np.ndarray]] = {}
        self._tokens: dict[str, np.ndarray] | None = None
        self._static_trained: dict[str, np.ndarray] | None = None
        self._static_random: dict[str, np.ndarray] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def numeric_tags(self) -> list[str]:
        return self.base.numeric_tags

    @property
    def pure_tokens(self) -> list[str]:
        return self.base.pure_tokens

    def _basis_name_for_latent(self, mode: str, key: str) -> str:
        prefix = "random" if mode == "random_numeric_only" else "trained"
        return f"sample_{prefix}_{key}"

    def _latent_reference(self, mode: str, key: str) -> np.ndarray:
        ref_mode = "random_numeric_only" if mode == "random_numeric_only" else "numeric_only"
        return self.base.latents(ref_mode)[key]

    @property
    def static_trained(self) -> dict[str, np.ndarray]:
        if self._static_trained is None:
            self._static_trained = {}
            for key, values in self.base.static_trained.items():
                if values.ndim == 2 and len(values) >= 2:
                    self._static_trained[key] = self.basis_store.transform(
                        f"static_trained_{key}", values, values, self.dimension
                    )
                else:
                    self._static_trained[key] = values
        return self._static_trained

    @property
    def static_random(self) -> dict[str, np.ndarray]:
        if self._static_random is None:
            self._static_random = {}
            for key, values in self.base.static_random.items():
                if values.ndim == 2 and len(values) >= 2:
                    self._static_random[key] = self.basis_store.transform(
                        f"static_random_{key}", values, values, self.dimension
                    )
                else:
                    self._static_random[key] = values
        return self._static_random

    def latents(self, mode: str = "numeric_only") -> dict[str, np.ndarray]:
        if mode not in self._latents:
            raw = self.base.latents(mode)
            self._latents[mode] = {}
            for key, values in raw.items():
                reference = self._latent_reference(mode, key)
                self._latents[mode][key] = self.basis_store.transform(
                    self._basis_name_for_latent(mode, key), reference, values, self.dimension
                )
        return self._latents[mode]

    def token_arrays(self) -> dict[str, np.ndarray]:
        if self._tokens is None:
            raw = self.base.token_arrays()
            self._tokens = {
                key: self.basis_store.transform(f"token_trained_{key}", values, values, self.dimension)
                for key, values in raw.items()
            }
        return self._tokens

    def project_numeric_grid(self, values: np.ndarray, trained: bool = True) -> np.ndarray:
        raw = self.base.project_numeric_grid(values, trained=trained)
        if trained:
            reference = self.base.token_arrays()["projected"]
            name = "token_trained_projected"
        else:
            reference = self.base.project_numeric_grid(np.linspace(-3.0, 3.0, 25), trained=False).reshape(-1, raw.shape[-1])
            name = "token_random_projected_grid"
        flat = raw.reshape(-1, raw.shape[-1])
        transformed = self.basis_store.transform(name, reference, flat, self.dimension)
        return transformed.reshape(raw.shape[0], raw.shape[1], -1)

    def project_pure_labels(self, trained: bool = True) -> np.ndarray:
        raw = self.base.project_pure_labels(trained=trained)
        name = "pure_projected_trained" if trained else "pure_projected_random"
        return self.basis_store.transform(name, raw, raw, self.dimension)

    def project_pure_labels_for_sample_space(self, trained: bool = True) -> np.ndarray:
        raw = self.base.project_pure_labels(trained=trained)
        mode = "numeric_only" if trained else "random_numeric_only"
        reference = self._latent_reference(mode, "encoded_mean_numeric")
        name = self._basis_name_for_latent(mode, "encoded_mean_numeric")
        return self.basis_store.transform(name, reference, raw, self.dimension)

    def _transform_inference(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = {}
        for key, values in raw.items():
            reference = self._latent_reference("numeric_only", key)
            output[key] = self.basis_store.transform(
                self._basis_name_for_latent("numeric_only", key), reference, values, self.dimension
            )
        return output

    def infer_transformed_frame(self, frame: pd.DataFrame, pure_mode: str = "numeric_only") -> dict[str, np.ndarray]:
        raw = self.inference_cache.infer("transformed", frame, pure_mode)
        return self._transform_inference(raw)

    def infer_raw_frame(self, frame: pd.DataFrame, pure_mode: str = "numeric_only") -> dict[str, np.ndarray]:
        raw = self.inference_cache.infer("raw", frame, pure_mode)
        return self._transform_inference(raw)

    def basis_summary(self) -> list[dict[str, Any]]:
        rows = []
        for mode, key in [
            ("numeric_only", "projected_mean_numeric"),
            ("numeric_only", "encoded_mean_numeric"),
            ("numeric_only", "cls"),
        ]:
            reference = self._latent_reference(mode, key)
            rows.append(
                self.basis_store.dimension_info(
                    self._basis_name_for_latent(mode, key), reference, self.dimension
                )
            )
        for key, values in self.base.token_arrays().items():
            rows.append(self.basis_store.dimension_info(f"token_trained_{key}", values, self.dimension))
        return rows

    def release(self) -> None:
        self._latents.clear()
        self._tokens = None
        self._static_trained = None
        self._static_random = None
        gc.collect()


class PCADimensionRegistry(ExperimentRegistry):
    def __init__(self, results_dir: str | Path, dimension: int) -> None:
        super().__init__(results_dir)
        self.dimension = int(dimension)

    def run(
        self,
        spec: ExperimentSpec,
        function: Callable[[Path, ExperimentSpec], dict[str, Any]],
        samples: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        config = dict(spec.config)
        config.update({"pca_dimension": self.dimension, "space": f"pca_{self.dimension}d"})
        tagged = replace(spec, config=config)

        def timed(out: Path, inner_spec: ExperimentSpec) -> dict[str, Any]:
            started = time.perf_counter()
            result = function(out, inner_spec)
            metrics = dict(result.get("quantitative_metrics", {}))
            metrics.update(
                {
                    "pca_requested_dimension": self.dimension,
                    "experiment_elapsed_seconds": time.perf_counter() - started,
                }
            )
            result["quantitative_metrics"] = metrics
            result["stability"] = f"PCA dimension sweep at {self.dimension}D"
            return result

        result = super().run(tagged, timed, samples=samples)
        out = self.experiment_dir(tagged.experiment_id)
        if result.get("status") == "Failed technically":
            error_path = out / "errors.log"
            if not error_path.read_text(encoding="utf-8").strip():
                error_path.write_text(
                    "Inherited or deliberately isolated technical failure; see metrics.json and observations.md.\n",
                    encoding="utf-8",
                )
        return result


def dimension_config(base_config: dict[str, Any], output_root: str | Path, dimension: int) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    root = resolve_path(output_root)
    dim_root = root / "results" / f"{int(dimension)}d"
    config["project"]["results_dir"] = str(dim_root)
    config["project"]["reports_dir"] = str(root / "reports")
    config["project"]["audit_dir"] = str(root / "audit")
    return config
