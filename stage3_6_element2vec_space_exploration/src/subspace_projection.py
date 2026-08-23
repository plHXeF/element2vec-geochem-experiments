from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    method: str
    target_dimension: int | None = None
    seed: int = 42
    drop_count: int = 0
    whiten: bool = False


class ProjectionManager:
    """Deterministic, nested PCA/RP manager with explicit rank failures.

    PCA bases are fit once at the largest estimable rank. Smaller PCA spaces
    are prefixes of that basis. No padding or duplicated dimensions is allowed.
    """

    def __init__(self, root: str | Path, seed: int = 42, dtype: str = "float32") -> None:
        self.root = Path(root).resolve()
        self.seed = int(seed)
        self.dtype = np.dtype(dtype)
        self.pca_root = self.root / "pca_models"
        self.rp_root = self.root / "random_projection_models"
        self.variance_root = self.root.parent / "explained_variance"
        for path in (self.pca_root, self.rp_root, self.variance_root):
            path.mkdir(parents=True, exist_ok=True)
        self._pca: dict[str, dict[str, Any]] = {}
        self._rp: dict[tuple[str, int, int], np.ndarray] = {}

    @staticmethod
    def _safe(name: str) -> str:
        return name.replace("/", "_").replace(":", "_").replace(" ", "_")

    @staticmethod
    def stable_offset(name: str) -> int:
        return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")

    def fit_pca(self, name: str, reference: np.ndarray, max_components: int = 64) -> dict[str, Any]:
        if name in self._pca:
            return self._pca[name]
        path = self.pca_root / f"{self._safe(name)}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                model = {key: z[key] for key in z.files}
            self._pca[name] = model
            return model
        x = np.asarray(reference, dtype=self.dtype)
        if x.ndim != 2 or min(x.shape) < 2:
            raise ValueError(f"Cannot fit {name!r} on {x.shape}")
        finite = np.all(np.isfinite(x), axis=1)
        x = x[finite]
        rank_cap = min(x.shape[0] - 1, x.shape[1], int(max_components))
        if rank_cap < 1:
            raise ValueError(f"No estimable component for {name!r}")
        solver = "full" if min(x.shape) <= 512 else "randomized"
        pca = PCA(
            n_components=rank_cap,
            svd_solver=solver,
            random_state=self.seed,
            iterated_power=4 if solver == "randomized" else "auto",
        ).fit(x)
        model = {
            "mean": pca.mean_.astype(self.dtype),
            "components": pca.components_.astype(self.dtype),
            "explained_variance": pca.explained_variance_.astype(np.float64),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float64),
            "singular_values": pca.singular_values_.astype(np.float64),
            "input_dimension": np.asarray(x.shape[1], dtype=np.int64),
            "fit_samples": np.asarray(x.shape[0], dtype=np.int64),
            "estimable_rank": np.asarray(np.linalg.matrix_rank(x - pca.mean_), dtype=np.int64),
        }
        np.savez_compressed(path, **model)
        self._pca[name] = model
        self._write_variance(name, model)
        return model

    def _write_variance(self, name: str, model: dict[str, Any]) -> None:
        evr = np.asarray(model["explained_variance_ratio"], dtype=float)
        frame = pd.DataFrame({
            "pc": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": np.cumsum(evr),
        })
        frame.to_csv(self.variance_root / f"{self._safe(name)}.csv", index=False)

    def availability(self, name: str, reference: np.ndarray, target_dimension: int) -> tuple[bool, str]:
        model = self.fit_pca(name, reference)
        native = int(np.asarray(model["input_dimension"]))
        fitted = len(np.asarray(model["components"]))
        if native < target_dimension:
            return False, "not_applicable_native_dimension"
        if fitted < target_dimension:
            return False, "not_applicable_fitting_rank"
        return True, "applicable"

    def pca_info(self, name: str, reference: np.ndarray, target_dimension: int) -> dict[str, Any]:
        model = self.fit_pca(name, reference)
        ok, reason = self.availability(name, reference, target_dimension)
        evr = np.asarray(model["explained_variance_ratio"], dtype=float)
        return {
            "representation": name,
            "native_dimension": int(np.asarray(model["input_dimension"])),
            "fit_samples": int(np.asarray(model["fit_samples"])),
            "estimable_rank": int(np.asarray(model["estimable_rank"])),
            "fitted_components": len(model["components"]),
            "target_dimension": int(target_dimension),
            "applicable": ok,
            "reason": reason,
            "explained_variance": float(evr[:target_dimension].sum()) if ok else np.nan,
        }

    def pca_transform(
        self, name: str, reference: np.ndarray, x: np.ndarray, dimension: int, whiten: bool = False
    ) -> np.ndarray:
        model = self.fit_pca(name, reference)
        ok, reason = self.availability(name, reference, dimension)
        if not ok:
            raise ValueError(f"{name} PCA{dimension}: {reason}")
        centered = np.asarray(x, dtype=self.dtype) - np.asarray(model["mean"], dtype=self.dtype)
        result = centered @ np.asarray(model["components"][:dimension], dtype=self.dtype).T
        if whiten:
            scale = np.sqrt(np.maximum(np.asarray(model["explained_variance"][:dimension]), 1e-12))
            result = result / scale
        return result.astype(self.dtype, copy=False)

    def drop_pc_transform(self, name: str, reference: np.ndarray, x: np.ndarray, count: int) -> np.ndarray:
        model = self.fit_pca(name, reference)
        if count > len(model["components"]):
            raise ValueError(f"{name}: cannot remove {count} PCs")
        centered = np.asarray(x, dtype=self.dtype) - np.asarray(model["mean"], dtype=self.dtype)
        comp = np.asarray(model["components"][:count], dtype=self.dtype)
        return (centered - (centered @ comp.T) @ comp).astype(self.dtype, copy=False)

    def random_matrix(self, name: str, input_dimension: int, target_dimension: int, seed: int) -> np.ndarray:
        key = (name, int(target_dimension), int(seed))
        if key in self._rp:
            return self._rp[key]
        if target_dimension > input_dimension:
            raise ValueError("not_applicable_native_dimension")
        path = self.rp_root / f"{self._safe(name)}__rp{target_dimension}__seed{seed:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                matrix = z["matrix"]
        else:
            rng = np.random.default_rng(self.seed + int(seed) * 1009 + self.stable_offset(name))
            gaussian = rng.normal(size=(input_dimension, target_dimension))
            matrix, _ = np.linalg.qr(gaussian, mode="reduced")
            matrix = matrix.astype(self.dtype)
            np.savez_compressed(
                path,
                matrix=matrix,
                input_dimension=np.asarray(input_dimension),
                target_dimension=np.asarray(target_dimension),
                seed=np.asarray(seed),
            )
        self._rp[key] = matrix
        return matrix

    def random_transform(
        self, name: str, reference: np.ndarray, x: np.ndarray, dimension: int, seed: int
    ) -> np.ndarray:
        reference = np.asarray(reference, dtype=self.dtype)
        mean = reference.mean(axis=0, dtype=np.float64).astype(self.dtype)
        matrix = self.random_matrix(name, reference.shape[1], dimension, seed)
        return ((np.asarray(x, dtype=self.dtype) - mean) @ matrix).astype(self.dtype, copy=False)

    def transform(
        self, name: str, reference: np.ndarray, x: np.ndarray, spec: ProjectionSpec
    ) -> np.ndarray:
        if spec.method == "FULL":
            return np.asarray(x, dtype=self.dtype)
        if spec.method == "PCA":
            return self.pca_transform(name, reference, x, int(spec.target_dimension), whiten=spec.whiten)
        if spec.method == "RP":
            return self.random_transform(name, reference, x, int(spec.target_dimension), spec.seed)
        if spec.method == "DROP_PC":
            return self.drop_pc_transform(name, reference, x, spec.drop_count)
        raise ValueError(spec.method)

    def manifest(self) -> list[dict[str, Any]]:
        rows = []
        for name, model in sorted(self._pca.items()):
            evr = np.asarray(model["explained_variance_ratio"], dtype=float)
            row = {
                "representation": name,
                "native_dimension": int(np.asarray(model["input_dimension"])),
                "fit_samples": int(np.asarray(model["fit_samples"])),
                "estimable_rank": int(np.asarray(model["estimable_rank"])),
                "fitted_components": len(model["components"]),
            }
            for dimension in (2, 3, 32, 64):
                row[f"variance_{dimension}"] = float(evr[:dimension].sum()) if len(evr) >= dimension else np.nan
            rows.append(row)
        return rows

    def write_manifest(self) -> None:
        rows = self.manifest()
        if rows:
            pd.DataFrame(rows).to_csv(self.pca_root / "projection_manifest.csv", index=False)
            (self.pca_root / "projection_manifest.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
