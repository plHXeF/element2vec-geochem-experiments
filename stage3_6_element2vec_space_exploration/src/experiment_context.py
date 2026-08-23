from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage3_5_element2vec_v2.src.data.dataset import load_vocab_bundle
from stage3_5_element2vec_v2.src.training.train_v2 import choose_device
from stage3_5_element2vec_v2.src.models.element2vec_v2 import build_model
from stage3_5_element2vec_v2.src.models.token_projector import TokenProjector
from stage3_5_element2vec_v2.src.utils.io import load_config as load_source_config

from .common import ensure_directories, resolve_path, set_reproducible_seed


class ExperimentContext:
    def __init__(self, config: dict):
        self.config = config
        self.paths = ensure_directories(config)
        self.cache_dir = self.paths["cache"]
        self.source_config = load_source_config(config["source"]["model_config"])
        self.vocab = load_vocab_bundle(self.source_config)
        self.seed = int(config["sampling"]["seed"])
        self.dpi = int(config["analysis"].get("figure_dpi", 180))
        self.meta = pd.read_csv(self.cache_dir / "sample_metadata.csv", low_memory=False)
        self.token_meta = pd.read_csv(self.cache_dir / "token_reservoir_metadata.csv", low_memory=False)
        with np.load(self.cache_dir / "static_embeddings_trained.npz") as data:
            self.static_trained = {key: data[key] for key in data.files}
        with np.load(self.cache_dir / "static_embeddings_random.npz") as data:
            self.static_random = {key: data[key] for key in data.files}
        self._latents: dict[str, dict[str, np.ndarray]] = {}
        self._tokens: dict[str, np.ndarray] | None = None
        self._trained_projector: TokenProjector | None = None
        self._random_projector: TokenProjector | None = None
        self._full_model = None
        self._device: torch.device | None = None

    @property
    def numeric_tags(self) -> list[str]:
        return list(self.vocab.numeric_vocab["numeric_tags"])

    @property
    def pure_tokens(self) -> list[str]:
        return list(self.vocab.pure_vocab["pure_tokens"])

    def latents(self, mode: str = "numeric_only") -> dict[str, np.ndarray]:
        if mode not in self._latents:
            filename = {
                "numeric_only": "latents_numeric_only.npz",
                "masked_labels": "latents_masked_labels.npz",
                "visible_labels": "latents_visible_labels_control.npz",
                "random_numeric_only": "latents_random_numeric_only.npz",
            }[mode]
            with np.load(self.cache_dir / filename) as data:
                self._latents[mode] = {key: data[key] for key in data.files}
        return self._latents[mode]

    def token_arrays(self) -> dict[str, np.ndarray]:
        if self._tokens is None:
            with np.load(self.cache_dir / "token_reservoir_numeric_only.npz") as data:
                self._tokens = {key: data[key] for key in data.files}
        return self._tokens

    def raw_baseline(self) -> np.ndarray:
        z_cols = [f"z__{tag}" for tag in self.numeric_tags]
        values = self.meta[z_cols].to_numpy(dtype=np.float32)
        observed = np.isfinite(values).astype(np.float32)
        return np.concatenate([np.nan_to_num(values, nan=0.0), observed], axis=1)

    def major_clr(self) -> np.ndarray:
        cols = ["sio2", "tio2", "al2o3", "fe2o3", "feo", "mgo", "cao", "na2o", "k2o", "p2o5", "mno"]
        x = self.meta[cols].to_numpy(dtype=float)
        positive = x[x > 0]
        replacement = max(float(np.nanpercentile(positive, 0.5)) * 0.1, 1e-8)
        x = np.where(np.isfinite(x) & (x > 0), x, replacement)
        logx = np.log(x)
        return (logx - logx.mean(axis=1, keepdims=True)).astype(np.float32)

    def trained_projector(self) -> TokenProjector:
        if self._trained_projector is None:
            cfg = self.source_config["model"]["token_projector"]
            projector = TokenProjector(**cfg)
            state = torch.load(
                resolve_path(configured_stage3_output(self.config) / "checkpoints" / "token_projector.pt"),
                map_location="cpu",
                weights_only=True,
            )
            projector.load_state_dict(state)
            projector.eval()
            self._trained_projector = projector
        return self._trained_projector

    def random_projector(self) -> TokenProjector:
        if self._random_projector is None:
            set_reproducible_seed(self.seed, int(self.config["runtime"].get("num_threads", 8)))
            model = build_model(self.source_config, self.vocab)
            self._random_projector = model.token_projector.eval()
            model.token_projector = torch.nn.Identity()
            del model
            gc.collect()
        return self._random_projector

    @torch.no_grad()
    def project_numeric_grid(self, values: np.ndarray, trained: bool = True) -> np.ndarray:
        """Project a common z-value grid for all numeric tags.

        Returns ``[n_fields, n_values, d_model]``.
        """
        values = np.asarray(values, dtype=np.float32)
        embeddings = self.static_trained["numeric"] if trained else self.static_random["numeric"]
        embeddings = embeddings[: len(self.numeric_tags)]
        n_fields, n_values = len(self.numeric_tags), len(values)
        emb = np.repeat(embeddings[:, None, :], n_values, axis=1)
        val = np.broadcast_to(values[None, :, None], (n_fields, n_values, 1))
        flags = np.zeros((n_fields, n_values, 2), dtype=np.float32)
        raw = np.concatenate([emb, val, flags], axis=-1)
        projector = self.trained_projector() if trained else self.random_projector()
        return projector(torch.from_numpy(raw)).numpy().astype(np.float32)

    @torch.no_grad()
    def project_pure_labels(self, trained: bool = True) -> np.ndarray:
        embeddings = self.static_trained["pure"] if trained else self.static_random["pure"]
        zeros = np.zeros((len(embeddings), 3), dtype=np.float32)
        raw = np.concatenate([embeddings, zeros], axis=1)
        projector = self.trained_projector() if trained else self.random_projector()
        return projector(torch.from_numpy(raw)).numpy().astype(np.float32)

    def full_model(self):
        if self._full_model is None:
            device = choose_device(str(self.config["runtime"].get("device", "auto")))
            model = build_model(self.source_config, self.vocab)
            checkpoint = torch.load(resolve_path(self.config["source"]["checkpoint"]), map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            del checkpoint
            gc.collect()
            self._device = device
            self._full_model = model.to(device).eval()
        return self._full_model, self._device

    def transform_raw_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        from stage3_5_element2vec_v2.src.data.transforms import apply_value_transform

        frame = frame.copy()
        stats_by_col = {row["db_column"]: row.to_dict() for _, row in self.vocab.transform_stats.iterrows()}
        clip_z = self.source_config["data"]["value_transform"].get("clip_z")
        for item in self.source_config["data"]["numeric_fields"]:
            col = item["db_column"]
            frame[f"z__{item['name']}"] = apply_value_transform(frame[col], stats_by_col[col], clip_z=clip_z)
        return frame

    def infer_transformed_frame(self, frame: pd.DataFrame, pure_mode: str = "numeric_only") -> dict[str, np.ndarray]:
        from stage3_6_element2vec_space_exploration.src.extract_cache import _extract_mode

        model, device = self.full_model()
        if "sample_id" not in frame:
            frame = frame.copy()
            frame["sample_id"] = np.arange(-len(frame), 0)
        return _extract_mode(
            model,
            frame,
            self.source_config,
            self.vocab,
            device,
            int(self.config["sampling"]["batch_size"]),
            pure_mode,
        )

    def infer_raw_frame(self, frame: pd.DataFrame, pure_mode: str = "numeric_only") -> dict[str, np.ndarray]:
        return self.infer_transformed_frame(self.transform_raw_frame(frame), pure_mode=pure_mode)


def configured_stage3_output(config: dict) -> Path:
    return resolve_path(config["source"]["stage3_5_outputs"])
