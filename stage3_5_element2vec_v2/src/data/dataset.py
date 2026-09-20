from __future__ import annotations

import random
import math
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..utils.io import load_json, resolve_path, vocab_dir
from .geochemical_coordinates import transform_numeric_frame


TASK_TO_ID = {"mask_numeric_value": 0, "mask_numeric_tag": 1, "mask_pure_label": 2}
ID_TO_TASK = {v: k for k, v in TASK_TO_ID.items()}
TASK_ALIASES = {
    "numeric_value": "mask_numeric_value",
    "numeric_tag": "mask_numeric_tag",
    "pure_label": "mask_pure_label",
    "mask_numeric_value": "mask_numeric_value",
    "mask_numeric_tag": "mask_numeric_tag",
    "mask_pure_label": "mask_pure_label",
}


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_task_name(task: str | None) -> str | None:
    if task is None:
        return None
    return TASK_ALIASES.get(str(task), str(task))


def get_task_probabilities(config: dict) -> dict[str, float]:
    masking = config.get("masking", {})
    raw = masking.get("task_probs", masking.get("task_probabilities", {}))
    return {
        "mask_numeric_value": float(raw.get("numeric_value", raw.get("mask_numeric_value", 0.45))),
        "mask_numeric_tag": float(raw.get("numeric_tag", raw.get("mask_numeric_tag", 0.15))),
        "mask_pure_label": float(raw.get("pure_label", raw.get("mask_pure_label", 0.40))),
    }


def get_pure_label_probabilities(config: dict) -> dict[str, float]:
    masking = config.get("masking", {})
    return dict(masking.get("pure_label_probs", masking.get("pure_label_field_probabilities", {})))


def _weighted_choice(items: list[str], weights: dict[str, float], rng=random) -> str:
    vals = [max(float(weights.get(item, 0.0)), 0.0) for item in items]
    if sum(vals) <= 0:
        vals = [1.0] * len(items)
    return rng.choices(items, weights=vals, k=1)[0]


@dataclass
class VocabBundle:
    numeric_vocab: dict[str, Any]
    pure_vocab: dict[str, Any]
    transform_stats: pd.DataFrame


def load_vocab_bundle(config: dict) -> VocabBundle:
    root = vocab_dir(config)
    return VocabBundle(
        numeric_vocab=load_json(root / "numeric_tag_vocab.json"),
        pure_vocab=load_json(root / "pure_label_vocab.json"),
        transform_stats=pd.read_csv(root / "value_transform_stats.csv"),
    )


class WholeRockTokenDataset(Dataset):
    def __init__(
        self,
        config: dict,
        vocab: VocabBundle,
        mode: str = "train",
        max_samples: int | None = None,
        mask_pure_labels: bool = False,
    ):
        self.config = config
        self.vocab = vocab
        self.mode = mode
        self.mask_pure_labels = mask_pure_labels
        self.chemistry_only = False
        data_cfg = config["data"]
        self.numeric_fields = data_cfg["numeric_fields"]
        self.pure_fields = data_cfg["pure_label_fields"]
        self.sample_col = data_cfg["sample_id_column"]
        cols = [self.sample_col]
        cols += [item["db_column"] for item in self.numeric_fields]
        cols += list(self.pure_fields)
        limit = max_samples if max_samples is not None else data_cfg.get("max_samples")
        limit_clause = f" LIMIT {int(limit)}" if limit else ""
        query = f"SELECT {', '.join(_quote(c) for c in cols)} FROM {_quote(data_cfg['view_name'])}{limit_clause}"
        with sqlite3.connect(resolve_path(data_cfg["sqlite_path"])) as conn:
            self.df = pd.read_sql_query(query, conn)
        self.df, self.coordinate_qc = transform_numeric_frame(self.df, config, vocab)
        self.numeric_mask = config.get('masking', {}).get('numeric_values', {'strategy': 'single'})
        if self.numeric_mask['strategy'] not in {'single', 'observed_fraction'}:
            raise ValueError('Unsupported numeric masking strategy')
        if self.numeric_mask['strategy'] == 'observed_fraction':
            if self.numeric_mask.get('rounding', 'stochastic') != 'stochastic':
                raise ValueError('observed_fraction supports stochastic rounding only')
            if self.numeric_mask.get('scope') != 'numeric_value_task':
                raise ValueError('observed_fraction requires explicit scope=numeric_value_task')
            if not 0 < float(self.numeric_mask['fraction']) <= 1 or int(self.numeric_mask['minimum']) < 1:
                raise ValueError('Invalid numeric mask fraction/minimum')

    def __len__(self) -> int:
        return len(self.df)

    def _numeric_tokens(self, row: pd.Series) -> list[dict]:
        tokens = []
        tag_to_idx = self.vocab.numeric_vocab["tag_to_idx"]
        for item in self.numeric_fields:
            value = row[item["db_column"]]
            if pd.notna(value):
                tokens.append(
                    {
                        "kind": 0,
                        "name": item["name"],
                        "numeric_tag_id": int(tag_to_idx[item["name"]]),
                        "pure_label_id": 0,
                        "value": float(value),
                        "mask_tag": 0.0,
                        "mask_value": 0.0,
                    }
                )
        return tokens

    def _pure_tokens(self, row: pd.Series) -> tuple[list[dict], dict[str, str]]:
        tokens = []
        labels: dict[str, str] = {}
        token_to_idx = self.vocab.pure_vocab["token_to_idx"]
        for field in self.pure_fields:
            value = row.get(field)
            if pd.isna(value):
                continue
            value = str(value).strip()
            token = f"{field}:{value}"
            if value and token in token_to_idx:
                labels[field] = value
                if self.mask_pure_labels:
                    token = self.vocab.pure_vocab["mask_tokens"][field]
                    mask_tag = 1.0
                else:
                    mask_tag = 0.0
                tokens.append(
                    {
                        "kind": 1,
                        "name": token,
                        "numeric_tag_id": 0,
                        "pure_label_id": int(token_to_idx[token]),
                        "value": 0.0,
                        "mask_tag": mask_tag,
                        "mask_value": 0.0,
                        "field": field,
                        "label": value,
                    }
                )
        return tokens, labels

    def _choose_task(
        self,
        numeric_tokens: list[dict],
        labels: dict[str, str],
        rng=random,
        forced_task: str | None = None,
    ) -> str | None:
        eligible = []
        if numeric_tokens:
            eligible.extend(["mask_numeric_value", "mask_numeric_tag"])
        if labels:
            eligible.append("mask_pure_label")
        if not eligible:
            return None
        forced_task = normalize_task_name(forced_task)
        if forced_task in eligible:
            return forced_task
        weights = get_task_probabilities(self.config)
        return _weighted_choice(eligible, weights, rng)

    def __getitem__(self, idx: int | tuple) -> dict:
        forced_task = None
        forced_pure_field = None
        fixed_seed = None
        if isinstance(idx, tuple):
            spec = idx
            idx = spec[0]
            if len(spec) >= 2:
                forced_task = spec[1]
            if len(spec) >= 3:
                forced_pure_field = spec[2]
            if len(spec) >= 4:
                fixed_seed = spec[3]
        rng = random.Random(int(fixed_seed)) if fixed_seed is not None else random
        row = self.df.iloc[idx]
        numeric_tokens = self._numeric_tokens(row)
        pure_tokens, labels = self._pure_tokens(row)
        if self.chemistry_only and self.mode == "extract":
            pure_tokens = []
        tokens = numeric_tokens + pure_tokens
        numeric_targets_multi = []
        target = {
            "task": "none",
            "task_id": -1,
            "mask_position": 0,
            "numeric_value": 0.0,
            "numeric_tag_id": -100,
            "pure_field": "",
            "pure_class_id": -100,
        }
        if self.mode == "train":
            task = self._choose_task(numeric_tokens, labels, rng=rng, forced_task=forced_task)
            if task == "mask_numeric_value":
                candidates = [i for i, t in enumerate(tokens) if t["kind"] == 0]
                if self.numeric_mask['strategy'] == 'observed_fraction':
                    expected = float(self.numeric_mask['fraction']) * len(candidates)
                    count = math.floor(expected) + int(rng.random() < expected % 1)
                    count = min(len(candidates), max(int(self.numeric_mask['minimum']), count))
                    selected = rng.sample(candidates, count)
                else:
                    selected = [rng.choice(candidates)]
                pos = selected[0]
                target.update(
                    task=task,
                    task_id=TASK_TO_ID[task],
                    mask_position=pos,
                    numeric_value=float(tokens[pos]["value"]),
                )
                for selected_pos in selected:
                    numeric_targets_multi.append({'position': selected_pos, 'value': float(tokens[selected_pos]['value'])})
                    tokens[selected_pos] = {**tokens[selected_pos], "value": 0.0, "mask_value": 1.0}
            elif task == "mask_numeric_tag":
                candidates = [i for i, t in enumerate(tokens) if t["kind"] == 0]
                pos = rng.choice(candidates)
                original_id = int(tokens[pos]["numeric_tag_id"])
                target.update(task=task, task_id=TASK_TO_ID[task], mask_position=pos, numeric_tag_id=original_id)
                tokens[pos] = {
                    **tokens[pos],
                    "numeric_tag_id": int(self.vocab.numeric_vocab["mask_numeric_tag_idx"]),
                    "mask_tag": 1.0,
                }
            elif task == "mask_pure_label":
                field_probs = get_pure_label_probabilities(self.config)
                if forced_pure_field in labels:
                    field = forced_pure_field
                else:
                    field = _weighted_choice(list(labels), field_probs, rng)
                candidates = [i for i, t in enumerate(tokens) if t.get("field") == field]
                pos = candidates[0]
                label = labels[field]
                class_id = int(self.vocab.pure_vocab["class_to_idx"][field][label])
                mask_token = self.vocab.pure_vocab["mask_tokens"][field]
                target.update(
                    task=task,
                    task_id=TASK_TO_ID[task],
                    mask_position=pos,
                    pure_field=field,
                    pure_class_id=class_id,
                )
                tokens[pos] = {
                    **tokens[pos],
                    "pure_label_id": int(self.vocab.pure_vocab["token_to_idx"][mask_token]),
                    "mask_tag": 1.0,
                    "name": mask_token,
                }
        if self.config["data"].get("shuffle_tokens", True) and self.mode == "train" and len(tokens) > 1:
            old_pos = target["mask_position"]
            paired = list(enumerate(tokens))
            rng.shuffle(paired)
            tokens = [t for _, t in paired]
            position_map = {old_i: new_i for new_i, (old_i, _) in enumerate(paired)}
            for item in numeric_targets_multi:
                item['position'] = position_map[item['position']]
            if target["task_id"] >= 0:
                target["mask_position"] = next(new_i for new_i, (old_i, _) in enumerate(paired) if old_i == old_pos)
        result = {
            "sample_id": row[self.sample_col],
            "tokens": tokens,
            "target": target,
            "labels": labels,
        }
        if self.numeric_mask['strategy'] == 'observed_fraction':
            result['numeric_value_targets_multi'] = numeric_targets_multi
        return result
