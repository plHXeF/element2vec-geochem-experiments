"""Sample-keyed masks are independent of model width, padding and microbatch size."""
from __future__ import annotations

import numpy as np
import torch

from .schema import FIELDS, GROUPS, FAMILIES, LABELS, EXACT, SUPERVISED
from .prepare import coordinates


def make_batch(data, indices, model, seed=42, epoch=0, training=True, chemistry_only=False, hide=(), forced_task=None, target_tag=None):
    indices = np.asarray(indices, dtype=np.int64)
    values = np.array(data.values[indices], copy=True)
    states = np.array(data.states[indices], copy=True)
    if len(hide):
        states[:, hide] = 0; values[:, hide] = np.nan
    z = coordinates(values, states, data.stats)
    zero_z = coordinates(np.zeros_like(values), np.ones_like(states), data.stats)
    rows = []
    for local, idx in enumerate(indices):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, int(data.sample_ids[idx])]))
        tags = list(np.flatnonzero(states[local] != 0))
        labels = [(j, int(v)) for j, v in enumerate(data.labels[idx]) if v >= 0 and not chemistry_only]
        available = [0] if any(states[local, t] in SUPERVISED for t in tags) else []
        if tags: available.append(1)
        if labels: available.append(2)
        task = -1
        if training and available:
            p = np.array([.45, .15, .40])[available]; p /= p.sum()
            task = forced_task if forced_task in available else int(rng.choice(available, p=p))
        targets = set(); hidden_groups = set(); target_label = -1; target_field = -1
        if target_tag is not None:
            if target_tag not in tags or states[local, target_tag] not in SUPERVISED:
                raise ValueError("Explicit recovery target is absent or unsupported")
            task = 0
        if task == 0:
            groups = sorted({int(GROUPS[t]) for t in tags if states[local, t] in SUPERVISED})
            wanted = len(groups)*.15
            count = max(1, int(wanted) + int(rng.random() < wanted % 1))
            hidden_groups = set(map(int, rng.choice(groups, size=count, replace=False)))
            # One target per independent dependency group; prefer originals but
            # alternate deterministically so derived identities also receive supervision.
            for g in hidden_groups:
                options = [t for t in tags if GROUPS[t] == g and states[local, t] in SUPERVISED]
                targets.add(int(rng.choice(options)))
            if target_tag is not None:
                targets = {int(target_tag)}; hidden_groups = {int(GROUPS[target_tag])}
        elif task == 1:
            target = int(rng.choice(tags)); targets.add(target); hidden_groups.add(int(GROUPS[target]))
        elif task == 2:
            lp = np.array([.15, .25, .60])[[j for j, _ in labels]]; lp /= lp.sum()
            chosen = int(rng.choice(len(labels), p=lp))
            target_field, target_label = labels[chosen]
        tokens = []
        for tag in tags:
            if GROUPS[tag] in hidden_groups and tag not in targets:
                continue
            value_mask = task == 0 and tag in targets
            tag_mask = task == 1 and tag in targets
            tokens.append({"tag_ids": len(FIELDS) if tag_mask else tag, "label_ids": 0,
                           "states": 1 if value_mask or tag_mask else int(states[local, tag]),
                           "families": int(FAMILIES[tag]), "numeric": True, "value_mask": value_mask,
                           "z": 0.0 if value_mask else float(z[local, tag]),
                           "numeric_target": value_mask, "target_z": float(z[local, tag]),
                           "zero_z": float(zero_z[local, tag]), "target_state": int(states[local, tag]),
                           "target_tag": tag, "tag_target": tag_mask, "label_target": False})
        for field, target in labels:
            name = LABELS[field]
            selected = task == 2 and field == target_field
            tokens.append({"tag_ids": len(FIELDS), "label_ids": model.label_mask_ids[name] if selected else model.label_offsets[name]+target,
                           "states": 0, "families": 0, "numeric": False, "value_mask": False, "z": 0.,
                           "numeric_target": False, "target_z": 0., "zero_z": 0., "target_state": 0,
                           "target_tag": 0, "tag_target": False, "label_target": selected})
        if training: rng.shuffle(tokens)
        rows.append((tokens, target_field, target_label, task))
    length = max(1, max(len(r[0]) for r in rows))
    result = {}
    bools = {"numeric", "value_mask", "numeric_target", "tag_target"}
    floats = {"z", "target_z", "zero_z"}
    for key in ("tag_ids", "label_ids", "states", "families", "numeric", "value_mask", "z", "numeric_target", "target_z", "zero_z", "target_state", "target_tag", "tag_target"):
        a = np.zeros((len(rows), length), dtype=bool if key in bools else np.float32 if key in floats else np.int64)
        if key == "tag_ids": a.fill(len(FIELDS))
        for i, (tokens, _, _, _) in enumerate(rows):
            for j, token in enumerate(tokens): a[i, j] = token[key]
        result[key] = torch.from_numpy(a)
    mask = np.zeros((len(rows), length), bool)
    positions = np.full(len(rows), -1, np.int64)
    for i, (tokens, _, _, _) in enumerate(rows):
        mask[i, :len(tokens)] = True
        for j, token in enumerate(tokens):
            if token["label_target"]: positions[i] = j
    result.update(mask=torch.from_numpy(mask), label_position=torch.from_numpy(positions),
                  label_field=torch.tensor([r[1] for r in rows]), label_target=torch.tensor([r[2] for r in rows]),
                  task=torch.tensor([r[3] for r in rows]), weights=torch.tensor(np.array(data.weights[indices]), dtype=torch.float32),
                  row_indices=torch.from_numpy(indices.copy()))
    return result


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}
