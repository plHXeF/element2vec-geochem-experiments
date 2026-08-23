from __future__ import annotations

import torch


def collate_token_batch(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    batch_size = len(batch)
    max_len = max(max(len(item["tokens"]), 1) for item in batch)
    kind = torch.zeros(batch_size, max_len, dtype=torch.long)
    numeric_tag_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    pure_label_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    values = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask_tag = torch.zeros(batch_size, max_len, dtype=torch.float32)
    mask_value = torch.zeros(batch_size, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    numeric_token_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    task_ids = torch.full((batch_size,), -1, dtype=torch.long)
    mask_positions = torch.zeros(batch_size, dtype=torch.long)
    numeric_value_targets = torch.zeros(batch_size, dtype=torch.float32)
    numeric_tag_targets = torch.full((batch_size,), -100, dtype=torch.long)
    pure_class_targets = torch.full((batch_size,), -100, dtype=torch.long)
    sample_ids, labels, pure_fields, task_names = [], [], [], []
    for i, item in enumerate(batch):
        sample_ids.append(item["sample_id"])
        labels.append(item["labels"])
        tokens = item["tokens"]
        for j, tok in enumerate(tokens):
            kind[i, j] = int(tok["kind"])
            numeric_tag_ids[i, j] = int(tok["numeric_tag_id"])
            pure_label_ids[i, j] = int(tok["pure_label_id"])
            values[i, j] = float(tok["value"])
            mask_tag[i, j] = float(tok["mask_tag"])
            mask_value[i, j] = float(tok["mask_value"])
            attention_mask[i, j] = True
            numeric_token_mask[i, j] = int(tok["kind"]) == 0
        target = item["target"]
        task_ids[i] = int(target["task_id"])
        mask_positions[i] = int(target["mask_position"])
        numeric_value_targets[i] = float(target["numeric_value"])
        numeric_tag_targets[i] = int(target["numeric_tag_id"])
        pure_class_targets[i] = int(target["pure_class_id"])
        pure_fields.append(str(target["pure_field"]))
        task_names.append(str(target["task"]))
    return {
        "kind": kind,
        "numeric_tag_ids": numeric_tag_ids,
        "pure_label_ids": pure_label_ids,
        "values": values,
        "mask_tag_flags": mask_tag,
        "mask_value_flags": mask_value,
        "attention_mask": attention_mask,
        "numeric_token_mask": numeric_token_mask,
        "task_ids": task_ids,
        "mask_positions": mask_positions,
        "numeric_value_targets": numeric_value_targets,
        "numeric_tag_targets": numeric_tag_targets,
        "pure_class_targets": pure_class_targets,
        "pure_fields": pure_fields,
        "task_names": task_names,
        "sample_ids": sample_ids,
        "labels": labels,
    }
