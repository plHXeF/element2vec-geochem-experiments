from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_losses(outputs: dict, batch: dict, pure_fields: list[str], loss_weights: dict[str, float]) -> dict[str, torch.Tensor]:
    hidden = outputs["masked_hidden"]
    dec = outputs["decoder_outputs"]
    zero = hidden.sum() * 0.0
    task_ids = batch["task_ids"]
    losses = {
        "numeric_value_loss": zero,
        "numeric_tag_loss": zero,
        "rock_group_loss": zero,
        "rock_origin_loss": zero,
        "rock_type_loss": zero,
    }
    value_mask = task_ids == 0
    if 'numeric_prediction_mask' in batch:
        token_mask = batch['numeric_prediction_mask'] & batch['attention_mask'] & batch['numeric_token_mask'] & value_mask[:, None]
        if token_mask.any():
            losses['numeric_value_loss'] = F.huber_loss(outputs['numeric_value_predictions'][token_mask],
                                                       batch['numeric_targets_all'][token_mask], reduction='mean')
    elif value_mask.any():
        losses["numeric_value_loss"] = F.huber_loss(
            dec["numeric_value"][value_mask],
            batch["numeric_value_targets"][value_mask],
            reduction="mean",
        )
    tag_mask = task_ids == 1
    if tag_mask.any():
        losses["numeric_tag_loss"] = F.cross_entropy(
            dec["numeric_tag"][tag_mask],
            batch["numeric_tag_targets"][tag_mask],
        )
    for field in pure_fields:
        field_mask = torch.tensor([f == field for f in batch["pure_fields"]], device=task_ids.device) & (task_ids == 2)
        if field_mask.any():
            losses[f"{field}_loss"] = F.cross_entropy(
                dec["pure_labels"][field][field_mask],
                batch["pure_class_targets"][field_mask],
            )
    total = zero
    total = total + float(loss_weights.get("numeric_value", 1.0)) * losses["numeric_value_loss"]
    total = total + float(loss_weights.get("numeric_tag", 1.0)) * losses["numeric_tag_loss"]
    for field in pure_fields:
        total = total + float(loss_weights.get(field, 1.0)) * losses[f"{field}_loss"]
    losses["total_loss"] = total
    return losses
