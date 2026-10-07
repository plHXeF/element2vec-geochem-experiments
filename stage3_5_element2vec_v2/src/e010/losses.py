"""Robust exact and censored concentration likelihoods (Student t, nu=4)."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from .schema import LABELS, LEFT_CENSORED, LOWER_BOUND, INDEX


def log_t4_sf(x):
    """Stable log survival; closed form avoids unsupported special CDF kernels."""
    a = x.abs()
    root = torch.sqrt(a*a + 4.)
    r = a/root
    positive = -math.log(4.) + 2*(math.log(4.)-root.log()-(root+a).log()) + (r+2).log()
    return torch.where(x >= 0, positive, torch.log1p(-positive.exp()))


def log_difference(log_a, log_b):
    # Floating point saturation is bounded at the likelihood level, not by
    # pretending that an extreme concentration is another observed value.
    return log_a + torch.log(-torch.expm1((log_b-log_a).clamp(max=-1e-7)))


def concentration_nll(mu, scale, target, state, zero, signed):
    residual = (target-mu)/scale
    lower = (zero-mu)/scale
    normalization = torch.where(signed, torch.zeros_like(mu), log_t4_sf(lower))
    exact = math.log(8/3) + scale.log() + 2.5*torch.log1p(residual.square()/4) + normalization
    # F(target)-F(physical zero) == SF(zero)-SF(target).
    left = -log_difference(log_t4_sf(lower), log_t4_sf(residual)) + normalization
    right = -log_t4_sf(residual) + normalization
    return torch.where(state == LEFT_CENSORED, left, torch.where(state == LOWER_BOUND, right, exact))


def per_sample_loss(outputs, batch, field_weights, recipe="student_t"):
    shape = batch["mask"].shape
    numeric = batch["numeric_target"] & batch["mask"]
    tag = batch["tag_target"] & batch["mask"]
    values = torch.zeros(shape, device=outputs["mu"].device)
    if numeric.any():
        args = [outputs["mu"][numeric], outputs["scale"][numeric], batch["target_z"][numeric],
                batch["target_state"][numeric], batch["zero_z"][numeric], batch["target_tag"][numeric] == INDEX["LOI"]]
        nll = concentration_nll(*args)
        if recipe == "huber":
            exact = (args[3] != LEFT_CENSORED) & (args[3] != LOWER_BOUND)
            huber = F.huber_loss(args[0], args[2], reduction="none")
            nll = torch.where(exact, huber, nll)
        values[numeric] = nll * field_weights[batch["target_tag"][numeric]]
    num_loss = values.sum(1) / numeric.sum(1).clamp_min(1)
    tags = torch.zeros_like(values)
    if tag.any():
        tags[tag] = F.cross_entropy(outputs["tag_logits"][tag], batch["target_tag"][tag], reduction="none")
    tag_loss = tags.sum(1) / tag.sum(1).clamp_min(1)
    label_loss = torch.zeros(len(values), device=values.device)
    for i, field in enumerate(LABELS):
        use = (batch["label_field"] == i) & (batch["task"] == 2)
        if use.any():
            label_loss[use] = F.cross_entropy(outputs["label_logits"][field][use], batch["label_target"][use],
                                             label_smoothing=.05, reduction="none")
    return num_loss + tag_loss + label_loss, {"numeric": num_loss, "tag": tag_loss, "label": label_loss}
