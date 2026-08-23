from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    max_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
) -> LambdaLR:
    warmup_steps = max(int(warmup_steps), 0)
    max_steps = max(int(max_steps), 1)
    base_lr = float(optimizer.param_groups[0]["lr"])
    min_ratio = 0.0 if base_lr <= 0 else max(float(min_lr) / base_lr, 0.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
