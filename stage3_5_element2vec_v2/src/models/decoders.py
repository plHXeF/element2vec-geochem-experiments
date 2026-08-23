from __future__ import annotations

from torch import nn


class Decoders(nn.Module):
    def __init__(self, d_model: int, num_numeric_tags: int, pure_label_class_counts: dict[str, int]):
        super().__init__()
        self.numeric_value_decoder = nn.Linear(d_model, 1)
        self.numeric_tag_decoder = nn.Linear(d_model, num_numeric_tags)
        self.pure_label_decoders = nn.ModuleDict(
            {field: nn.Linear(d_model, count) for field, count in pure_label_class_counts.items()}
        )

    def forward(self, masked_hidden):
        return {
            "numeric_value": self.numeric_value_decoder(masked_hidden).squeeze(-1),
            "numeric_tag": self.numeric_tag_decoder(masked_hidden),
            "pure_labels": {field: head(masked_hidden) for field, head in self.pure_label_decoders.items()},
        }
