from __future__ import annotations

import torch


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1).clamp(min=1e-8)
    entropy = -(probabilities * torch.log(probabilities)).sum(dim=-1)
    return entropy
