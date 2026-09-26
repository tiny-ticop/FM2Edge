"""Stable model output contract for segmentation and distillation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: dict[str, torch.Tensor]
    aux_logits: dict[str, torch.Tensor] = field(default_factory=dict)
