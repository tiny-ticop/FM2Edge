"""Model factory kept deliberately small and explicit."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn

from fm2edge.models.pidnet import PIDNetSmall
from fm2edge.models.pp_liteseg import PPLiteSegSTDC1

STUDENTS: dict[str, type[nn.Module]] = {
    "pidnet_s": PIDNetSmall,
    "pp_liteseg_stdc1": PPLiteSegSTDC1,
}


def build_student(name: str, num_classes: int, pretrained: str | None = None) -> nn.Module:
    """Build a Student and optionally load a local state dict."""
    normalized = name.lower().replace("-", "_")
    if normalized not in STUDENTS:
        available = ", ".join(sorted(STUDENTS))
        raise ValueError(f"Unknown student {name!r}; available: {available}")
    model = STUDENTS[normalized](num_classes=num_classes)
    if pretrained:
        checkpoint = torch.load(Path(pretrained), map_location="cpu", weights_only=True)
        state_dict = (
            checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        current = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in current and current[key].shape == value.shape
        }
        if not compatible:
            raise ValueError(f"No compatible {normalized} tensors found in {pretrained}")
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        warnings.warn(
            f"Loaded {len(compatible)} tensors from {pretrained}; "
            f"{len(missing)} model tensors remain initialized, {len(unexpected)} were unexpected.",
            stacklevel=2,
        )
    return model
