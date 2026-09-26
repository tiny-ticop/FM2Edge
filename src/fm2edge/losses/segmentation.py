"""Cross-entropy plus multiclass soft Dice loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255, epsilon: float = 1e-6
) -> torch.Tensor:
    """Compute class-averaged Dice loss while excluding ignored pixels."""
    num_classes = logits.shape[1]
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_float = valid[:, None].float()
    probabilities = torch.softmax(logits, dim=1) * valid_float
    one_hot = one_hot * valid_float
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0


class SegmentationLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        cross_entropy = F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        dice = soft_dice_loss(logits, target, self.ignore_index)
        return {
            "total": cross_entropy + self.dice_weight * dice,
            "cross_entropy": cross_entropy,
            "dice": dice,
        }
