"""Per-image segmentation metrics, including boundary F1."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def confusion_matrix(
    prediction: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255
) -> torch.Tensor:
    valid = target != ignore_index
    encoded = target[valid].long() * num_classes + prediction[valid].long()
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.double()
    true_positive = matrix.diag()
    false_positive = matrix.sum(0) - true_positive
    false_negative = matrix.sum(1) - true_positive
    support = matrix.sum(1)
    iou = true_positive / (true_positive + false_positive + false_negative).clamp_min(1)
    dice = 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    present = support > 0
    return {
        "IoU": iou[present].mean().item() if present.any() else 0.0,
        "Dice": dice[present].mean().item() if present.any() else 0.0,
        "Precision": precision[present].mean().item() if present.any() else 0.0,
        "Recall": recall[present].mean().item() if present.any() else 0.0,
    }


def _boundary(labels: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Return pixels adjacent to a different valid semantic class."""
    valid = labels != ignore_index
    boundary = torch.zeros_like(valid)
    vertical = (labels[1:, :] != labels[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary


def boundary_f1(
    prediction: torch.Tensor, target: torch.Tensor, tolerance: int = 2, ignore_index: int = 255
) -> float:
    """Compute multiclass semantic boundary F1 with a pixel tolerance."""
    prediction = prediction.masked_fill(target == ignore_index, ignore_index)
    pred_boundary = _boundary(prediction, ignore_index)
    true_boundary = _boundary(target, ignore_index)
    if not pred_boundary.any() and not true_boundary.any():
        return 1.0
    if not pred_boundary.any() or not true_boundary.any():
        return 0.0
    kernel = tolerance * 2 + 1
    pred_dilated = F.max_pool2d(pred_boundary.float()[None, None], kernel, 1, tolerance).bool()[
        0, 0
    ]
    true_dilated = F.max_pool2d(true_boundary.float()[None, None], kernel, 1, tolerance).bool()[
        0, 0
    ]
    precision = (pred_boundary & true_dilated).sum().float() / pred_boundary.sum().clamp_min(1)
    recall = (true_boundary & pred_dilated).sum().float() / true_boundary.sum().clamp_min(1)
    return (2 * precision * recall / (precision + recall).clamp_min(1e-8)).item()


def sample_metrics(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255
) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=0)
    prediction = probabilities.argmax(dim=0)
    matrix = confusion_matrix(prediction, target, num_classes, ignore_index)
    metrics = metrics_from_confusion(matrix)
    valid = target != ignore_index
    pixel_confidence = probabilities.max(dim=0).values
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=0)
    foreground = prediction > 0
    metrics.update(
        {
            "Boundary_F1": boundary_f1(prediction, target, ignore_index=ignore_index),
            "foreground_ratio": foreground[valid].float().mean().item() if valid.any() else 0.0,
            "prediction_confidence": pixel_confidence[valid].mean().item() if valid.any() else 0.0,
            "prediction_entropy": entropy[valid].mean().item() if valid.any() else 0.0,
        }
    )
    return metrics
