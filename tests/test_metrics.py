import pytest
import torch

from fm2edge.metrics.segmentation import boundary_f1, sample_metrics


def test_perfect_prediction_metrics() -> None:
    target = torch.tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
    logits = torch.full((2, 3, 3), -8.0)
    logits.scatter_(0, target.unsqueeze(0), 8.0)
    metrics = sample_metrics(logits, target, num_classes=2)
    assert metrics["IoU"] == 1.0
    assert metrics["Dice"] == 1.0
    assert metrics["Boundary_F1"] == 1.0


def test_boundary_f1_reconstructs_boundary_across_ignore_band() -> None:
    target = torch.tensor(
        [
            [0, 0, 255, 1, 1],
            [0, 0, 255, 1, 1],
            [0, 0, 255, 1, 1],
        ]
    )

    assert boundary_f1(target.clone(), target, tolerance=0) == pytest.approx(1.0)
    assert boundary_f1(torch.zeros_like(target), target, tolerance=0) == pytest.approx(0.0)


def test_boundary_f1_does_not_create_boundary_inside_same_class_ignore_hole() -> None:
    target = torch.zeros((5, 5), dtype=torch.long)
    target[2, 2] = 255

    assert boundary_f1(torch.zeros_like(target), target, tolerance=0) == pytest.approx(1.0)
