import torch

from fm2edge.metrics.segmentation import sample_metrics


def test_perfect_prediction_metrics() -> None:
    target = torch.tensor([[0, 0, 1], [0, 1, 1], [0, 0, 0]])
    logits = torch.full((2, 3, 3), -8.0)
    logits.scatter_(0, target.unsqueeze(0), 8.0)
    metrics = sample_metrics(logits, target, num_classes=2)
    assert metrics["IoU"] == 1.0
    assert metrics["Dice"] == 1.0
    assert metrics["Boundary_F1"] == 1.0
