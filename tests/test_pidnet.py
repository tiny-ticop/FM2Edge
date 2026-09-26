import torch

from fm2edge.models import build_student


def test_pidnet_common_output_and_backward() -> None:
    model = build_student("pidnet_s", num_classes=3)
    image = torch.randn(2, 3, 64, 64)
    output = model(image)
    assert output.logits.shape == (2, 3, 64, 64)
    assert output.features["kd"].ndim == 4
    assert output.aux_logits["segmentation"].shape == output.logits.shape
    output.logits.mean().backward()
