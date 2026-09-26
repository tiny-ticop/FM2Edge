import torch

from fm2edge.models import build_student


def test_mobilenet_v3_lraspp_common_output_and_backward() -> None:
    model = build_student("mobilenet_v3_lraspp", num_classes=3)
    image = torch.randn(2, 3, 64, 64)
    output = model(image)

    assert output.logits.shape == (2, 3, 64, 64)
    assert output.features["kd"].shape == (2, 128, 8, 8)
    assert output.aux_logits == {}
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_218_478
    output.logits.mean().backward()
