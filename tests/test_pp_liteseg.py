import torch

from fm2edge.models import build_student


def test_pp_liteseg_stdc1_common_output_and_backward() -> None:
    model = build_student("pp_liteseg_stdc1", num_classes=3)
    image = torch.randn(2, 3, 64, 64)
    output = model(image)

    assert output.logits.shape == (2, 3, 64, 64)
    assert output.features["kd"].shape == (2, 32, 8, 8)
    assert set(output.aux_logits) == {"x16", "x32"}
    assert all(auxiliary.shape == output.logits.shape for auxiliary in output.aux_logits.values())
    assert 7_000_000 < sum(parameter.numel() for parameter in model.parameters()) < 10_000_000
    output.logits.mean().backward()


def test_student_registry_reports_both_implemented_models() -> None:
    try:
        build_student("not_a_model", num_classes=2)
    except ValueError as error:
        assert "mobilenet_v3_lraspp" in str(error)
        assert "pidnet_s" in str(error)
        assert "pp_liteseg_stdc1" in str(error)
    else:
        raise AssertionError("unknown Student should fail")
