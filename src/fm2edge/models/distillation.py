"""Student adapters preserving the historical inference structure and state keys."""

from __future__ import annotations

import torch


def student_features(student, images, representation: bool = False):
    if representation and hasattr(student, "backbone") and hasattr(student, "head"):
        low, high = student.backbone(images)
        logits, context = student.head(low, high)
        import torch.nn.functional as F

        from fm2edge.models.outputs import ModelOutput

        output = ModelOutput(
            logits=F.interpolate(logits, images.shape[-2:], mode="bilinear", align_corners=False),
            features={"kd": context},
        )
        return output, high
    output = student(images)
    return output, output.features["kd"]


def set_student_stage(student, name: str, stage: str, freeze_representation: bool = True):
    """PIDNet representation includes fusion; its existing segmentation heads stay intact."""
    student.requires_grad_(True)
    student.train()
    if stage not in {"representation", "task"}:
        return
    if name == "mobilenet_v3_lraspp":
        representation_modules = [student.backbone]
        head_modules = [student.head]
    elif name == "pidnet_s":
        head_names = {"final_layer", "seghead_p", "seghead_d"}
        representation_modules = [m for key, m in student.named_children() if key not in head_names]
        head_modules = [getattr(student, key) for key in sorted(head_names)]
    else:
        raise ValueError(f"GKD stage partition is unsupported: {name}")
    if stage == "representation":
        for module in head_modules:
            module.requires_grad_(False).eval()
    elif freeze_representation:
        for module in representation_modules:
            module.requires_grad_(False).eval()


def export_student(student, path, metadata):
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save({"model": student.state_dict(), "model_metadata": metadata}, temporary)
    temporary.replace(destination)
