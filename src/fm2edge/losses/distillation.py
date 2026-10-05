"""Training-only KD objectives. Sources and deliberate adaptations: docs/KD_METHODS.md."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid.to(value.dtype).expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def align_features(
    student: torch.Tensor,
    teacher: torch.Tensor,
    target: torch.Tensor,
    padded_size: tuple[int, int],
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand to padded pixel coordinates, crop, then sample valid-area Teacher grid.

    Unlike resizing a padded Teacher grid straight to the Student grid this preserves
    the right/bottom padding origin. Boundary cells use fractional valid-area weights.
    """
    size = target.shape[-2:]
    grid = teacher.shape[-2:]
    teacher = F.interpolate(teacher.float(), padded_size, mode="bilinear", align_corners=False)
    teacher = teacher[..., : size[0], : size[1]]
    teacher = F.adaptive_avg_pool2d(teacher, grid)
    student = F.interpolate(student.float(), size, mode="bilinear", align_corners=False)
    student = F.adaptive_avg_pool2d(student, grid)
    valid = F.adaptive_avg_pool2d((target != ignore_index).float().unsqueeze(1), grid)
    return student, teacher.detach(), valid


class MaskedGenerativeDistillation(nn.Module):
    def __init__(self, student_channels: int, teacher_channels: int, mask_ratio: float):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.projector = nn.Conv2d(student_channels, teacher_channels, 1)
        self.generator = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(teacher_channels, teacher_channels, 3, padding=1),
        )

    def forward(self, student, teacher, valid):
        projected = self.projector(student)
        mask = torch.rand_like(projected[:, :1]) >= self.mask_ratio
        recovered = self.generator(projected * mask)
        return masked_mean((recovered - teacher.detach()).square(), valid)


class QuerySoftDistillation(nn.Module):
    """GKD Eq. 7-9 feature component, CNN/source-only adaptation, no CLS/masked image."""

    def __init__(self, student_channels: int, teacher_channels: int, max_tokens: int):
        super().__init__()
        self.query = nn.Conv2d(student_channels, teacher_channels, 1)
        self.value = nn.Conv2d(student_channels, teacher_channels, 1)
        self.max_tokens = max_tokens

    def forward(self, student, teacher, valid):
        height, width = student.shape[-2:]
        if height * width > self.max_tokens:
            scale = math.sqrt(self.max_tokens / (height * width))
            grid = (max(1, int(height * scale)), max(1, int(width * scale)))
            student = F.adaptive_avg_pool2d(student, grid)
            teacher = F.adaptive_avg_pool2d(teacher, grid)
            valid = F.adaptive_avg_pool2d(valid, grid)
        query = self.query(student).flatten(2).transpose(1, 2)
        value = self.value(student).flatten(2).transpose(1, 2)
        reference = teacher.detach().flatten(2).transpose(1, 2)
        # Preserve Eq. 7-8 (unscaled dot product), use fp32 for numerical stability.
        affinity = query.float() @ reference.float().transpose(1, 2)
        key_valid = valid.flatten(2).squeeze(1) > 0
        affinity = affinity.masked_fill(~key_valid[:, None, :], -1e4)
        recovered = affinity.softmax(-1) @ value.float()
        return masked_mean((recovered - reference).square().transpose(1, 2), valid.flatten(2))


def logit_kd(student, teacher, valid, temperature):
    teacher_probability = (teacher.detach().float() / temperature).softmax(1)
    loss = F.kl_div(
        (student.float() / temperature).log_softmax(1), teacher_probability, reduction="none"
    ).sum(1, keepdim=True)
    return masked_mean(loss, valid) * temperature**2


def heterogeneous_knowledge(student, teacher, target, ignore_index, temperature):
    """HeteroAKD KMM/KEM Eq. 5-10 with detached targets/reliability and mean reduction."""
    valid = (target != ignore_index).unsqueeze(1)
    labels = F.one_hot(target.clamp(0, student.shape[1] - 1), student.shape[1])
    labels = labels.permute(0, 3, 1, 2).float()
    with torch.no_grad():
        s, t = student.detach().float(), teacher.detach().float()
        student_error = F.binary_cross_entropy_with_logits(s, labels, reduction="none")
        teacher_error = F.binary_cross_entropy_with_logits(t, labels, reduction="none")
        preference = student_error / (teacher_error + student_error).clamp_min(1e-6)
        hybrid = preference * t + (1 - preference) * s
        hybrid_error = F.binary_cross_entropy_with_logits(hybrid, labels, reduction="none")
        discrepancy = (student_error - hybrid_error).clamp_min(0)
        importance = (student_error + discrepancy).softmax(1)
        probability = (hybrid / temperature).softmax(1)
    loss = -(probability * (student.float() / temperature).log_softmax(1) * importance)
    return masked_mean(loss, valid)
