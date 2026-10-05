"""Opt-in KD configuration; historical experiment configuration remains unchanged."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fm2edge.config import ExperimentConfig, ModelConfig, load_config

METHODS = {"none", "mgd", "heteroakd", "logit_kd", "gkd_cnn_source_only"}


@dataclass(frozen=True)
class DistillationConfig:
    method: str = "mgd"
    teacher: ModelConfig = field(default_factory=ModelConfig)
    weight: float = 1.0
    mask_ratio: float = 0.5
    temperature: float = 4.0
    logit_weight: float = 1.0
    projection_supervision_weight: float = 1.0
    warmup_epochs: int = 5
    representation_epochs: int = 25
    task_epochs: int = 25
    max_tokens: int = 512
    freeze_representation: bool = True
    probe_checkpoint: str | None = None
    probe_config: str | None = None


def load_distillation_config(path: str | Path) -> tuple[ExperimentConfig, DistillationConfig]:
    experiment = load_config(path)
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    values = dict(raw.get("distillation", {}))
    teacher_values = dict(values.pop("teacher", {}))
    teacher_values.setdefault("name", "foundation_probe")
    teacher_values.setdefault("head", "linear")
    kd = DistillationConfig(teacher=ModelConfig(**teacher_values), **values)
    if kd.method not in METHODS:
        raise ValueError(f"Unknown distillation method: {kd.method}")
    if experiment.model.name not in {"pidnet_s", "mobilenet_v3_lraspp", "pp_liteseg_stdc1"}:
        raise ValueError("KD requires a supported Student")
    if experiment.augmentation.preset != "none":
        raise ValueError("Initial KD experiments require augmentation.preset=none")
    if kd.method != "none":
        if (kd.teacher.family, kd.teacher.variant) not in {
            ("dinov2", "vits14"),
            ("dinov3", "vits16"),
        }:
            raise ValueError("Teacher must be dinov2/vits14 or dinov3/vits16")
        if kd.teacher.initialization != "pretrained":
            raise ValueError("KD requires a pretrained frozen Teacher")
        if kd.teacher.feature_mode not in {"online", "cached"}:
            raise ValueError("feature_mode must be online or cached")
        if kd.teacher.feature_mode == "cached" and not kd.teacher.cache_dir:
            raise ValueError("Cached KD requires teacher.cache_dir")
        if kd.teacher.head not in {"linear", "lightweight_conv"}:
            raise ValueError("Unsupported Teacher probe head")
    if not 0 <= kd.mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    if min(kd.weight, kd.logit_weight, kd.projection_supervision_weight) < 0:
        raise ValueError("Loss weights must be nonnegative")
    if kd.temperature <= 0 or kd.max_tokens < 2:
        raise ValueError("temperature must be positive and max_tokens >= 2")
    if kd.warmup_epochs < 0 or min(kd.representation_epochs, kd.task_epochs) < 1:
        raise ValueError("Invalid stage epochs")
    if kd.method == "gkd_cnn_source_only" and experiment.model.name == "pp_liteseg_stdc1":
        raise ValueError("GKD representation partition is not implemented for PP-LiteSeg")
    return experiment, kd
