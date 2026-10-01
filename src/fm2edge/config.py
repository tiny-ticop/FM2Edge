"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    manifest: str
    root: str
    split: str
    num_classes: int
    image_size: tuple[int, int] = (256, 512)
    ignore_index: int = 255
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    mask_value_map: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "pidnet_s"
    pretrained: str | None = None
    family: str | None = None
    variant: str | None = None
    repository_path: str | None = None
    weights_path: str | None = None
    head: str | None = None
    feature_layers: int = 4
    initialization: str = "pretrained"
    feature_mode: str = "online"
    cache_dir: str | None = None
    embedding_dim: int | None = None


@dataclass(frozen=True)
class AugmentationConfig:
    preset: str = "none"
    probability: float = 0.8
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.05
    noise_std: float = 0.04
    blur_radius: float = 1.5


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 1.0
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    gradient_accumulation: int = 1
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    keep_last_checkpoint: bool = True
    lightweight_best_checkpoint: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple(value: Any, length: int, name: str) -> tuple[Any, ...]:
    result = tuple(value)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML into validated dataclasses."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_raw = dict(raw.get("data", {}))
    data_raw["image_size"] = _tuple(data_raw.get("image_size", (256, 512)), 2, "image_size")
    data_raw["mean"] = _tuple(data_raw.get("mean", (0.485, 0.456, 0.406)), 3, "mean")
    data_raw["std"] = _tuple(data_raw.get("std", (0.229, 0.224, 0.225)), 3, "std")
    data_raw["mask_value_map"] = {
        int(key): int(value) for key, value in data_raw.get("mask_value_map", {}).items()
    }
    config = ExperimentConfig(
        name=str(raw["name"]),
        output_dir=str(raw.get("output_dir", "results")),
        data=DataConfig(**data_raw),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        augmentation=AugmentationConfig(**raw.get("augmentation", {})),
    )
    if config.data.num_classes < 2:
        raise ValueError("num_classes must be at least 2; binary masks use classes 0 and 1")
    if config.train.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    if config.train.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.train.early_stopping_patience is not None:
        if config.train.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 or null")
        if config.train.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
    allowed_presets = {
        "none",
        "brightness_contrast",
        "color",
        "noise",
        "blur",
        "combined",
    }
    if config.augmentation.preset not in allowed_presets:
        raise ValueError(
            f"augmentation.preset must be one of {sorted(allowed_presets)}, "
            f"got {config.augmentation.preset!r}"
        )
    if not 0.0 <= config.augmentation.probability <= 1.0:
        raise ValueError("augmentation.probability must be between 0 and 1")
    if config.model.name == "foundation_probe":
        if config.model.family not in {"dinov2", "dinov3"}:
            raise ValueError("foundation model family must be 'dinov2' or 'dinov3'")
        if config.model.head not in {"linear", "lightweight_conv"}:
            raise ValueError("foundation probe head must be 'linear' or 'lightweight_conv'")
        if config.model.initialization not in {"pretrained", "random_init"}:
            raise ValueError("initialization must be 'pretrained' or 'random_init'")
        if config.model.feature_mode not in {"online", "cached"}:
            raise ValueError("feature_mode must be 'online' or 'cached'")
        if config.model.feature_layers < 1:
            raise ValueError("feature_layers must be >= 1")
        if config.augmentation.preset != "none":
            raise ValueError("foundation probes currently require augmentation.preset='none'")
    return config
