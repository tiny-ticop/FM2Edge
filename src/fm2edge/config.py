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


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

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
    )
    if config.data.num_classes < 2:
        raise ValueError("num_classes must be at least 2; binary masks use classes 0 and 1")
    if config.train.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be >= 1")
    return config
