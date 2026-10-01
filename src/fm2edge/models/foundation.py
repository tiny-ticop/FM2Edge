"""Frozen DINO backbones with small, trainable semantic-segmentation probes."""

from __future__ import annotations

import hashlib
import importlib
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fm2edge.config import ModelConfig
from fm2edge.models.outputs import ModelOutput

MODEL_IDS = {
    ("dinov2", "vits14"): ("dinov2_vits14", 14, 384),
    ("dinov3", "vits16"): ("dinov3_vits16", 16, 384),
}


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit(path: str | None) -> str | None:
    if not path or not (Path(path) / ".git").exists():
        return None
    return subprocess.run(
        ["git", "-C", path, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("model", "teacher", "state_dict"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                value = candidate
                break
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported DINO checkpoint format: {path}")
    return {
        key.removeprefix("module.").removeprefix("backbone."): tensor
        for key, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    }


def build_teacher(config: ModelConfig) -> nn.Module:
    """Build a DINO backbone from an explicitly local official repository."""
    key = (str(config.family), str(config.variant))
    if key not in MODEL_IDS:
        available = ", ".join(f"{family}/{variant}" for family, variant in MODEL_IDS)
        raise ValueError(f"Unsupported foundation model {key}; available: {available}")
    if not config.repository_path:
        raise ValueError("model.repository_path is required for foundation probes")
    repository = Path(config.repository_path)
    if not (repository / "hubconf.py").is_file():
        raise FileNotFoundError(f"Official DINO repository is missing hubconf.py: {repository}")
    model_id, _, _ = MODEL_IDS[key]
    if config.family == "dinov3":
        if config.initialization == "pretrained" and not config.weights_path:
            raise ValueError("DINOv3 pretrained evaluation requires model.weights_path")
        # Import only the official backbone module. The upstream hubconf eagerly
        # imports unrelated detection heads and therefore requires torchvision.
        sys.path.insert(0, str(repository.resolve()))
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
            builder = getattr(backbones, model_id)
            backbone = builder(
                pretrained=config.initialization == "pretrained",
                weights=config.weights_path,
            )
        finally:
            sys.path.remove(str(repository.resolve()))
    else:
        backbone = torch.hub.load(str(repository), model_id, source="local", pretrained=False)
        if config.initialization == "pretrained":
            if not config.weights_path:
                raise ValueError("DINOv2 pretrained evaluation requires model.weights_path")
            state = _load_state_dict(config.weights_path)
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            if unexpected or len(missing) > 8:
                raise ValueError(
                    f"DINOv2 weights are incompatible: missing={missing[:8]}, "
                    f"unexpected={unexpected[:8]}"
                )
    backbone.requires_grad_(False)
    backbone.eval()
    return backbone


class LinearProbe(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features[:, -1])


class LightweightConvProbe(nn.Module):
    def __init__(self, channels: int, layers: int, num_classes: int) -> None:
        super().__init__()
        hidden = 128
        self.decoder = nn.Sequential(
            nn.Conv2d(channels * layers, hidden, 1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features.flatten(1, 2))


class FrozenDinoSegmentor(nn.Module):
    """Keep DINO frozen while training only a dense prediction probe."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        family: str,
        variant: str,
        head: str,
        num_classes: int,
        feature_layers: int,
        embedding_dim: int,
        patch_size: int,
        feature_mode: str = "online",
        artifact_identity: dict[str, object] | None = None,
        random_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.family = family
        self.variant = variant
        self.head_name = head
        self.feature_layers = feature_layers
        self.patch_size = patch_size
        self.feature_mode = feature_mode
        self.artifact_identity = artifact_identity or {}
        self.random_seed = random_seed
        self.probe = (
            LinearProbe(embedding_dim, num_classes)
            if head == "linear"
            else LightweightConvProbe(embedding_dim, feature_layers, num_classes)
        )
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def to(self, *args, **kwargs):
        if self.feature_mode == "cached":
            self.probe.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    def _pad(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = images.shape[-2:]
        padded_height = math.ceil(height / self.patch_size) * self.patch_size
        padded_width = math.ceil(width / self.patch_size) * self.patch_size
        return F.pad(images, (0, padded_width - width, 0, padded_height - height)), (
            padded_height,
            padded_width,
        )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        padded, padded_size = self._pad(images)
        values = self.backbone.get_intermediate_layers(
            padded, n=self.feature_layers, reshape=True, norm=True
        )
        if isinstance(values, torch.Tensor):
            values = (values,)
        features = torch.stack(tuple(values), dim=1)
        return features, padded_size

    def decode(
        self,
        features: torch.Tensor,
        original_size: tuple[int, int],
        padded_size: tuple[int, int],
    ) -> ModelOutput:
        logits = self.probe(features)
        logits = F.interpolate(logits, size=padded_size, mode="bilinear", align_corners=False)
        height, width = original_size
        logits = logits[..., :height, :width]
        return ModelOutput(logits=logits, features={"kd": features[:, -1]})

    def forward(self, images: torch.Tensor) -> ModelOutput:
        features, padded_size = self.encode(images)
        return self.decode(features, tuple(images.shape[-2:]), padded_size)

    def forward_batch(
        self, batch: dict[str, object], images: torch.Tensor, device: torch.device
    ) -> ModelOutput:
        if self.feature_mode != "cached":
            return self(images)
        if "foundation_features" not in batch:
            raise ValueError("Cached foundation run received a batch without cached features")
        features = batch["foundation_features"]
        padded_size = batch["foundation_padded_size"]
        if not isinstance(features, torch.Tensor):
            raise TypeError("foundation_features must collate to a tensor")
        if isinstance(padded_size, (list, tuple)) and len(padded_size) == 2:
            padded = (int(padded_size[0][0]), int(padded_size[1][0]))
        else:
            raise TypeError("foundation_padded_size has an unsupported batch format")
        probe_dtype = next(self.probe.parameters()).dtype
        return self.decode(features.to(device=device, dtype=probe_dtype), tuple(images.shape[-2:]), padded)

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        return self.probe.state_dict()

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "family": self.family,
            "variant": self.variant,
            "head": self.head_name,
            "feature_layers": self.feature_layers,
            "patch_size": self.patch_size,
            "random_seed": self.random_seed,
            "artifact_identity": self.artifact_identity,
        }

    def validate_checkpoint_metadata(self, metadata: dict[str, object]) -> None:
        expected = self.checkpoint_metadata()
        if metadata != expected:
            raise ValueError(
                "Probe checkpoint backbone identity does not match the configured DINO "
                f"artifact. expected={expected}, actual={metadata}"
            )

    def load_checkpoint_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.probe.load_state_dict(state)


def build_foundation_segmentor(
    config: ModelConfig,
    num_classes: int,
    backbone: nn.Module | None = None,
    random_seed: int | None = None,
) -> FrozenDinoSegmentor:
    key = (str(config.family), str(config.variant))
    if key not in MODEL_IDS:
        raise ValueError(f"Unsupported foundation model: {key}")
    _, patch_size, default_dim = MODEL_IDS[key]
    return FrozenDinoSegmentor(
        backbone or build_teacher(config),
        family=key[0],
        variant=key[1],
        head=str(config.head),
        num_classes=num_classes,
        feature_layers=config.feature_layers,
        embedding_dim=config.embedding_dim or default_dim,
        patch_size=patch_size,
        feature_mode=config.feature_mode,
        artifact_identity={
            "repository_commit": _repository_commit(config.repository_path),
            "weights_sha256": _file_sha256(config.weights_path),
            "initialization": config.initialization,
        }
        if backbone is None
        else {"injected_backbone": type(backbone).__name__},
        random_seed=random_seed if config.initialization == "random_init" else None,
    )
