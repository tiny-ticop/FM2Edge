from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fm2edge.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, load_config
from fm2edge.data.foundation_cache import (
    CachedFoundationDataset,
    cache_file,
    cache_identity,
    image_identity,
    write_cache_metadata,
)
from fm2edge.engine.evaluator import evaluate
from fm2edge.engine.trainer import train
from fm2edge.models.foundation import FrozenDinoSegmentor, build_foundation_segmentor


class FakeDino(nn.Module):
    def __init__(self, channels: int = 8, patch_size: int = 14) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, channels, patch_size, stride=patch_size)
        self.last_input_size: tuple[int, int] | None = None

    def get_intermediate_layers(self, images, n, reshape=True, norm=True):
        self.last_input_size = tuple(images.shape[-2:])
        value = self.projection(images)
        return tuple(value + index * 0.01 for index in range(n))


def _model(head: str = "linear", feature_mode: str = "online") -> FrozenDinoSegmentor:
    return FrozenDinoSegmentor(
        FakeDino(),
        family="dinov2",
        variant="vits14",
        head=head,
        num_classes=2,
        feature_layers=4,
        embedding_dim=8,
        patch_size=14,
        feature_mode=feature_mode,
    )


def test_frozen_dino_padding_heads_and_gradients() -> None:
    for head in ("linear", "lightweight_conv"):
        model = _model(head)
        model.train()
        assert not model.backbone.training
        assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
        output = model(torch.randn(2, 3, 416, 640))
        assert output.logits.shape == (2, 2, 416, 640)
        assert model.backbone.last_input_size == (420, 644)
        output.logits.mean().backward()
        assert all(parameter.grad is None for parameter in model.backbone.parameters())
        assert any(parameter.grad is not None for parameter in model.probe.parameters())
        assert all(not key.startswith("backbone") for key in model.checkpoint_state_dict())

    cached = _model(feature_mode="cached").to(dtype=torch.float64)
    assert next(cached.probe.parameters()).dtype == torch.float64
    assert next(cached.backbone.parameters()).dtype == torch.float32


def test_foundation_config_is_backward_compatible(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        "name: old\noutput_dir: results\ndata:\n  manifest: x.csv\n  root: data\n"
        "  split: fold.yaml\n  num_classes: 2\nmodel:\n  name: pidnet_s\n",
        encoding="utf-8",
    )
    assert load_config(baseline).model.name == "pidnet_s"


class TinyDataset(Dataset):
    def __init__(self, root: Path, count: int = 2) -> None:
        self.root = root
        self.count = count
        root.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (19, 15), "gray").save(root / "image.png")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "image": torch.randn(3, 15, 19),
            "mask": torch.randint(0, 2, (15, 19)),
            "sample_id": f"sample-{index}",
            "machine_id": "machine-a",
            "delay": "0",
            "image_path": "image.png",
        }


def test_probe_train_evaluate_checkpoint_excludes_backbone(tmp_path: Path) -> None:
    model = _model()
    loader = DataLoader(TinyDataset(tmp_path / "images"), batch_size=1)
    best = train(
        model,
        loader,
        loader,
        torch.device("cpu"),
        tmp_path / "run",
        num_classes=2,
        ignore_index=255,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0,
        dice_weight=1,
        gradient_accumulation=1,
        amp=False,
        keep_last_checkpoint=False,
        lightweight_best_checkpoint=True,
    )
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    assert checkpoint["model"]
    assert all(not key.startswith("backbone") for key in checkpoint["model"])
    model.validate_checkpoint_metadata(checkpoint["model_metadata"])
    invalid_metadata = dict(checkpoint["model_metadata"])
    invalid_metadata["variant"] = "wrong"
    try:
        model.validate_checkpoint_metadata(invalid_metadata)
    except ValueError as exc:
        assert "identity" in str(exc)
    else:
        raise AssertionError("Mismatched backbone metadata was not rejected")
    model.load_checkpoint_state_dict(checkpoint["model"])
    summary = evaluate(
        model,
        loader,
        torch.device("cpu"),
        2,
        255,
        tmp_path / "run",
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    assert summary["num_images"] == 2
    assert (tmp_path / "run/predictions/random").is_dir()


def test_cache_matches_online_and_rejects_changed_image(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,image_path,mask_path,machine_id,delay,width,height\n", encoding="utf-8")
    config = ExperimentConfig(
        name="cache",
        output_dir=str(tmp_path),
        data=DataConfig(
            manifest=str(manifest), root=str(tmp_path / "images"), split="fold.yaml", num_classes=2
        ),
        model=ModelConfig(
            name="foundation_probe",
            family="dinov2",
            variant="vits14",
            head="linear",
            initialization="random_init",
            feature_mode="cached",
            cache_dir=str(tmp_path / "cache"),
            embedding_dim=8,
        ),
        train=TrainConfig(device="cpu"),
    )
    identity = cache_identity(config)
    write_cache_metadata(config.model.cache_dir, identity)
    base = TinyDataset(tmp_path / "images", count=1)
    online = _model()
    image = base[0]["image"].unsqueeze(0)
    features, padded_size = online.encode(image)
    destination = cache_file(config.model.cache_dir, identity["fingerprint"], "sample-0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = base.root / "image.png"
    torch.save(
        {
            "fingerprint": identity["fingerprint"],
            "features": features[0].half(),
            "padded_size": padded_size,
            "image": image_identity(source),
        },
        destination,
    )
    cached_dataset = CachedFoundationDataset(base, config.model.cache_dir, identity)
    batch = next(iter(DataLoader(cached_dataset, batch_size=1)))
    cached_model = _model(feature_mode="cached")
    cached_model.probe.load_state_dict(online.probe.state_dict())
    actual = cached_model.forward_batch(batch, batch["image"], torch.device("cpu")).logits
    expected = online(image).logits
    assert torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
    source.write_bytes(source.read_bytes() + b"changed")
    try:
        cached_dataset[0]
    except ValueError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("Changed source image was not rejected")


def test_builder_accepts_injected_backbone() -> None:
    for family, variant in (("dinov2", "vits14"), ("dinov3", "vits16")):
        config = ModelConfig(
            name="foundation_probe",
            family=family,
            variant=variant,
            head="linear",
            embedding_dim=8,
        )
        assert isinstance(
            build_foundation_segmentor(config, 2, FakeDino()), FrozenDinoSegmentor
        )
