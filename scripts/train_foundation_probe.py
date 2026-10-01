"""Train a segmentation probe without updating its frozen DINO backbone."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.foundation_cache import CachedFoundationDataset, cache_identity
from fm2edge.data.splits import load_split
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.engine.trainer import train
from fm2edge.engine.utils import (
    environment_info,
    file_sha256,
    resolve_device,
    seed_everything,
    write_json,
)
from fm2edge.models.foundation import build_foundation_segmentor


def _dataset(config, machines, transform, sample_ids=None):
    base = ManifestSegmentationDataset(
        config.data.manifest,
        config.data.root,
        machines,
        transform,
        sample_ids=sample_ids,
    )
    if config.model.feature_mode == "cached":
        if not config.model.cache_dir:
            raise ValueError("model.cache_dir is required for cached mode")
        return CachedFoundationDataset(base, config.model.cache_dir, cache_identity(config))
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model.name != "foundation_probe":
        raise ValueError("Expected model.name=foundation_probe")
    seed_everything(config.train.seed)
    split = load_split(config.data.split)
    output = Path(config.output_dir) / config.name / f"fold_{split.fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    train_dataset = _dataset(
        config, split.train_machines, transform, split.train_sample_ids or None
    )
    val_dataset = _dataset(config, split.val_machines, transform)
    loader_args = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model = build_foundation_segmentor(
        config.model, config.data.num_classes, random_seed=config.train.seed
    )
    device = resolve_device(config.train.device)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    shutil.copy2(config.data.split, output / "split.yaml")
    shutil.copy2(config.data.manifest, output / "manifest.csv")
    environment = environment_info()
    environment.update(
        {
            "manifest_sha256": file_sha256(config.data.manifest),
            "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "weights_sha256": file_sha256(config.model.weights_path)
            if config.model.weights_path
            else None,
            "cache_identity": cache_identity(config),
        }
    )
    write_json(environment, output / "environment.json")
    best = train(
        model,
        train_loader,
        val_loader,
        device,
        output,
        num_classes=config.data.num_classes,
        ignore_index=config.data.ignore_index,
        epochs=config.train.epochs,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        dice_weight=config.train.dice_weight,
        gradient_accumulation=config.train.gradient_accumulation,
        amp=config.train.amp,
        early_stopping_patience=config.train.early_stopping_patience,
        early_stopping_min_delta=config.train.early_stopping_min_delta,
        keep_last_checkpoint=config.train.keep_last_checkpoint,
        lightweight_best_checkpoint=True,
    )
    print(f"best validation-selected probe checkpoint: {best}")


if __name__ == "__main__":
    main()
