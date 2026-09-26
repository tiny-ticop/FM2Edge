"""Train a Student using train and validation machines only."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
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
from fm2edge.models.registry import build_student


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
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
    train_dataset = ManifestSegmentationDataset(
        config.data.manifest, config.data.root, split.train_machines, transform
    )
    val_dataset = ManifestSegmentationDataset(
        config.data.manifest, config.data.root, split.val_machines, transform
    )
    generator = torch.Generator().manual_seed(config.train.seed)
    if config.train.batch_size < 2 or len(train_dataset) < 2:
        raise ValueError(
            "BatchNorm-based Students require batch_size >= 2 and at least two train samples"
        )
    loader_args = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=len(train_dataset) % config.train.batch_size == 1,
        **loader_args,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    model = build_student(config.model.name, config.data.num_classes, config.model.pretrained)
    device = resolve_device(config.train.device)

    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    shutil.copy2(config.data.split, output / "split.yaml")
    shutil.copy2(config.data.manifest, output / "manifest.csv")
    environment = environment_info()
    environment["manifest_sha256"] = file_sha256(config.data.manifest)
    environment["pretrained_sha256"] = (
        file_sha256(config.model.pretrained) if config.model.pretrained else None
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
    )
    print(f"best validation-selected checkpoint: {best}")


if __name__ == "__main__":
    main()
