"""Run the one-time final evaluation on held-out test machines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.splits import load_split
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.engine.evaluator import evaluate
from fm2edge.engine.utils import resolve_device, seed_everything, write_json
from fm2edge.models.registry import build_student


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.train.seed)
    configured_split = load_split(config.data.split)
    output = Path(config.output_dir) / config.name / f"fold_{configured_split.fold:02d}"
    frozen_split_path = output / "split.yaml"
    split = load_split(frozen_split_path) if frozen_split_path.is_file() else configured_split
    manifest_path = output / "manifest.csv"
    if not manifest_path.is_file():
        manifest_path = Path(config.data.manifest)
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint else output / "checkpoints" / "best.pt"
    )
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    dataset = ManifestSegmentationDataset(
        manifest_path, config.data.root, split.test_machines, transform
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    model = build_student(config.model.name, config.data.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    device = resolve_device(config.train.device)
    model.to(device)
    summary = evaluate(
        model,
        loader,
        device,
        config.data.num_classes,
        config.data.ignore_index,
        output,
        config.data.mean,
        config.data.std,
        seed=config.train.seed,
    )
    summary["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    summary["checkpoint_size_mb"] = checkpoint_path.stat().st_size / (1024**2)
    write_json(summary, output / "metrics" / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
