"""Evaluate a validation-selected frozen-DINO probe on held-out machines."""

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
from fm2edge.models.foundation import build_foundation_segmentor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config.train.seed)
    configured_split = load_split(config.data.split)
    output = Path(config.output_dir) / config.name / f"fold_{configured_split.fold:02d}"
    split = load_split(output / "split.yaml")
    manifest = output / "manifest.csv"
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    base = ManifestSegmentationDataset(manifest, config.data.root, split.test_machines, transform)
    # Cached features accelerate head training; evaluation remains end-to-end.
    dataset = base
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    model = build_foundation_segmentor(
        config.model, config.data.num_classes, random_seed=config.train.seed
    )
    model.feature_mode = "online"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output / "checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.validate_checkpoint_metadata(checkpoint.get("model_metadata", {}))
    model.load_checkpoint_state_dict(checkpoint["model"])
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
    summary.update(
        {
            "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
            "inference_scope": "end_to_end_backbone_and_probe",
        }
    )
    write_json(summary, output / "metrics/summary.json")
    print(summary)


if __name__ == "__main__":
    main()
