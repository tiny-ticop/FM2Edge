"""Precompute frozen DINO features once for all manifest images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.foundation_cache import (
    cache_file,
    cache_identity,
    image_identity,
    write_cache_metadata,
)
from fm2edge.data.records import read_manifest
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.engine.utils import resolve_device, seed_everything
from fm2edge.models.foundation import build_foundation_segmentor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.model.name != "foundation_probe":
        raise ValueError("cache_foundation_features requires a foundation_probe config")
    if not config.model.cache_dir:
        raise ValueError("model.cache_dir is required")
    seed_everything(config.train.seed)
    records = read_manifest(config.data.manifest)
    machines = sorted({record.machine_id for record in records})
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    dataset = ManifestSegmentationDataset(
        config.data.manifest, config.data.root, machines, transform
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.train.num_workers)
    identity = cache_identity(config)
    metadata_path = write_cache_metadata(config.model.cache_dir, identity)
    device = resolve_device(config.train.device)
    model = build_foundation_segmentor(
        config.model, config.data.num_classes, random_seed=config.train.seed
    )
    model.feature_mode = "online"
    model.to(device).eval()
    started = time.perf_counter()
    written = skipped = 0
    for batch in loader:
        sample_id = str(batch["sample_id"][0])
        destination = cache_file(
            config.model.cache_dir, str(identity["fingerprint"]), sample_id
        )
        source = Path(config.data.root) / Path(str(batch["image_path"][0]))
        if destination.is_file() and not args.overwrite:
            existing = torch.load(destination, map_location="cpu", weights_only=True)
            if (
                existing.get("fingerprint") == identity["fingerprint"]
                and existing.get("image") == image_identity(source)
            ):
                skipped += 1
                continue
        images = batch["image"].to(device)
        features, padded_size = model.encode(images)
        payload = {
            "fingerprint": identity["fingerprint"],
            "sample_id": sample_id,
            "image": image_identity(source),
            "features": features[0].half().cpu(),
            "padded_size": list(padded_size),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        written += 1
    elapsed = time.perf_counter() - started
    summary = {
        "metadata": str(metadata_path),
        "fingerprint": identity["fingerprint"],
        "written": written,
        "skipped": skipped,
        "seconds": elapsed,
        "cache_bytes": sum(
            item.stat().st_size
            for item in metadata_path.parent.rglob("*")
            if item.is_file()
        ),
    }
    (metadata_path.parent / "cache_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(summary)


if __name__ == "__main__":
    main()
