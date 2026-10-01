import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from fm2edge.analysis.study import build_study_plan
from fm2edge.config import AugmentationConfig
from fm2edge.data.records import SampleRecord, write_manifest
from fm2edge.data.splits import load_split, make_machine_folds, save_split
from fm2edge.data.transforms import SegmentationTransform


def _base_experiment(path: Path, model: str = "pidnet_s") -> None:
    value = {
        "name": model,
        "output_dir": "results",
        "data": {
            "root": "data",
            "manifest": "unused.csv",
            "split": "unused.yaml",
            "num_classes": 2,
            "image_size": [32, 32],
            "mask_value_map": {0: 0, 1: 1, 255: 255},
        },
        "model": {"name": model, "pretrained": None},
        "train": {"epochs": 2, "batch_size": 2, "num_workers": 0},
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_configurable_study_plan_is_leakage_safe_and_exact(tmp_path: Path) -> None:
    records = [
        SampleRecord(
            sample_id=f"{machine}-{sample}",
            image_path=f"raw/{machine}/delay/90/{sample}.jpg",
            mask_path=f"masks/{machine}/delay/90/{sample}.png",
            machine_id=machine,
            delay="90",
        )
        for machine in [f"machine_{index:02d}" for index in range(5)]
        for sample in range(6)
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    splits = make_machine_folds(records, n_folds=5, seed=42)
    split_dir = tmp_path / "splits"
    for split in splits:
        save_split(split, split_dir / f"fold_{split.fold:02d}.yaml")
    base = tmp_path / "base.yaml"
    _base_experiment(base)
    study = {
        "name": "test",
        "output_dir": str(tmp_path / "output"),
        "data": {"root": str(tmp_path), "manifest": str(manifest), "splits_dir": str(split_dir)},
        "models": {"pidnet_s": str(base)},
        "folds": [1],
        "seeds": [42],
        "sample_seed": 7,
        "training": {"epochs": 2},
        "suites": {
            "reference": {"enabled": True},
            "machine_diversity": {
                "enabled": True,
                "machine_counts": [1, 2, 3],
                "total_images": 3,
                "max_combinations_per_count": None,
            },
            "images_per_machine": {"enabled": True, "counts": [1, "all"]},
            "augmentation": {
                "enabled": True,
                "presets": ["noise"],
                "parameters": {"probability": 1.0},
            },
        },
    }
    study_path = tmp_path / "study.yaml"
    study_path.write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")

    plan_path = build_study_plan(study_path)

    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert json.loads((plan_path.parent / "plan_summary.json").read_text())["total_runs"] == 10
    diversity = [row for row in rows if row["suite"] == "machine_diversity"]
    assert {int(row["train_machine_count"]) for row in diversity} == {1, 2, 3}
    assert all(int(row["train_image_count"]) == 3 for row in diversity)
    for row in rows:
        split = load_split(row["split_path"])
        assert not set(split.train_machines) & set(split.val_machines)
        assert not set(split.train_machines) & set(split.test_machines)
        assert len(split.train_sample_ids) == int(row["train_image_count"])


def test_augmentation_is_seeded_and_never_changes_mask() -> None:
    image = Image.fromarray(np.full((24, 32, 3), 100, dtype=np.uint8))
    mask_array = np.zeros((24, 32), dtype=np.uint8)
    mask_array[:, 16:] = 1
    mask = Image.fromarray(mask_array)
    transform = SegmentationTransform(
        (24, 32),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        mask_value_map={0: 0, 1: 1},
        augmentation=AugmentationConfig(preset="combined", probability=1.0),
    )

    torch.manual_seed(123)
    first_image, first_mask = transform(image, mask)
    torch.manual_seed(123)
    second_image, second_mask = transform(image, mask)

    assert torch.equal(first_image, second_image)
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_mask, torch.from_numpy(mask_array).long())


def test_default_five_machine_study_generates_240_runs(tmp_path: Path) -> None:
    machines = [f"machine_{index:02d}" for index in range(5)]
    records = [
        SampleRecord(
            sample_id=f"{machine}-{sample:03d}",
            image_path=f"raw/{machine}/delay/{90 + sample % 2}/{sample}.jpg",
            mask_path=f"masks/{machine}/delay/{90 + sample % 2}/{sample}.png",
            machine_id=machine,
            delay=str(90 + sample % 2),
        )
        for machine in machines
        for sample in range(50)
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    split_dir = tmp_path / "splits"
    for split in make_machine_folds(records, n_folds=5, seed=42):
        save_split(split, split_dir / f"fold_{split.fold:02d}.yaml")
    model_configs = {}
    for model in ("pidnet_s", "pp_liteseg_stdc1", "mobilenet_v3_lraspp"):
        path = tmp_path / f"{model}.yaml"
        _base_experiment(path, model)
        model_configs[model] = str(path)
    study = yaml.safe_load(
        Path("configs/analyses/machine_generalization.yaml").read_text(encoding="utf-8")
    )
    study["output_dir"] = str(tmp_path / "output")
    study["data"] = {
        "root": str(tmp_path),
        "manifest": str(manifest),
        "splits_dir": str(split_dir),
    }
    study["models"] = model_configs
    study_path = tmp_path / "default_study.yaml"
    study_path.write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")

    plan_path = build_study_plan(study_path)

    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 240
