"""Optional same-sample inference panels; writes results only, never training inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from fm2edge.analysis.distillation_study import write_csv
from fm2edge.analysis.pretrained_study import mismatch
from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.splits import load_split
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.engine.evaluator import _save_prediction
from fm2edge.metrics.segmentation import sample_metrics
from fm2edge.models.registry import build_student


def prediction_comparisons(runs, output, samples=3):
    """Select first sorted test IDs, independent of performance; CPU inference only."""
    output = Path(output) / "same_sample_predictions"
    output.mkdir(parents=True, exist_ok=True)
    index = []
    for treatment in runs:
        if treatment["initialization"] != "imagenet" or treatment["method"] == "none":
            continue
        selected = []
        for init, method in (
            ("random", "none"),
            ("random", treatment["method"]),
            ("imagenet", "none"),
            ("imagenet", treatment["method"]),
        ):
            candidates = [
                r
                for r in runs
                if r["initialization"] == init
                and r["method"] == method
                and (r["student"], r["fold"], r["seed"])
                == (treatment["student"], treatment["fold"], treatment["seed"])
                and (method == "none" or r["teacher"] == treatment["teacher"])
                and not mismatch(
                    treatment, r, same_kd=method != "none", same_initialization=init == "imagenet"
                )
            ]
            if len(candidates) == 1:
                selected.append(candidates[0])
        if len(selected) < 2:
            continue
        case = f"{treatment['student']}__{treatment['teacher']}__{treatment['method']}"
        case += f"__seed_{treatment['seed']}__fold_{int(treatment['fold']):02d}"
        panels = {}
        try:
            with torch.inference_mode():
                for run in selected:
                    directory = Path(run["directory"])
                    config = load_config(directory / "config.yaml")
                    split = load_split(directory / "split.yaml")
                    transform = SegmentationTransform(
                        config.data.image_size,
                        config.data.mean,
                        config.data.std,
                        config.data.ignore_index,
                        config.data.mask_value_map,
                    )
                    dataset = ManifestSegmentationDataset(
                        directory / "manifest.csv", config.data.root, split.test_machines, transform
                    )
                    # Dataset ordering is not a ranking; sort by sample ID explicitly.
                    ordered = sorted(
                        range(len(dataset)), key=lambda i: dataset.records[i].sample_id
                    )
                    model = build_student(config.model.name, config.data.num_classes).eval()
                    checkpoint = directory / "student.pt"
                    if not checkpoint.exists():
                        checkpoint = directory / "checkpoints/best.pt"
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    model.load_state_dict(state["model"], strict=True)
                    label = run["initialization"] + "/" + run["method"]
                    for i in ordered[:samples]:
                        item = dataset[i]
                        logits = model(item["image"].unsqueeze(0)).logits[0]
                        metrics = sample_metrics(
                            logits, item["mask"], config.data.num_classes, config.data.ignore_index
                        )
                        row = {**metrics, "machine_id": item["machine_id"], "delay": item["delay"]}
                        sid = item["sample_id"]
                        filename = hashlib.sha256(sid.encode()).hexdigest()[:16]
                        path = output / case / (label.replace("/", "_") + "__" + filename + ".png")
                        _save_prediction(
                            item["image"],
                            item["mask"],
                            logits.argmax(0),
                            row,
                            path,
                            config.data.mean,
                            config.data.std,
                        )
                        with Image.open(path) as image:
                            panels.setdefault(sid, []).append((label, image.copy()))
                    del model, state
            for sid, images in panels.items():
                width = max(img.width for _, img in images)
                height = sum(img.height + 25 for _, img in images)
                canvas = Image.new("RGB", (width, height), "white")
                draw = ImageDraw.Draw(canvas)
                y = 0
                for label, img in images:
                    draw.text((5, y + 4), label + " | " + sid, fill="black")
                    canvas.paste(img, (0, y + 25))
                    y += img.height + 25
                target = (
                    output
                    / case
                    / (hashlib.sha256(sid.encode()).hexdigest()[:16] + "__comparison.png")
                )
                canvas.save(target)
                index.append(
                    {"case": case, "sample_id": sid, "path": str(target), "status": "saved"}
                )
        except (OSError, ValueError, RuntimeError) as exc:
            index.append(
                {"case": case, "sample_id": "", "status": "unavailable", "detail": str(exc)}
            )
    write_csv(index, output / "index.csv", ["case", "sample_id", "path", "status", "detail"])
    return output
