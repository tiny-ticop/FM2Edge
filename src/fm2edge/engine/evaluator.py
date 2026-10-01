"""Evaluation at image, delay, and machine granularity."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from fm2edge.engine.trainer import _forward_batch
from fm2edge.metrics.segmentation import confusion_matrix, metrics_from_confusion, sample_metrics

METRIC_COLUMNS = (
    "IoU",
    "Dice",
    "Boundary_F1",
    "Precision",
    "Recall",
    "foreground_ratio",
    "target_foreground_ratio",
    "prediction_confidence",
    "prediction_entropy",
)


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: Iterable[dict[str, object]],
    key: str,
    confusions: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    output = []
    for name, values in sorted(groups.items()):
        aggregate: dict[str, object] = {key: name, "num_images": len(values)}
        for metric in METRIC_COLUMNS:
            aggregate[metric] = float(np.mean([float(value[metric]) for value in values]))
        aggregate.update(metrics_from_confusion(confusions[name]))
        output.append(aggregate)
    return output


def _colorize(mask: np.ndarray) -> Image.Image:
    palette = np.array(
        [[0, 0, 0], [230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48]],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[mask % len(palette)])


def _save_prediction(
    image_tensor: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    row: dict[str, object],
    path: Path,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> None:
    image = (
        image_tensor.cpu() * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
    )
    image_array = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    input_image = Image.fromarray(image_array)
    gt_image = _colorize(target.cpu().numpy().astype(np.int64))
    pred_image = _colorize(prediction.cpu().numpy().astype(np.int64))
    error = Image.fromarray(
        ((target.cpu().numpy() != prediction.cpu().numpy()) * 255).astype(np.uint8)
    ).convert("RGB")
    width, height = input_image.size
    display_width = max(width, 160)
    display_height = round(height * display_width / width)
    panels = (input_image, gt_image, pred_image, error)
    panels = tuple(
        panel.resize((display_width, display_height), Image.Resampling.NEAREST) for panel in panels
    )
    header_height = 55
    canvas = Image.new("RGB", (display_width * 4, display_height + header_height), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * display_width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 4), f"machine={row['machine_id']}  delay={row['delay']}", fill="black")
    draw.text(
        (5, 20),
        f"IoU={float(row['IoU']):.3f}  Dice={float(row['Dice']):.3f}  "
        f"Boundary F1={float(row['Boundary_F1']):.3f}",
        fill="black",
    )
    for index, label in enumerate(("Input", "Ground truth", "Prediction", "Error map")):
        draw.text((index * display_width + 5, 38), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    output_dir: str | Path,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    save_predictions: bool = True,
    seed: int = 42,
) -> dict[str, float]:
    model.eval()
    output = Path(output_dir)
    rows: list[dict[str, object]] = []
    machine_confusions: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros((num_classes, num_classes), dtype=torch.long)
    )
    delay_confusions: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros((num_classes, num_classes), dtype=torch.long)
    )
    candidates: list[tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]] = []
    inference_seconds = 0.0
    inference_images = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        if batch_index == 0:
            # Exclude one-time CUDA kernel/setup overhead from the reported latency.
            _forward_batch(model, batch, images, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits = _forward_batch(model, batch, images, device).logits
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        inference_images += images.shape[0]
        predictions = logits.argmax(dim=1)
        for index in range(images.shape[0]):
            metrics = sample_metrics(logits[index], targets[index], num_classes, ignore_index)
            row: dict[str, object] = {
                "sample_id": batch["sample_id"][index],
                "image_path": batch["image_path"][index],
                "machine_id": batch["machine_id"][index],
                "delay": batch["delay"][index],
                **metrics,
            }
            rows.append(row)
            matrix = confusion_matrix(
                predictions[index].cpu(), targets[index].cpu(), num_classes, ignore_index
            )
            machine_confusions[str(row["machine_id"])] += matrix
            delay_confusions[str(row["delay"])] += matrix
            if save_predictions:
                candidates.append(
                    (row, images[index].cpu(), targets[index].cpu(), predictions[index].cpu())
                )

    _write_csv(rows, output / "metrics" / "per_image.csv")
    machine_rows = _aggregate(rows, "machine_id", machine_confusions)
    delay_rows = _aggregate(rows, "delay", delay_confusions)
    _write_csv(machine_rows, output / "metrics" / "per_machine.csv")
    _write_csv(delay_rows, output / "metrics" / "per_delay.csv")
    summary = {
        metric: float(np.mean([float(row[metric]) for row in machine_rows]))
        for metric in METRIC_COLUMNS
    }
    machine_ious = [float(row["IoU"]) for row in machine_rows]
    summary.update(
        {
            "num_images": len(rows),
            "num_machines": len(machine_rows),
            "machine_iou_std": float(np.std(machine_ious)) if machine_ious else 0.0,
            "worst_machine_iou": min(machine_ious) if machine_ious else 0.0,
            "inference_ms_per_image": (
                inference_seconds * 1000.0 / inference_images if inference_images else 0.0
            ),
            "inference_fps": (
                inference_images / inference_seconds if inference_seconds > 0 else 0.0
            ),
            "gpu_peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    with (output / "metrics" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if candidates:
        worst = sorted(candidates, key=lambda item: float(item[0]["IoU"]))[:5]
        best = sorted(candidates, key=lambda item: float(item[0]["IoU"]), reverse=True)[:5]
        random_items = random.Random(seed).sample(candidates, min(5, len(candidates)))
        for category, items in (("worst", worst), ("best", best), ("random", random_items)):
            for index, (row, image, target, prediction) in enumerate(items):
                _save_prediction(
                    image,
                    target,
                    prediction,
                    row,
                    output / "predictions" / category / f"{index:02d}_{row['sample_id']}.png",
                    mean,
                    std,
                )
    return summary
