"""Training loop that selects checkpoints using validation machines only."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fm2edge.losses.segmentation import SegmentationLoss
from fm2edge.metrics.segmentation import boundary_f1, confusion_matrix, metrics_from_confusion


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: SegmentationLoss,
    device: torch.device,
    num_classes: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    accumulation: int = 1,
    amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_dice = total_boundary_f1 = 0.0
    sample_count = 0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                output = model(images)
                losses = criterion(output.logits, targets)
                loss = losses["total"]
                if "segmentation" in output.aux_logits:
                    aux = criterion(output.aux_logits["segmentation"], targets)
                    loss = loss + 0.4 * aux["total"]
                scaled_loss = loss / accumulation
            if training:
                assert scaler is not None
                scaler.scale(scaled_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        sample_count += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_ce += float(losses["cross_entropy"].detach()) * batch_size
        total_dice += float(losses["dice"].detach()) * batch_size
        predictions = output.logits.detach().argmax(1).cpu()
        for index in range(batch_size):
            total_boundary_f1 += boundary_f1(
                predictions[index],
                targets[index].detach().cpu(),
                ignore_index=criterion.ignore_index,
            )
            matrix += confusion_matrix(
                predictions[index],
                targets[index].detach().cpu(),
                num_classes,
                criterion.ignore_index,
            )

    count = sample_count
    if count == 0:
        raise ValueError("DataLoader yielded no samples")
    metrics = metrics_from_confusion(matrix)
    return {
        "loss": total_loss / count,
        "seg_loss": total_ce / count,
        "dice_loss": total_dice / count,
        "mIoU": metrics["IoU"],
        "Dice": metrics["Dice"],
        "BoundaryF1": total_boundary_f1 / count,
    }


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    *,
    num_classes: int,
    ignore_index: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    dice_weight: float,
    gradient_accumulation: int,
    amp: bool,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    keep_last_checkpoint: bool = True,
    lightweight_best_checkpoint: bool = False,
) -> Path:
    """Train and return the validation-selected best checkpoint path."""
    output = Path(output_dir)
    checkpoint_dir = output / "checkpoints"
    history_dir = output / "history"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = SegmentationLoss(ignore_index=ignore_index, dice_weight=dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: max(0.0, (1.0 - epoch / max(epochs, 1)) ** 0.9)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_iou = -1.0
    early_stopping_reference = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    best_path = checkpoint_dir / "best.pt"
    history: list[dict[str, float | int]] = []

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output / "tensorboard")
    except ImportError:
        pass

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes,
            optimizer=optimizer,
            scaler=scaler,
            accumulation=gradient_accumulation,
            amp=amp,
        )
        val_metrics = _run_epoch(model, val_loader, criterion, device, num_classes, amp=amp)
        elapsed = time.perf_counter() - started
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": train_metrics["loss"],
            "train_seg_loss": train_metrics["seg_loss"],
            "train_dice_loss": train_metrics["dice_loss"],
            "train_mIoU": train_metrics["mIoU"],
            "val_loss": val_metrics["loss"],
            "val_mIoU": val_metrics["mIoU"],
            "val_Dice": val_metrics["Dice"],
            "val_BoundaryF1": val_metrics["BoundaryF1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time": elapsed,
            "gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(row)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_mIoU": val_metrics["mIoU"],
        }
        if keep_last_checkpoint:
            torch.save(state, checkpoint_dir / "last.pt")
        current_iou = val_metrics["mIoU"]
        significant_improvement = (
            current_iou > early_stopping_reference + early_stopping_min_delta
        )
        if current_iou > best_iou:
            best_iou = current_iou
            best_epoch = epoch
            best_state = (
                {"epoch": epoch, "model": model.state_dict(), "val_mIoU": current_iou}
                if lightweight_best_checkpoint
                else state
            )
            torch.save(best_state, best_path)
        if significant_improvement:
            early_stopping_reference = current_iou
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if writer:
            for key, value in row.items():
                if key != "epoch":
                    writer.add_scalar(key, value, epoch)
        scheduler.step()
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['mIoU']:.4f}"
        )

        with (history_dir / "epochs.csv").open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            csv_writer.writeheader()
            csv_writer.writerows(history)

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early stopping at epoch={epoch:03d}; "
                f"best_epoch={best_epoch:03d} best_val_mIoU={best_iou:.4f}"
            )
            break

    if writer:
        writer.close()
    with (history_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "requested_epochs": epochs,
                "completed_epochs": len(history),
                "best_epoch": best_epoch,
                "best_val_mIoU": best_iou,
                "stopped_early": stopped_early,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "total_epoch_time_seconds": sum(float(row["epoch_time"]) for row in history),
                "mean_epoch_time_seconds": sum(float(row["epoch_time"]) for row in history)
                / len(history),
                "max_training_gpu_memory_mb": max(
                    float(row["gpu_memory_mb"]) for row in history
                ),
                "keep_last_checkpoint": keep_last_checkpoint,
                "lightweight_best_checkpoint": lightweight_best_checkpoint,
            },
            handle,
            indent=2,
        )
    return best_path
