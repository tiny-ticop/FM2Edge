"""Leakage-safe, epoch-resumable KD training and Student-only evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.foundation_cache import (
    CachedFoundationDataset,
    cache_file,
    cache_identity,
    image_identity,
    write_cache_metadata,
)
from fm2edge.data.records import read_manifest
from fm2edge.data.splits import load_split
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.distillation_config import load_distillation_config
from fm2edge.engine.evaluator import evaluate
from fm2edge.engine.trainer import _run_epoch
from fm2edge.engine.utils import environment_info, file_sha256, resolve_device, seed_everything
from fm2edge.losses.distillation import (
    MaskedGenerativeDistillation,
    QuerySoftDistillation,
    align_features,
    heterogeneous_knowledge,
    logit_kd,
)
from fm2edge.losses.segmentation import SegmentationLoss
from fm2edge.models.distillation import export_student, set_student_stage, student_features
from fm2edge.models.foundation import build_foundation_segmentor
from fm2edge.models.registry import build_student


def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_identity(config, kd):
    """Include actual image/mask content, split, weights and probe; never secrets."""
    identity = {"config": config.to_dict(), "distillation": asdict(kd), "schema": 1}
    identity["manifest_sha256"] = file_sha256(config.data.manifest)
    identity["split_sha256"] = file_sha256(config.data.split)
    identity["implementation_sha256"] = {
        name: file_sha256(Path(__file__).parents[1] / name)
        for name in ("engine/distillation.py", "losses/distillation.py", "models/distillation.py")
    }
    digest = hashlib.sha256()
    for record in sorted(read_manifest(config.data.manifest), key=lambda item: item.sample_id):
        digest.update(record.sample_id.encode())
        for relative in (record.image_path, record.mask_path):
            digest.update(file_sha256(Path(config.data.root) / relative).encode())
    identity["dataset_content_sha256"] = digest.hexdigest()
    identity["student_initial_sha256"] = (
        file_sha256(config.model.pretrained) if config.model.pretrained else None
    )
    identity["teacher"] = (
        cache_identity(replace(config, model=kd.teacher)) if kd.method != "none" else None
    )
    identity["probe_sha256"] = file_sha256(kd.probe_checkpoint) if kd.probe_checkpoint else None
    identity = json.loads(json.dumps(identity, default=str))
    encoded = json.dumps(identity, sort_keys=True).encode()
    identity["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return identity


def validate_probe(config, kd, teacher):
    """Refuse a probe trained with different machines, GT subset, or preprocessing."""
    if not kd.probe_checkpoint or not kd.probe_config:
        raise FileNotFoundError("HeteroAKD/logit KD requires probe_checkpoint AND probe_config")
    from fm2edge.config import load_config

    probe = load_config(kd.probe_config)
    directory = Path(kd.probe_checkpoint).parent.parent
    saved_split = directory / "split.yaml"
    saved_manifest = directory / "manifest.csv"
    saved_config = directory / "config.yaml"
    if not all(path.is_file() for path in (saved_split, saved_manifest, saved_config)):
        raise ValueError("Probe provenance snapshots are required: config/split/manifest")
    frozen = load_config(saved_config)
    if probe.to_dict() != frozen.to_dict():
        raise ValueError("Probe config differs from its saved training config")
    if load_split(saved_split).to_dict() != load_split(config.data.split).to_dict():
        raise ValueError("Probe train/val/test machines or train sample subset mismatch")
    if file_sha256(saved_manifest) != file_sha256(config.data.manifest):
        raise ValueError("Probe manifest mismatch")
    for key in ("image_size", "mean", "std", "mask_value_map", "ignore_index", "num_classes"):
        if getattr(probe.data, key) != getattr(config.data, key):
            raise ValueError(f"Probe preprocessing mismatch: {key}")
    if probe.augmentation.preset != "none":
        raise ValueError("Probe must have been trained without augmentation")
    checkpoint = torch.load(kd.probe_checkpoint, map_location="cpu", weights_only=True)
    teacher.validate_checkpoint_metadata(checkpoint["model_metadata"])
    teacher.load_checkpoint_state_dict(checkpoint["model"])


def prepare_cache(config, kd, teacher, base, device):
    """Append train samples only. Cache is shared with historical foundation probes."""
    teacher_config = replace(config, model=kd.teacher)
    identity = cache_identity(teacher_config)
    root = kd.teacher.cache_dir
    metadata_path = write_cache_metadata(root, identity)
    teacher.backbone.to(device).eval()
    started = time.perf_counter()
    written = 0
    for batch in DataLoader(base, batch_size=1, num_workers=0):
        sample_id = batch["sample_id"][0]
        path = cache_file(root, identity["fingerprint"], sample_id)
        source = Path(config.data.root) / batch["image_path"][0]
        source_identity = image_identity(source)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload.get("fingerprint") == identity["fingerprint"]
                and payload.get("image") == source_identity
            ):
                continue
        features, padded_size = teacher.encode(batch["image"].to(device))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "features": features[0].half().cpu(),
                "sample_id": sample_id,
                "padded_size": list(padded_size),
                "image": source_identity,
                "fingerprint": identity["fingerprint"],
            },
            temporary,
        )
        temporary.replace(path)
        written += 1
    teacher.backbone.cpu()
    summary = {
        "written": written,
        "seconds": time.perf_counter() - started,
        "cache_bytes": sum(p.stat().st_size for p in metadata_path.parent.rglob("*.pt")),
        "fingerprint": identity["fingerprint"],
    }
    return CachedFoundationDataset(base, root, identity), summary


def _teacher_features(teacher, batch, images, device):
    if "foundation_features" in batch:
        features = batch["foundation_features"].to(device).float()
        padded = batch["foundation_padded_size"]
        size = (int(padded[0][0]), int(padded[1][0]))
        return features, size
    return teacher.encode(images)


def _save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _rng_state(generator):
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "loader": generator.get_state(),
    }


def _restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    generator.set_state(state["loader"])


def _train_distillation(config_path, *, resume=True, teacher_override=None):
    """Run a KD experiment; test evaluation happens only after validation selection.

    teacher_override is for synthetic tests, never enabled by the company CLI.
    """
    config, kd = load_distillation_config(config_path)
    split = load_split(config.data.split)
    output = Path(config.output_dir) / config.name / f"fold_{split.fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    identity = run_identity(config, kd)
    identity_path = output / "identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("Run identity changed; use a new experiment/output name")
    else:
        write_json(identity, identity_path)
    completion = output / "completed.json"
    if completion.exists() and resume:
        completed = json.loads(completion.read_text(encoding="utf-8"))
        if (
            completed.get("fingerprint") == identity["fingerprint"]
            and (output / "student.pt").exists()
            and (output / "metrics/summary.json").exists()
        ):
            return output
    if completion.exists():
        completion.unlink()
    seed_everything(config.train.seed)
    device = resolve_device(config.train.device)
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    base = ManifestSegmentationDataset(
        config.data.manifest,
        config.data.root,
        split.train_machines,
        transform,
        split.train_sample_ids or None,
    )
    val = ManifestSegmentationDataset(
        config.data.manifest,
        config.data.root,
        split.val_machines,
        transform,
    )
    if config.train.batch_size < 2 or len(base) < 2:
        raise ValueError("BatchNorm Students require batch_size >= 2 and >= 2 train samples")
    student = build_student(config.model.name, config.data.num_classes, config.model.pretrained)
    student.to(device).eval()
    representation = kd.method == "gkd_cnn_source_only"
    with torch.no_grad():
        _, feature = student_features(
            student, base[0]["image"].unsqueeze(0).to(device), representation
        )
    student_channels = feature.shape[1]
    teacher = None
    cache_summary = {"written": 0, "seconds": 0, "cache_bytes": 0}
    train_dataset = base
    if kd.method != "none":
        # Teacher construction must not alter identical Student initialization/data RNG.
        saved_rng = _rng_state(torch.Generator())
        teacher = teacher_override or build_foundation_segmentor(
            kd.teacher, config.data.num_classes, random_seed=config.train.seed
        )
        teacher.requires_grad_(False).eval()
        if kd.method in {"heteroakd", "logit_kd"}:
            validate_probe(config, kd, teacher)
        if kd.teacher.feature_mode == "cached":
            train_dataset, cache_summary = prepare_cache(config, kd, teacher, base, device)
            teacher.probe.to(device)
        else:
            teacher.to(device)
        _restore_rng(saved_rng, torch.Generator())
    channels = kd.teacher.embedding_dim or 384
    if teacher_override is not None and teacher is not None:
        with torch.no_grad():
            channels = teacher.encode(base[0]["image"].unsqueeze(0).to(device))[0].shape[2]
    if kd.method == "mgd":
        adapter = MaskedGenerativeDistillation(student_channels, channels, kd.mask_ratio)
    elif representation:
        adapter = QuerySoftDistillation(student_channels, channels, kd.max_tokens)
    elif kd.method == "heteroakd":
        adapter = torch.nn.Conv2d(student_channels, config.data.num_classes, 1)
    else:
        adapter = torch.nn.Identity()
    adapter.to(device)
    generator = torch.Generator().manual_seed(config.train.seed)
    loader_args = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=len(base) % config.train.batch_size == 1,
        **loader_args,
    )
    val_loader = DataLoader(val, shuffle=False, **loader_args)
    criterion = SegmentationLoss(config.data.ignore_index, config.train.dice_weight)
    stages = (
        [("representation", kd.representation_epochs), ("task", kd.task_epochs)]
        if representation
        else [("joint", config.train.epochs)]
    )
    checkpoint_dir = output / "checkpoints"
    last_path = checkpoint_dir / "last.pt"
    best_path = checkpoint_dir / "best.pt"
    restored = (
        torch.load(last_path, map_location="cpu", weights_only=True)
        if resume and last_path.exists()
        else None
    )
    if restored:
        if restored["fingerprint"] != identity["fingerprint"]:
            raise ValueError("Checkpoint experiment fingerprint mismatch")
        student.load_state_dict(restored["student"])
        adapter.load_state_dict(restored["adapter"])
    history = restored["history"] if restored else []
    history_dir = output / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.data.split, output / "split.yaml")
    shutil.copy2(config.data.manifest, output / "manifest.csv")
    snapshot = {**config.to_dict(), "distillation": asdict(kd)}
    (output / "config.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    write_json(
        {
            **environment_info(),
            "student_parameters": sum(p.numel() for p in student.parameters()),
            "kd_parameters": sum(p.numel() for p in adapter.parameters()),
            "teacher_parameters": sum(p.numel() for p in teacher.parameters()) if teacher else 0,
            "implementation": "FM2Edge DINO adaptations; see docs/KD_METHODS.md",
            "cache": cache_summary,
        },
        output / "environment.json",
    )
    for stage_index, (stage, epochs) in enumerate(stages):
        if restored and stage_index < restored["stage_index"]:
            continue
        set_student_stage(student, config.model.name, stage, kd.freeze_representation)
        parameters = [p for p in student.parameters() if p.requires_grad]
        if stage != "task":
            parameters += list(adapter.parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=config.train.learning_rate, weight_decay=config.train.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e, count=epochs: max(0, 1 - e / count) ** 0.9
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device.type == "cuda")
        best_iou, reference, stale, start = -1.0, -1.0, 0, 1
        if restored and stage_index == restored["stage_index"]:
            optimizer.load_state_dict(restored["optimizer"])
            scheduler.load_state_dict(restored["scheduler"])
            scaler.load_state_dict(restored["scaler"])
            best_iou, reference, stale = (
                restored["best_iou"],
                restored["reference"],
                restored["stale"],
            )
            start = restored["epoch"] + 1
            _restore_rng(restored["rng"], generator)
            if restored["stage_done"]:
                continue
        for epoch in range(start, epochs + 1):
            started = time.perf_counter()
            set_student_stage(student, config.model.name, stage, kd.freeze_representation)
            adapter.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            totals = {"total_loss": 0.0, "seg_loss": 0.0, "kd_loss": 0.0}
            count = 0
            for step, batch in enumerate(train_loader):
                images, target = batch["image"].to(device), batch["mask"].to(device)
                accumulation = config.train.gradient_accumulation
                group_size = min(
                    accumulation, len(train_loader) - (step // accumulation) * accumulation
                )
                with torch.autocast(
                    device_type=device.type, enabled=config.train.amp and device.type == "cuda"
                ):
                    prediction, sf = student_features(student, images, representation)
                    zero = prediction.logits.sum() * 0
                    seg_loss = zero
                    if stage != "representation":
                        seg_loss = criterion(prediction.logits, target)["total"]
                        if "segmentation" in prediction.aux_logits:
                            seg_loss += (
                                0.4
                                * criterion(prediction.aux_logits["segmentation"], target)["total"]
                            )
                    kd_loss = zero
                    warmup = kd.method == "heteroakd" and epoch <= kd.warmup_epochs
                    projected = None
                    if kd.method == "heteroakd":
                        projected = F.interpolate(
                            adapter(sf).float(),
                            target.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        seg_loss += (
                            kd.projection_supervision_weight * criterion(projected, target)["total"]
                        )
                    if stage != "task" and kd.method != "none" and not warmup:
                        with torch.no_grad():
                            tf, padded = _teacher_features(teacher, batch, images, device)
                        if kd.method in {"mgd", "gkd_cnn_source_only"}:
                            sf, reference_features, valid = align_features(
                                sf, tf[:, -1], target, padded, config.data.ignore_index
                            )
                            with torch.autocast(device_type=device.type, enabled=False):
                                kd_loss = adapter(sf.float(), reference_features.float(), valid)
                        else:
                            with torch.no_grad():
                                teacher_logits = teacher.decode(
                                    tf, tuple(target.shape[-2:]), padded
                                ).logits
                            valid = (target != config.data.ignore_index).unsqueeze(1)
                            kd_loss = kd.logit_weight * logit_kd(
                                prediction.logits, teacher_logits, valid, kd.temperature
                            )
                            if projected is not None:
                                kd_loss += heterogeneous_knowledge(
                                    projected,
                                    teacher_logits,
                                    target,
                                    config.data.ignore_index,
                                    kd.temperature,
                                )
                    loss = seg_loss + kd.weight * kd_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite KD loss; inspect weights/features")
                scaler.scale(loss / group_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                batch_size = images.shape[0]
                count += batch_size
                for key, value in (
                    ("total_loss", loss),
                    ("seg_loss", seg_loss),
                    ("kd_loss", kd_loss),
                ):
                    totals[key] += float(value.detach()) * batch_size
            row = {
                "stage": stage,
                "epoch": epoch,
                **{key: value / count for key, value in totals.items()},
                "val_mIoU": None,
                "val_Dice": None,
                "val_BoundaryF1": None,
                "seconds": time.perf_counter() - started,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gpu_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda"
                    else 0
                ),
            }
            if stage != "representation":
                metrics = _run_epoch(
                    student,
                    val_loader,
                    criterion,
                    device,
                    config.data.num_classes,
                    amp=config.train.amp,
                )
                row.update(
                    val_mIoU=metrics["mIoU"],
                    val_Dice=metrics["Dice"],
                    val_BoundaryF1=metrics["BoundaryF1"],
                )
                current = metrics["mIoU"]
                if current > best_iou:
                    best_iou = current
                    export_student(
                        student,
                        best_path,
                        {
                            "fingerprint": identity["fingerprint"],
                            "epoch": epoch,
                            "stage": stage,
                            "val_mIoU": current,
                        },
                    )
                if current > reference + config.train.early_stopping_min_delta:
                    reference, stale = current, 0
                else:
                    stale += 1
            row["seconds"] = time.perf_counter() - started
            history.append(row)
            patience = config.train.early_stopping_patience
            stopped = stage != "representation" and patience is not None and stale >= patience
            scheduler.step()
            state = {
                "fingerprint": identity["fingerprint"],
                "student": student.state_dict(),
                "adapter": adapter.state_dict(),
                "stage_index": stage_index,
                "epoch": epoch,
                "stage_done": stopped or epoch == epochs,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(generator),
                "history": history,
                "best_iou": best_iou,
                "reference": reference,
                "stale": stale,
            }
            _save_checkpoint(state, last_path)
            with (history_dir / "epochs.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
            print(f"stage={stage} epoch={epoch} loss={row['total_loss']:.4f} val={row['val_mIoU']}")
            if stopped:
                break
        restored = None
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    student.load_state_dict(best["model"])
    export_student(student, output / "student.pt", best["model_metadata"])
    write_json(
        {
            "epochs": len(history),
            "training_seconds": sum(r["seconds"] for r in history),
            "stages": stages,
            "best": best["model_metadata"],
            "cache": cache_summary,
        },
        history_dir / "training_summary.json",
    )
    # Free every training-only module before measuring deployment inference.
    del teacher, adapter, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def train_distillation(config_path, *, resume=True, evaluate_test=True, teacher_override=None):
    # Training tensors, optimizer states and adapters leave scope before measuring inference.
    output = _train_distillation(config_path, resume=resume, teacher_override=teacher_override)
    if not evaluate_test or (output / "completed.json").exists():
        return output
    config, _kd = load_distillation_config(config_path)
    seed_everything(config.train.seed)
    split = load_split(output / "split.yaml")
    device = resolve_device(config.train.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    student = build_student(config.model.name, config.data.num_classes)
    checkpoint = torch.load(output / "student.pt", map_location="cpu", weights_only=True)
    student.load_state_dict(checkpoint["model"], strict=True)
    student.to(device)
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    test = ManifestSegmentationDataset(
        output / "manifest.csv", config.data.root, split.test_machines, transform
    )
    loader = DataLoader(
        test,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    summary = evaluate(
        student,
        loader,
        device,
        config.data.num_classes,
        config.data.ignore_index,
        output,
        config.data.mean,
        config.data.std,
        seed=config.train.seed,
        include_foreground_metrics=True,
    )
    summary.update(
        student_parameters=sum(p.numel() for p in student.parameters()),
        checkpoint_size_mb=(output / "student.pt").stat().st_size / 1024**2,
    )
    write_json(summary, output / "metrics/summary.json")
    write_json(
        {"fingerprint": checkpoint["model_metadata"]["fingerprint"]}, output / "completed.json"
    )
    return output
