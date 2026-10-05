"""Run selected KD experiments; automatically prepare fold-specific Teacher probes."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from fm2edge.analysis.distillation_study import build_distillation_plan, write_csv
from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.splits import load_split
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.distillation_config import load_distillation_config
from fm2edge.engine.distillation import prepare_cache
from fm2edge.engine.utils import resolve_device
from fm2edge.models.foundation import build_foundation_segmentor


def prepare_probe(config, kd):
    if kd.method not in {"heteroakd", "logit_kd"}:
        return
    checkpoint = Path(kd.probe_checkpoint)
    if checkpoint.is_file():
        return  # provenance is checked by train_distillation before any KD step
    probe = load_config(kd.probe_config)
    split = load_split(probe.data.split)
    if probe.model.feature_mode == "cached":
        teacher = build_foundation_segmentor(probe.model, probe.data.num_classes)
        transform = SegmentationTransform(
            probe.data.image_size,
            probe.data.mean,
            probe.data.std,
            probe.data.ignore_index,
            probe.data.mask_value_map,
        )
        for machines, samples in (
            (split.train_machines, split.train_sample_ids or None),
            (split.val_machines, None),
        ):
            base = ManifestSegmentationDataset(
                probe.data.manifest, probe.data.root, machines, transform, samples
            )
            prepare_cache(
                probe,
                replace(kd, teacher=probe.model),
                teacher,
                base,
                resolve_device(config.train.device),
            )
        del teacher
    subprocess.run(
        [sys.executable, "scripts/train_foundation_probe.py", "--config", kd.probe_config],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analyses/knowledge_distillation.yaml")
    parser.add_argument("--mode", required=True, choices=["smoke", "full"])
    parser.add_argument("--teachers", nargs="+")
    parser.add_argument("--students", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument(
        "--rerun", action="store_true", help="Restart same identity; never delete results"
    )
    args = parser.parse_args()
    plan = build_distillation_plan(args.config, smoke=args.mode == "smoke")
    status_path = plan.parent.parent / "run_status.csv"
    statuses = {}
    if status_path.exists():
        with status_path.open(encoding="utf-8", newline="") as handle:
            statuses = {row["run_id"]: row for row in csv.DictReader(handle)}
    with plan.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = 0
    for row in rows:
        if any(
            values and row[field] not in values
            for field, values in (
                ("teacher", args.teachers),
                ("student", args.students),
                ("method", args.methods),
            )
        ):
            continue
        if args.folds and int(row["fold"]) not in args.folds:
            continue
        if args.seeds and int(row["seed"]) not in args.seeds:
            continue
        if args.mode == "smoke" and not args.folds and int(row["fold"]) != 1:
            continue
        selected += 1
        config, kd = load_distillation_config(row["config_path"])
        status = {**row, "status": "pending", "detail": ""}
        teacher = kd.teacher
        missing = []
        if kd.method != "none":
            if not (Path(teacher.repository_path or "") / "hubconf.py").exists():
                missing.append("official repository")
            if not Path(teacher.weights_path or "").is_file():
                missing.append("pretrained weights")
        if missing:
            status.update(status="skipped_prerequisite", detail="missing " + ", ".join(missing))
            print(f"skip {row['run_id']}: {status['detail']}")
        else:
            try:
                prepare_probe(config, kd)
                command = [
                    sys.executable,
                    "scripts/train_distillation.py",
                    "--config",
                    row["config_path"],
                ]
                if args.rerun:
                    command += ["--no-resume"]
                if args.train_only:
                    command += ["--train-only"]
                subprocess.run(command, check=True)
                status.update(status="trained" if args.train_only else "completed")
            except (subprocess.CalledProcessError, ValueError, OSError) as exc:
                status.update(status="failed", detail=str(exc))
                statuses[row["run_id"]] = status
                write_csv(list(statuses.values()), status_path)
                raise
        statuses[row["run_id"]] = status
        write_csv(list(statuses.values()), status_path)
    if not selected:
        raise ValueError("No runs matched filters; check --teachers/--methods/--folds")


if __name__ == "__main__":
    main()
