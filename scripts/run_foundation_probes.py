"""Run resumable frozen-DINO probe smoke or full experiments."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml

from fm2edge.analysis.foundation_study import build_foundation_plan
from fm2edge.config import load_config
from fm2edge.data.foundation_cache import cache_identity


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _available(config) -> tuple[bool, str]:
    repository = Path(config.model.repository_path or "")
    if not (repository / "hubconf.py").is_file():
        return False, f"missing official repository: {repository}"
    if config.model.initialization == "pretrained":
        weights = Path(config.model.weights_path or "")
        if not weights.is_file():
            return False, f"missing pretrained weights: {weights}"
    return True, "ready"


def _smoke_config(source: Path, output_root: Path) -> Path:
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    value["name"] = f"smoke__{value['name']}"
    value["output_dir"] = str(output_root / "smoke")
    value["model"]["feature_mode"] = "online"
    value["train"]["epochs"] = 1
    value["train"]["early_stopping_patience"] = None
    value["train"]["num_workers"] = 0
    destination = output_root / "generated/smoke" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)
    return destination


def _write_status(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analyses/foundation_probe.yaml")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--teachers", nargs="+")
    parser.add_argument("--heads", nargs="+", choices=("linear", "lightweight_conv"))
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    plan_path = build_foundation_plan(args.config)
    output_root = plan_path.parent.parent
    plan = _read_plan(plan_path)
    statuses: list[dict[str, object]] = []
    cached_fingerprints: set[str] = set()
    for row in plan:
        fold = int(row["fold"])
        if args.mode == "smoke" and fold != 1:
            continue
        if args.teachers and row["teacher"] not in args.teachers:
            continue
        if args.heads and row["head"] not in args.heads:
            continue
        if args.folds and fold not in args.folds:
            continue
        source = Path(row["config_path"])
        config_path = _smoke_config(source, output_root) if args.mode == "smoke" else source
        config = load_config(config_path)
        ready, reason = _available(config)
        result_dir = (
            Path(config.output_dir) / config.name / f"fold_{fold:02d}"
        )
        summary = result_dir / "metrics/summary.json"
        status: dict[str, object] = {**row, "mode": args.mode, "status": "pending", "detail": ""}
        if not ready:
            status.update(status="skipped_prerequisite", detail=reason)
            print(f"skip {row['run_id']}: {reason}")
            statuses.append(status)
            continue
        if summary.is_file() and not args.rerun:
            status.update(status="completed", detail="existing result")
            print(f"skip completed: {row['run_id']}")
            statuses.append(status)
            continue
        if config.model.feature_mode == "cached":
            identity = cache_identity(config)
            fingerprint = str(identity["fingerprint"])
            cache_summary = (
                Path(config.model.cache_dir or "") / fingerprint[:16] / "cache_summary.json"
            )
            if fingerprint not in cached_fingerprints and not cache_summary.is_file():
                subprocess.run(
                    [sys.executable, "scripts/cache_foundation_features.py", "--config", str(config_path)],
                    check=True,
                )
            cached_fingerprints.add(fingerprint)
        print(f"run: {row['run_id']} ({args.mode})")
        try:
            subprocess.run(
                [sys.executable, "scripts/train_foundation_probe.py", "--config", str(config_path)],
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/evaluate_foundation_probe.py", "--config", str(config_path)],
                check=True,
            )
            status.update(status="completed", detail="")
        except subprocess.CalledProcessError as exc:
            status.update(status="failed", detail=f"exit code {exc.returncode}")
            statuses.append(status)
            _write_status(statuses, output_root / f"run_status_{args.mode}.csv")
            raise
        statuses.append(status)
        _write_status(statuses, output_root / f"run_status_{args.mode}.csv")
    _write_status(statuses, output_root / f"run_status_{args.mode}.csv")


if __name__ == "__main__":
    main()
