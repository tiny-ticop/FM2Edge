"""Execute a generated generalization study safely and resumably."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summary_path(row: dict[str, str]) -> Path:
    return ROOT / row["result_dir"] / "metrics/summary.json"


def _write_status(rows: list[dict[str, str]], output: Path) -> None:
    status_rows = []
    for row in rows:
        log_path = ROOT / row["result_dir"] / "run.log"
        status_rows.append(
            {
                "run_id": row["run_id"],
                "suite": row["suite"],
                "model": row["model"],
                "fold": row["fold"],
                "status": "completed" if _summary_path(row).is_file() else "pending",
                "summary_path": str(_summary_path(row)),
                "log_path": str(log_path),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(status_rows[0]))
        writer.writeheader()
        writer.writerows(status_rows)


def _derived_config(
    row: dict[str, str],
    study_root: Path,
    *,
    smoke_epochs: int | None,
    force_cpu: bool,
) -> Path:
    source = ROOT / row["config_path"]
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    variant = "smoke" if smoke_epochs is not None else "cpu"
    if smoke_epochs is not None:
        config["name"] = f"smoke__{row['run_id']}"
        config["output_dir"] = study_root.joinpath("smoke").as_posix()
        config["train"]["epochs"] = smoke_epochs
        config["train"]["early_stopping_patience"] = None
    if force_cpu:
        config["train"]["device"] = "cpu"
        config["train"]["amp"] = False
        config["train"]["num_workers"] = 0
    generated = study_root / "generated" / variant
    generated.mkdir(parents=True, exist_ok=True)
    path = generated / f"{row['run_id']}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return path


def _run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        default="results/generalization_analysis/plan/experiment_plan.csv",
    )
    parser.add_argument("--suite", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU is unavailable. Install CUDA-enabled PyTorch or use --allow-cpu "
            "only for a small smoke test."
        )
    plan_path = ROOT / args.plan
    study_root = plan_path.parent.parent
    all_rows = _read_plan(plan_path)
    rows = [
        row
        for row in all_rows
        if (args.suite is None or row["suite"] in args.suite)
        and (args.models is None or row["model"] in args.models)
        and (args.folds is None or int(row["fold"]) in args.folds)
    ]
    if args.smoke:
        reference = [row for row in rows if row["suite"] == "reference"] or rows
        first_fold = min(int(row["fold"]) for row in reference)
        selected = []
        for model in sorted({row["model"] for row in reference}):
            selected.append(
                next(
                    row
                    for row in reference
                    if row["model"] == model and int(row["fold"]) == first_fold
                )
            )
        rows = selected
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No plan rows matched the requested filters")
    print(
        f"selected runs={len(rows)}; cuda={torch.cuda.is_available()}; "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    status_path = study_root / "run_status.csv"
    _write_status(all_rows, status_path)
    for index, row in enumerate(rows, start=1):
        if args.smoke:
            config_path = _derived_config(
                row,
                study_root,
                smoke_epochs=args.smoke_epochs,
                force_cpu=not torch.cuda.is_available() and args.allow_cpu,
            )
            with config_path.open("r", encoding="utf-8") as handle:
                smoke_config = yaml.safe_load(handle)
            result_dir = (
                ROOT
                / smoke_config["output_dir"]
                / smoke_config["name"]
                / f"fold_{int(row['fold']):02d}"
            )
            summary = result_dir / "metrics/summary.json"
        else:
            config_path = ROOT / row["config_path"]
            if not torch.cuda.is_available() and args.allow_cpu:
                config_path = _derived_config(
                    row, study_root, smoke_epochs=None, force_cpu=True
                )
            result_dir = ROOT / row["result_dir"]
            summary = _summary_path(row)
        if summary.is_file() and not args.rerun:
            print(f"[{index}/{len(rows)}] skip completed: {row['run_id']}")
            continue
        print(f"[{index}/{len(rows)}] run: {row['run_id']}")
        log_path = result_dir / "run.log"
        try:
            _run_command(
                [sys.executable, "scripts/train.py", "--config", str(config_path)], log_path
            )
            _run_command(
                [sys.executable, "scripts/evaluate.py", "--config", str(config_path)], log_path
            )
        except subprocess.CalledProcessError:
            print(f"failed: {row['run_id']}; see {log_path}")
            _write_status(all_rows, status_path)
            raise
        _write_status(all_rows, status_path)
    print(f"status: {status_path}")


if __name__ == "__main__":
    main()
