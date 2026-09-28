"""Run the three lightweight baselines over smoke or five-fold experiments."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "pidnet_s": REPOSITORY_ROOT / "configs/experiments/machine_pidnet_s.yaml",
    "pp_liteseg_stdc1": REPOSITORY_ROOT
    / "configs/experiments/machine_pp_liteseg_stdc1.yaml",
    "mobilenet_v3_lraspp": REPOSITORY_ROOT
    / "configs/experiments/machine_mobilenet_v3_lraspp.yaml",
}
SUMMARY_METRICS = (
    "IoU",
    "Dice",
    "Boundary_F1",
    "Precision",
    "Recall",
    "foreground_ratio",
    "prediction_confidence",
    "prediction_entropy",
    "worst_machine_iou",
    "inference_ms_per_image",
    "inference_fps",
    "gpu_peak_memory_mb",
    "model_parameters",
    "checkpoint_size_mb",
)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_results(mode: str, experiment_names: dict[str, str]) -> None:
    """Write fold-level and model-level summaries for every completed run."""
    rows: list[dict[str, object]] = []
    for model_name, experiment_name in experiment_names.items():
        experiment_dir = REPOSITORY_ROOT / "results" / experiment_name
        for fold_dir in sorted(experiment_dir.glob("fold_*")):
            metrics_path = fold_dir / "metrics/summary.json"
            if not metrics_path.is_file():
                continue
            metrics = _load_json(metrics_path)
            training_path = fold_dir / "history/training_summary.json"
            training = _load_json(training_path) if training_path.is_file() else {}
            row: dict[str, object] = {
                "model": model_name,
                "fold": int(fold_dir.name.removeprefix("fold_")),
                "best_epoch": training.get("best_epoch", ""),
                "best_val_mIoU": training.get("best_val_mIoU", ""),
                "completed_epochs": training.get("completed_epochs", ""),
                "training_time_seconds": training.get("total_epoch_time_seconds", ""),
                "mean_epoch_time_seconds": training.get("mean_epoch_time_seconds", ""),
                "max_training_gpu_memory_mb": training.get(
                    "max_training_gpu_memory_mb", ""
                ),
                "num_test_images": metrics.get("num_images", ""),
            }
            row.update({metric: metrics.get(metric, "") for metric in SUMMARY_METRICS})
            rows.append(row)

    output_dir = REPOSITORY_ROOT / "results/machine_baseline_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = output_dir / f"{mode}_runs.csv"
    if rows:
        with runs_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: (str(row["model"]), int(row["fold"]))))

    model_summaries: dict[str, object] = {}
    for model_name in experiment_names:
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        metrics_summary: dict[str, object] = {"completed_folds": len(model_rows)}
        for metric in (
            "best_val_mIoU",
            "training_time_seconds",
            "mean_epoch_time_seconds",
            "max_training_gpu_memory_mb",
            *SUMMARY_METRICS,
        ):
            values = [float(row[metric]) for row in model_rows if row.get(metric) != ""]
            if values:
                metrics_summary[f"{metric}_mean"] = statistics.fmean(values)
                metrics_summary[f"{metric}_std"] = statistics.pstdev(values)
        model_summaries[model_name] = metrics_summary
    with (output_dir / f"{mode}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(model_summaries, handle, indent=2)
    print(f"comparison files: {output_dir}")


def _runtime_config(
    source: Path,
    mode: str,
    fold: int,
    epochs: int,
    batch_size: int | None,
    num_workers: int | None,
    force_cpu: bool,
) -> tuple[dict[str, object], str]:
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base_name = str(config["name"])
    experiment_name = base_name if mode == "full" else base_name.replace("machine_", "machine_smoke_", 1)
    config["name"] = experiment_name
    config["data"]["split"] = f"configs/splits/company/fold_{fold:02d}.yaml"
    config["train"]["epochs"] = epochs
    if mode == "smoke":
        config["train"]["early_stopping_patience"] = None
    if batch_size is not None:
        config["train"]["batch_size"] = batch_size
    if num_workers is not None:
        config["train"]["num_workers"] = num_workers
    if force_cpu:
        config["train"]["device"] = "cpu"
        config["train"]["amp"] = False
    return config, experiment_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU is not available. Install a CUDA-enabled PyTorch build, or use "
            "--allow-cpu only for a short diagnostic run."
        )
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch={torch.__version__}")

    manifest = REPOSITORY_ROOT / "data/company/manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(
            "Missing data/company/manifest.csv. Run the dataset preparation commands first."
        )

    folds = args.folds or ([1] if args.mode == "smoke" else [1, 2, 3, 4, 5])
    if any(fold < 1 or fold > 5 for fold in folds):
        raise ValueError("folds must be between 1 and 5")
    epochs = args.epochs if args.epochs is not None else (2 if args.mode == "smoke" else 50)
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    generated_dir = REPOSITORY_ROOT / "results/_generated_configs" / args.mode
    generated_dir.mkdir(parents=True, exist_ok=True)
    experiment_names: dict[str, str] = {}
    for model_name in args.models:
        for fold in folds:
            split_path = REPOSITORY_ROOT / f"configs/splits/company/fold_{fold:02d}.yaml"
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing split: {split_path}")
            config, experiment_name = _runtime_config(
                CONFIGS[model_name],
                args.mode,
                fold,
                epochs,
                args.batch_size,
                args.num_workers,
                force_cpu=not torch.cuda.is_available() and args.allow_cpu,
            )
            experiment_names[model_name] = experiment_name
            generated_path = generated_dir / f"{model_name}_fold_{fold:02d}.yaml"
            with generated_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(config, handle, sort_keys=False)

            result_dir = REPOSITORY_ROOT / "results" / experiment_name / f"fold_{fold:02d}"
            summary_path = result_dir / "metrics/summary.json"
            if summary_path.is_file() and not args.rerun:
                print(f"skip completed: {model_name} fold {fold:02d}")
                continue
            print(f"run: mode={args.mode} model={model_name} fold={fold:02d} epochs={epochs}")
            subprocess.run(
                [sys.executable, "scripts/train.py", "--config", str(generated_path)],
                cwd=REPOSITORY_ROOT,
                check=True,
            )
            subprocess.run(
                [sys.executable, "scripts/evaluate.py", "--config", str(generated_path)],
                cwd=REPOSITORY_ROOT,
                check=True,
            )
            aggregate_results(args.mode, experiment_names)

    aggregate_results(args.mode, experiment_names)


if __name__ == "__main__":
    main()
