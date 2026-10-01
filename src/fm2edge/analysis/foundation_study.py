"""Plan reproducible frozen-foundation probe experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def load_foundation_study(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key in ("name", "output_dir", "data", "folds", "teachers", "training"):
        if key not in config:
            raise ValueError(f"Foundation study is missing {key!r}")
    return config


def _write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def build_foundation_plan(config_path: str | Path) -> Path:
    study = load_foundation_study(config_path)
    output = Path(study["output_dir"])
    generated = output / "generated/configs"
    plan_dir = output / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for teacher_name, teacher in study["teachers"].items():
        if not teacher.get("enabled", True):
            continue
        for head in teacher["heads"]:
            for fold in study["folds"]:
                fold = int(fold)
                run_id = f"{teacher_name}__{head}__fold_{fold:02d}"
                data = dict(study["data"])
                splits_dir = data.pop("splits_dir")
                data["split"] = str(Path(splits_dir) / f"fold_{fold:02d}.yaml")
                experiment = {
                    "name": run_id,
                    "output_dir": str(output / "runs"),
                    "data": data,
                    "model": {
                        "name": "foundation_probe",
                        "family": teacher["family"],
                        "variant": teacher["variant"],
                        "repository_path": teacher["repository_path"],
                        "weights_path": teacher.get("weights_path"),
                        "head": head,
                        "feature_layers": int(teacher.get("feature_layers", 4)),
                        "initialization": teacher.get("initialization", "pretrained"),
                        "feature_mode": study.get("feature_mode", "cached"),
                        "cache_dir": study.get(
                            "cache_dir", "results/foundation_feature_cache"
                        ),
                    },
                    "train": study["training"],
                    "augmentation": {"preset": "none"},
                }
                config_file = generated / f"{run_id}.yaml"
                _write_yaml(experiment, config_file)
                result_dir = output / "runs" / run_id / f"fold_{fold:02d}"
                runs.append(
                    {
                        "run_id": run_id,
                        "teacher": teacher_name,
                        "family": teacher["family"],
                        "variant": teacher["variant"],
                        "initialization": teacher.get("initialization", "pretrained"),
                        "head": head,
                        "fold": fold,
                        "config_path": config_file.as_posix(),
                        "result_dir": result_dir.as_posix(),
                    }
                )
    if not runs:
        raise ValueError("Foundation study generated no runs")
    plan_path = plan_dir / "experiment_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    summary = {
        "study": study["name"],
        "total_runs": len(runs),
        "teachers": sorted({str(run["teacher"]) for run in runs}),
        "folds": study["folds"],
    }
    (plan_dir / "plan_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return plan_path
