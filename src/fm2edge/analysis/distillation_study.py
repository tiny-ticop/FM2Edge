"""Config-driven KD planning and provenance-aware comparison with older experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from fm2edge.distillation_config import METHODS, load_distillation_config


def write_csv(rows, path, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        keys = fields or (list(rows[0]) if rows else ["status"])
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_distillation_plan(path, *, smoke=False):
    study = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("output_dir", "data", "students", "teachers", "methods", "training", "folds"):
        if key not in study:
            raise ValueError(f"Missing study field: {key}")
    root = Path(study["output_dir"])
    root = root / "smoke" if smoke else root
    rows = []
    for student in study["students"]:
        for seed in study.get("seeds", [study["training"].get("seed", 42)]):
            for fold in study["folds"]:
                for method, options in study["methods"].items():
                    if method not in METHODS:
                        raise ValueError(f"Unknown method {method}")
                    if not options.get("enabled", True):
                        continue
                    teachers = {"none": {}} if method == "none" else study["teachers"]
                    for teacher_name, teacher in teachers.items():
                        if not teacher.get("enabled", True):
                            continue
                        if method == "gkd_cnn_source_only" and student == "pp_liteseg_stdc1":
                            continue
                        run_id = (
                            f"{teacher_name}__{student}__{method}__seed_{seed}__fold_{fold:02d}"
                        )
                        data = dict(study["data"])
                        splits = data.pop("splits_dir")
                        data["split"] = str(Path(splits) / f"fold_{fold:02d}.yaml")
                        kd = {k: v for k, v in options.items() if k != "enabled"}
                        kd["method"] = method
                        kd["teacher"] = {
                            k: v
                            for k, v in teacher.items()
                            if k
                            not in {"enabled", "probe_checkpoint_pattern", "probe_config_pattern"}
                        }
                        if method != "none":
                            kd["teacher"].setdefault("name", "foundation_probe")
                            kd["teacher"].setdefault("head", "linear")
                            kd["teacher"]["feature_mode"] = (
                                "online" if smoke else study.get("feature_mode", "cached")
                            )
                            kd["teacher"]["cache_dir"] = study.get(
                                "cache_dir", "results/foundation_feature_cache"
                            )
                        training = {**study["training"], "seed": seed}
                        if smoke:
                            training.update(epochs=1, num_workers=0, early_stopping_patience=None)
                            kd.update(warmup_epochs=0, representation_epochs=1, task_epochs=1)
                        config = {
                            "name": run_id,
                            "output_dir": str(root / "runs"),
                            "data": data,
                            "model": {
                                "name": student,
                                "pretrained": study.get("student_weights", {}).get(student),
                            },
                            "train": training,
                            "augmentation": {"preset": "none"},
                            "distillation": kd,
                        }
                        if method in {"heteroakd", "logit_kd"}:
                            probe_id = f"{teacher_name}__probe__seed_{seed}__fold_{fold:02d}"
                            probe = {
                                "name": probe_id,
                                "output_dir": str(root / "teacher_probes"),
                                "data": data,
                                "model": kd["teacher"],
                                "train": {
                                    **training,
                                    **(study.get("probe_training", {}) if not smoke else {}),
                                },
                                "augmentation": {"preset": "none"},
                            }
                            probe_path = root / "generated/probes" / f"{probe_id}.yaml"
                            write_yaml(probe, probe_path)
                            substitutions = {"fold": fold, "seed": seed, "teacher": teacher_name}
                            kd["probe_config"] = teacher.get(
                                "probe_config_pattern", str(probe_path)
                            ).format(**substitutions)
                            kd["probe_checkpoint"] = teacher.get(
                                "probe_checkpoint_pattern",
                                str(
                                    root
                                    / "teacher_probes"
                                    / probe_id
                                    / f"fold_{fold:02d}/checkpoints/best.pt"
                                ),
                            ).format(**substitutions)
                        config_path = root / "generated/configs" / f"{run_id}.yaml"
                        write_yaml(config, config_path)
                        load_distillation_config(config_path)
                        rows.append(
                            {
                                "run_id": run_id,
                                "teacher": teacher_name,
                                "student": student,
                                "method": method,
                                "seed": seed,
                                "fold": fold,
                                "config_path": str(config_path),
                                "result_dir": str(root / "runs" / run_id / f"fold_{fold:02d}"),
                            }
                        )
    if not rows:
        raise ValueError("No experiments generated")
    plan = root / "plan/experiment_plan.csv"
    write_csv(rows, plan)
    (plan.parent / "summary.json").write_text(
        json.dumps({"runs": len(rows)}, indent=2), encoding="utf-8"
    )
    return plan


def comparison_contract(config_path):
    """Only exact matching known training conditions qualify for paired deltas."""
    from fm2edge.config import load_config
    from fm2edge.data.splits import load_split
    from fm2edge.engine.utils import file_sha256

    config = load_config(config_path)
    split = load_split(config.data.split)
    return {
        "student": config.model.name,
        "split": split.to_dict(),
        "data": {
            key: value
            for key, value in config.to_dict()["data"].items()
            if key not in {"root", "split", "manifest"}
        },
        "manifest": file_sha256(config.data.manifest),
        "train": config.to_dict()["train"],
        "augmentation": config.to_dict()["augmentation"],
        "initial_weight": file_sha256(config.model.pretrained) if config.model.pretrained else None,
    }


def analyze_distillation(plan_path, baseline_root=None):
    with Path(plan_path).open(encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    output = Path(plan_path).parent.parent / "analysis"
    runs, deltas, machines = [], [], []
    baselines = []
    if baseline_root:
        baselines = [p.parent.parent for p in Path(baseline_root).rglob("metrics/summary.json")]
    baselines += [Path(row["result_dir"]) for row in plan if row["method"] == "none"]
    for row in plan:
        directory = Path(row["result_dir"])
        summary_path = directory / "metrics/summary.json"
        if not (directory / "completed.json").exists() or not summary_path.exists():
            continue
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        history_path = directory / "history/training_summary.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        runs.append({**row, **metrics, "training_seconds": history["training_seconds"]})
        per_machine = directory / "metrics/per_machine.csv"
        if per_machine.exists():
            with per_machine.open(encoding="utf-8", newline="") as handle:
                machines.extend({**row, **m} for m in csv.DictReader(handle))
        if row["method"] == "none":
            continue
        _config, kd = load_distillation_config(row["config_path"])
        matched = False
        for baseline in baselines:
            baseline_config = baseline / "config.yaml"
            if not baseline_config.exists() or not (baseline / "metrics/summary.json").exists():
                continue
            baseline_raw = yaml.safe_load(baseline_config.read_text(encoding="utf-8"))
            if baseline_raw.get("distillation", {}).get("method", "none") != "none":
                continue  # never pair a KD run with itself or another KD method
            try:
                strict = comparison_contract(row["config_path"]) == comparison_contract(
                    baseline_config
                )
            except (OSError, ValueError):
                continue
            if not strict:
                continue
            baseline_metrics = json.loads(
                (baseline / "metrics/summary.json").read_text(encoding="utf-8")
            )
            deltas.append(
                {
                    **row,
                    "baseline": str(baseline),
                    "status": "budget_different"
                    if kd.method == "gkd_cnn_source_only"
                    else "matched",
                    "IoU_delta": metrics["IoU"] - baseline_metrics["IoU"],
                }
            )
            matched = True
            break
        if not matched:
            deltas.append(
                {**row, "baseline": "", "status": "no_matching_baseline", "IoU_delta": ""}
            )
    write_csv(runs, output / "runs.csv")
    write_csv(deltas, output / "paired_baseline_deltas.csv")
    write_csv(machines, output / "per_machine.csv")
    if runs:
        import pandas as pd

        frame = pd.DataFrame(runs)
        frame.groupby(["teacher", "student", "method"])[
            ["IoU", "Dice", "Boundary_F1", "worst_machine_iou", "training_seconds"]
        ].agg(["mean", "std", "count"]).to_csv(output / "summary_by_condition.csv")
    if machines:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        frame = pd.DataFrame(machines)
        frame["IoU"] = pd.to_numeric(frame["IoU"])
        frame["condition"] = frame.teacher + "/" + frame.student + "/" + frame.method
        pivot = frame.pivot_table(
            index="machine_id", columns="condition", values="IoU", aggfunc="mean"
        )
        pivot.to_csv(output / "machine_heatmap.csv")
        figure, axis = plt.subplots(figsize=(max(8, len(pivot.columns)), 4))
        plot = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        figure.colorbar(plot, ax=axis, label="IoU")
        figure.tight_layout()
        figure.savefig(output / "machine_heatmap.png")
        plt.close(figure)
    return output
