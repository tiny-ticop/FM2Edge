"""Additive Phase5.5 planning and artifact-only, condition-checked comparisons."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from fm2edge.analysis.distillation_study import build_distillation_plan, write_csv, write_yaml
from fm2edge.config import AugmentationConfig, DataConfig, TrainConfig
from fm2edge.distillation_config import DistillationConfig, load_distillation_config
from fm2edge.engine.utils import file_sha256
from fm2edge.models.imagenet_conversion import SOURCES, validate_converted

METRICS = [
    "IoU",
    "Foreground_IoU",
    "Dice",
    "Boundary_F1",
    "Precision",
    "Recall",
    "Foreground_Recall",
    "worst_machine_iou",
    "training_seconds",
]


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_pretrained_plan(path, *, smoke=False, phase5_plan=None, students=None):
    study = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if students:
        if set(students) - set(study["students"]):
            raise ValueError("Unknown Student filter")
        study["students"] = list(students)
    if set(study["students"]) - set(SOURCES):
        raise ValueError("Phase5.5 only supports PIDNet-S and MobileNetV3")
    root = Path(study["output_dir"]) / ("smoke" if smoke else "")
    audits = {
        name: validate_converted(name, study["student_weights"][name]) for name in study["students"]
    }
    if not phase5_plan:
        effective = root / "generated/study.yaml"
        write_yaml(study, effective)
        plan = build_distillation_plan(effective, smoke=smoke)
    else:
        reference = Path(phase5_plan)
        original_root = reference.parent.parent.resolve()
        if root.resolve() == original_root or original_root.is_relative_to(root.resolve()):
            raise ValueError("Phase5.5 output must not overlap Phase5 output")
        rows = []
        for row in read_rows(reference):
            if row["student"] not in study["students"]:
                continue
            if int(row["fold"]) not in study["folds"] or int(row["seed"]) not in study["seeds"]:
                continue
            if row["method"] not in study["methods"]:
                continue
            if not study["methods"][row["method"]].get("enabled", True):
                continue
            if row["method"] != "none" and (
                row["teacher"] not in study["teachers"]
                or not study["teachers"][row["teacher"]].get("enabled", True)
            ):
                continue
            raw = yaml.safe_load(Path(row["config_path"]).read_text(encoding="utf-8"))
            if raw["model"].get("pretrained"):
                raise ValueError("Phase5 reference must use random Student initialization")
            raw = copy.deepcopy(raw)
            raw["name"] = "imagenet__" + row["run_id"]
            raw["output_dir"] = str(root / "runs")
            raw["model"]["pretrained"] = study["student_weights"][row["student"]]
            if smoke:
                raw["train"].update(epochs=1, num_workers=0, early_stopping_patience=None)
                raw["distillation"].update(warmup_epochs=0, representation_epochs=1, task_epochs=1)
                if row["method"] != "none":
                    raw["distillation"]["teacher"]["feature_mode"] = "online"
            target = root / "generated/configs" / f"{raw['name']}.yaml"
            write_yaml(raw, target)
            load_distillation_config(target)
            rows.append(
                {
                    **row,
                    "run_id": raw["name"],
                    "config_path": str(target),
                    "result_dir": str(root / "runs" / raw["name"] / f"fold_{int(row['fold']):02d}"),
                }
            )
        if not rows:
            raise ValueError("No Phase5 reference runs matched configured conditions")
        plan = root / "plan/experiment_plan.csv"
        write_csv(rows, plan)
    (root / "weight_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    rows = read_rows(plan)
    (plan.parent / "summary.json").write_text(
        json.dumps(
            {
                "runs": len(rows),
                "initialization": "imagenet",
                "phase5_plan": str(phase5_plan or ""),
                "source_ids": {k: v["source_id"] for k, v in audits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return plan


def artifact(directory, initialization):
    """Use saved snapshots, not paths pointing to original training-machine files."""
    directory = Path(directory)
    if not (directory / "completed.json").exists():
        return None
    raw = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    identity_path = directory / "identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    data = dict(raw["data"])
    for key in ("root", "manifest", "split"):
        data.pop(key, None)
    defaults = asdict(DataConfig("", "", "", data["num_classes"]))
    for key in ("root", "manifest", "split"):
        defaults.pop(key)
    data = {**defaults, **data}
    train = {**asdict(TrainConfig()), **raw["train"]}
    seed = train["seed"]
    # Environment/serialization settings are not experimental treatments.
    for key in (
        "device",
        "num_workers",
        "seed",
        "keep_last_checkpoint",
        "lightweight_best_checkpoint",
    ):
        train.pop(key, None)
    split = yaml.safe_load((directory / "split.yaml").read_text(encoding="utf-8"))
    kd_raw = raw.get("distillation", {"method": "none"})
    method = kd_raw.get("method", "none")
    teacher = kd_raw.get("teacher", {})
    kd = {**asdict(DistillationConfig()), **kd_raw}
    for key in ("teacher", "probe_config", "probe_checkpoint", "method"):
        kd.pop(key, None)
    teacher_identity = identity.get("teacher")
    if teacher_identity:
        teacher_identity = {k: v for k, v in teacher_identity.items() if k != "fingerprint"}
    contract = {
        "data": data,
        "train": train,
        "augmentation": {**asdict(AugmentationConfig()), **raw.get("augmentation", {})},
        "split": split,
        "manifest_sha256": file_sha256(directory / "manifest.csv"),
        "dataset_content_sha256": identity.get("dataset_content_sha256"),
        "implementation_sha256": identity.get("implementation_sha256"),
    }
    contract = json.loads(json.dumps(contract))  # normalize tuple/list serialization
    history_path = directory / "history/training_summary.json"
    metrics = json.loads((directory / "metrics/summary.json").read_text())
    metrics["training_seconds"] = (
        json.loads(history_path.read_text()).get("training_seconds")
        if history_path.exists()
        else None
    )
    return {
        "directory": str(directory),
        "initialization": initialization,
        "student": raw["model"]["name"],
        "method": method,
        "teacher": teacher.get("family", "none") if method != "none" else "none",
        "fold": split["fold"],
        "seed": seed,
        "contract": contract,
        "kd_contract": {
            "options": kd,
            "teacher": teacher_identity,
            "head": teacher.get("head"),
            "probe_sha256": identity.get("probe_sha256"),
        },
        "initial_weight": identity.get("student_initial_sha256"),
        "metrics": metrics,
    }


def mismatch(left, right, *, same_kd=False, same_initialization=False):
    reasons = [k for k in left["contract"] if left["contract"][k] != right["contract"][k]]
    # Missing content proof is not a scientific match, even when both are absent.
    if (
        not left["contract"]["dataset_content_sha256"]
        or not right["contract"]["dataset_content_sha256"]
    ):
        reasons.append("missing_dataset_content_proof")
    if same_kd and left["kd_contract"] != right["kd_contract"]:
        reasons.append("teacher_or_kd_conditions")
    if same_initialization and left["initial_weight"] != right["initial_weight"]:
        reasons.append("student_initial_weight")
    return sorted(set(reasons))


def analyze_pretrained(plan_path, phase5_root=None, output=None, prediction_samples=0):
    import pandas as pd

    output = Path(output or Path(plan_path).parent.parent / "phase5_5_analysis")
    output.mkdir(parents=True, exist_ok=True)
    runs, issues = [], []
    for row in read_rows(plan_path):
        directory = Path(row["result_dir"])
        if not (directory / "completed.json").exists():
            issues.append({"directory": str(directory), "status": "incomplete"})
            continue
        runs.append(artifact(directory, "imagenet"))
    if phase5_root:
        for path in sorted(Path(phase5_root).rglob("metrics/summary.json")):
            directory = path.parent.parent
            if not (directory / "config.yaml").exists():
                continue
            raw = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            if raw.get("model", {}).get("pretrained"):
                continue
            if raw.get("model", {}).get("name") not in SOURCES:
                continue
            item = artifact(directory, "random")
            if item:
                candidates = [
                    r
                    for r in runs
                    if r["initialization"] == "imagenet"
                    and all(
                        r[k] == item[k] for k in ("student", "fold", "seed", "method", "teacher")
                    )
                ]
                if not candidates:
                    issues.append(
                        {"directory": str(directory), "status": "reference_not_requested"}
                    )
                    continue
                failures = [mismatch(r, item, same_kd=item["method"] != "none") for r in candidates]
                if all(failures):
                    issues.append(
                        {
                            "directory": str(directory),
                            "status": "condition_mismatch",
                            "detail": ",".join(sorted({key for keys in failures for key in keys})),
                        }
                    )
                    continue
                runs.append(item)
    columns = ["directory", "initialization", "student", "method", "teacher", "fold", "seed"]
    flat = [{**{k: r[k] for k in columns}, **r["metrics"]} for r in runs]
    write_csv(flat, output / "runs.csv", columns + METRICS)
    deltas = []

    def pair(treatment, candidates, effect, *, same_kd=False, same_init=False):
        candidates = [
            r
            for r in candidates
            if (r["student"], r["fold"], r["seed"])
            == (treatment["student"], treatment["fold"], treatment["seed"])
        ]
        matches, rejected = [], []
        for control in candidates:
            reasons = mismatch(treatment, control, same_kd=same_kd, same_initialization=same_init)
            if reasons:
                rejected.append(
                    {
                        "effect": effect,
                        "treatment": treatment["directory"],
                        "control": control["directory"],
                        "status": "condition_mismatch",
                        "detail": ",".join(reasons),
                    }
                )
            else:
                matches.append(control)
        issues.extend(rejected)
        if len(matches) != 1:
            issues.append(
                {
                    "effect": effect,
                    "treatment": treatment["directory"],
                    "status": "ambiguous_control" if matches else "no_matching_control",
                }
            )
            return None
        control = matches[0]
        delta = {
            **{k: treatment[k] for k in columns},
            "effect": effect,
            "control": control["directory"],
            "status": "matched",
            "interpretation": "KD additional effect",
        }
        if treatment["method"] == "gkd_cnn_source_only" and not same_kd:
            delta["status"] = "budget_different"
            delta["interpretation"] = "Includes two-stage schedule and representation freezing"
        for metric in METRICS:
            a, b = treatment["metrics"].get(metric), control["metrics"].get(metric)
            delta[metric + "_delta"] = a - b if a is not None and b is not None else None
        deltas.append(delta)
        return delta

    gt = [r for r in runs if r["method"] == "none"]
    for run in runs:
        if run["initialization"] == "imagenet":
            if run["method"] == "none":
                pair(run, [r for r in gt if r["initialization"] == "random"], "pretraining_gt")
            else:
                pair(
                    run,
                    [
                        r
                        for r in runs
                        if r["initialization"] == "random"
                        and r["method"] == run["method"]
                        and r["teacher"] == run["teacher"]
                    ],
                    "pretraining_kd",
                    same_kd=True,
                )
        if run["method"] != "none":
            pair(
                run,
                [r for r in gt if r["initialization"] == run["initialization"]],
                "kd_addition",
                same_init=True,
            )
    additions = [d for d in deltas if d["effect"] == "kd_addition"]
    interactions = []
    for run in [r for r in runs if r["initialization"] == "imagenet" and r["method"] != "none"]:
        key = tuple(run[k] for k in ("student", "method", "teacher", "fold", "seed"))
        find = lambda init, key=key: [
            d
            for d in additions
            if d["initialization"] == init
            and tuple(d[k] for k in ("student", "method", "teacher", "fold", "seed")) == key
        ]
        p, r = find("imagenet"), find("random")
        random_runs = [
            r
            for r in runs
            if r["initialization"] == "random"
            and tuple(r[k] for k in ("student", "method", "teacher", "fold", "seed")) == key
        ]
        if len(p) == len(r) == len(random_runs) == 1 and not mismatch(
            run, random_runs[0], same_kd=True
        ):
            interactions.append(
                {
                    **{k: run[k] for k in columns},
                    "effect": "kd_interaction",
                    "status": p[0]["status"],
                    "interpretation": p[0]["interpretation"],
                    **{
                        m + "_delta": p[0][m + "_delta"] - r[0][m + "_delta"]
                        if p[0][m + "_delta"] is not None and r[0][m + "_delta"] is not None
                        else None
                        for m in METRICS
                    },
                }
            )
    deltas.extend(interactions)
    write_csv(
        deltas,
        output / "paired_deltas.csv",
        columns
        + ["effect", "control", "status", "interpretation"]
        + [m + "_delta" for m in METRICS],
    )
    write_csv(
        issues,
        output / "comparison_status.csv",
        ["effect", "directory", "treatment", "control", "status", "detail"],
    )
    if flat:
        frame = pd.DataFrame(flat)
        for metric in METRICS:
            if metric not in frame:
                frame[metric] = float("nan")
        frame.groupby(["initialization", "student", "teacher", "method"])[METRICS].agg(
            ["mean", "std", "count"]
        ).to_csv(output / "summary_by_condition.csv")
        _plots(runs, output)
    if deltas:
        pd.DataFrame(deltas).groupby(["effect", "student", "teacher", "method", "status"])[
            [m + "_delta" for m in METRICS]
        ].agg(["mean", "std", "count"]).to_csv(output / "summary_paired_deltas.csv")
    if prediction_samples:
        from fm2edge.analysis.pretrained_predictions import prediction_comparisons

        prediction_comparisons(runs, output, prediction_samples)
    return output


def _plots(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    machine_rows = []
    for run in runs:
        path = Path(run["directory"]) / "metrics/per_machine.csv"
        if path.exists():
            for row in read_rows(path):
                machine_rows.append(
                    {
                        **row,
                        "condition": "/".join(
                            str(run[k]) for k in ("student", "initialization", "teacher", "method")
                        ),
                    }
                )
    if machine_rows:
        frame = pd.DataFrame(machine_rows)
        frame["IoU"] = pd.to_numeric(frame["IoU"])
        pivot = frame.pivot_table(index="machine_id", columns="condition", values="IoU")
        pivot.to_csv(output / "machine_heatmap.csv")
        figure, axis = plt.subplots(figsize=(max(9, len(pivot.columns) * 0.5), 5))
        plot = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=7)
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        figure.colorbar(plot, ax=axis, label="IoU")
        figure.tight_layout()
        figure.savefig(output / "machine_heatmap.png")
        plt.close(figure)
    for student in sorted({r["student"] for r in runs}):
        frame = pd.DataFrame(
            [
                {
                    "condition": "/".join(
                        str(r[k]) for k in ("initialization", "teacher", "method")
                    ),
                    "IoU": r["metrics"]["IoU"],
                }
                for r in runs
                if r["student"] == student
            ]
        )
        grouped = frame.groupby("condition").IoU.agg(["mean", "std"])
        figure, axis = plt.subplots(figsize=(max(8, len(grouped) * 0.6), 5))
        axis.bar(range(len(grouped)), grouped["mean"], yerr=grouped["std"].fillna(0))
        axis.set_xticks(range(len(grouped)), grouped.index, rotation=90, fontsize=8)
        axis.set_ylim(0, 1)
        axis.set_ylabel("IoU (mean +/- fold SD)")
        axis.set_title(student + " | descriptive; check paired_deltas.csv")
        figure.tight_layout()
        figure.savefig(output / (student + "_four_conditions.png"))
        plt.close(figure)
