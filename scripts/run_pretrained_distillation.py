"""Phase5.5 wrapper; reuse the unchanged KD trainer and probe preparation."""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

from fm2edge.analysis.distillation_study import write_csv
from fm2edge.analysis.pretrained_study import build_pretrained_plan, read_rows
from fm2edge.distillation_config import load_distillation_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analyses/pretrained_distillation.yaml")
    parser.add_argument(
        "--phase5-plan", help="Copy exact Phase5 per-run settings (random initialization)"
    )
    parser.add_argument("--mode", choices=["plan", "smoke", "full"], required=True)
    parser.add_argument("--students", nargs="+")
    parser.add_argument("--teachers", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    plan = build_pretrained_plan(
        args.config,
        smoke=args.mode == "smoke",
        phase5_plan=args.phase5_plan,
        students=args.students,
    )
    print(f"Plan: {plan}")
    if args.mode == "plan":
        print(json.loads((plan.parent / "summary.json").read_text()))
        return
    prepare_probe = runpy.run_path("scripts/run_distillation_study.py")["prepare_probe"]
    status_path = plan.parent.parent / "run_status.csv"
    statuses = {r["run_id"]: r for r in read_rows(status_path)} if status_path.exists() else {}
    selected = 0
    for row in read_rows(plan):
        if any(
            values and row[key] not in values
            for key, values in (("student", args.students), ("method", args.methods))
        ):
            continue
        if args.teachers and row["method"] != "none" and row["teacher"] not in args.teachers:
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
        missing = []
        if kd.method != "none":
            if not (Path(kd.teacher.repository_path or "") / "hubconf.py").exists():
                missing.append("official Teacher repository")
            if not Path(kd.teacher.weights_path or "").is_file():
                missing.append("Teacher weights")
        if missing:
            status.update(status="skipped_prerequisite", detail=", ".join(missing))
            print(f"Skip {row['run_id']}: {status['detail']}")
        else:
            try:
                prepare_probe(config, kd)
                command = [
                    sys.executable,
                    "scripts/train_distillation.py",
                    "--config",
                    row["config_path"],
                ]
                if args.train_only:
                    command.append("--train-only")
                if args.rerun:
                    command.append("--no-resume")
                subprocess.run(command, check=True)
                status["status"] = "trained" if args.train_only else "completed"
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                status.update(status="failed", detail=str(exc))
                statuses[row["run_id"]] = status
                write_csv(list(statuses.values()), status_path)
                raise
        statuses[row["run_id"]] = status
        write_csv(list(statuses.values()), status_path)
    if not selected:
        raise ValueError("No runs matched filters")


if __name__ == "__main__":
    main()
