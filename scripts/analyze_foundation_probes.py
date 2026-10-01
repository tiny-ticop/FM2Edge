"""Aggregate frozen-DINO probe results and optionally compare lightweight baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/foundation_probe")
    parser.add_argument(
        "--baseline-runs", default="results/machine_baseline_comparison/full_runs.csv"
    )
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('Install analysis dependencies: pip install -e ".[analysis]"') from exc

    root = Path(args.root)
    plan_path = root / "plan/experiment_plan.csv"
    plan = pd.read_csv(plan_path)
    rows: list[dict[str, object]] = []
    machine_rows: list[dict[str, object]] = []
    for record in plan.to_dict("records"):
        result = Path(str(record["result_dir"]))
        summary_path = result / "metrics/summary.json"
        if not summary_path.is_file():
            continue
        summary = _json(summary_path)
        training_path = result / "history/training_summary.json"
        training = _json(training_path) if training_path.is_file() else {}
        rows.append({**record, **summary, **{f"training_{k}": v for k, v in training.items()}})
        per_machine = result / "metrics/per_machine.csv"
        if per_machine.is_file():
            with per_machine.open("r", encoding="utf-8", newline="") as handle:
                for machine in csv.DictReader(handle):
                    machine_rows.append({**record, **machine})
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    if runs.empty:
        (analysis / "README.txt").write_text(
            "No completed foundation probe results were found.\n", encoding="utf-8"
        )
        print(f"no completed runs; analysis directory: {analysis}")
        return
    runs.to_csv(analysis / "runs.csv", index=False)
    metric_columns = [
        "IoU",
        "Dice",
        "Boundary_F1",
        "worst_machine_iou",
        "inference_ms_per_image",
        "gpu_peak_memory_mb",
        "training_total_epoch_time_seconds",
    ]
    existing = [column for column in metric_columns if column in runs]
    grouped = (
        runs.groupby(["teacher", "family", "variant", "initialization", "head"])[existing]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    grouped.to_csv(analysis / "summary_by_condition.csv", index=False)

    linear = runs[(runs["family"] == "dinov2") & (runs["head"] == "linear")]
    pretrained = linear[linear["initialization"] == "pretrained"]
    random = linear[linear["initialization"] == "random_init"]
    ablation = pretrained.merge(random, on="fold", suffixes=("_pretrained", "_random"))
    if not ablation.empty:
        ablation["IoU_delta_pretrained_minus_random"] = (
            ablation["IoU_pretrained"] - ablation["IoU_random"]
        )
        ablation.to_csv(analysis / "pretrained_vs_random.csv", index=False)

    baseline_path = Path(args.baseline_runs)
    if baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
        comparisons = runs.merge(
            baseline[["model", "fold", "IoU"]], on="fold", suffixes=("_dino", "_baseline")
        )
        comparisons["IoU_delta_dino_minus_baseline"] = (
            comparisons["IoU_dino"] - comparisons["IoU_baseline"]
        )
        comparisons.to_csv(analysis / "paired_baseline_deltas.csv", index=False)
    else:
        (analysis / "baseline_comparison_status.txt").write_text(
            f"Baseline results not found at {baseline_path}. Re-run with --baseline-runs PATH.\n",
            encoding="utf-8",
        )

    machines = pd.DataFrame(machine_rows)
    if not machines.empty:
        machines["IoU"] = machines["IoU"].astype(float)
        machines["condition"] = machines["teacher"] + " / " + machines["head"]
        heatmap = machines.pivot_table(
            index="condition", columns="machine_id", values="IoU", aggfunc="mean"
        )
        heatmap.to_csv(analysis / "machine_iou_heatmap.csv")
        figure, axis = plt.subplots(
            figsize=(max(7, heatmap.shape[1] * 1.2), max(3, heatmap.shape[0] * 0.55))
        )
        image = axis.imshow(heatmap.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(heatmap.shape[1]), heatmap.columns, rotation=30, ha="right")
        axis.set_yticks(np.arange(heatmap.shape[0]), heatmap.index)
        axis.set_title("Unseen-machine IoU")
        figure.colorbar(image, ax=axis, label="IoU")
        figure.tight_layout()
        figure.savefig(analysis / "machine_iou_heatmap.png", dpi=180)
        plt.close(figure)
    print(f"foundation analysis: {analysis}")


if __name__ == "__main__":
    main()
