"""Aggregate completed study runs and generate decision-oriented figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from fm2edge.data.records import read_manifest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "IoU",
    "Dice",
    "Boundary_F1",
    "Precision",
    "Recall",
    "foreground_ratio",
    "target_foreground_ratio",
    "all_background_rate",
    "all_foreground_rate",
    "prediction_confidence",
    "prediction_entropy",
    "inference_ms_per_image",
    "inference_fps",
    "gpu_peak_memory_mb",
)


def _json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _factor_value(row: dict[str, str]) -> str:
    if row["suite"] == "reference":
        return "none"
    if row["suite"] == "machine_diversity":
        return row["train_machine_count"]
    if row["suite"] == "augmentation":
        return row["augmentation"]
    allocation = json.loads(row["allocation"])
    values = {str(value) for value in allocation.values()}
    return next(iter(values)) if len(values) == 1 else "mixed"


def collect_runs(plan_path: Path) -> pd.DataFrame:
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        plan = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for item in plan:
        result_dir = ROOT / item["result_dir"]
        metrics_path = result_dir / "metrics/summary.json"
        if not metrics_path.is_file():
            continue
        metrics = _json(metrics_path)
        training_path = result_dir / "history/training_summary.json"
        training = _json(training_path) if training_path.is_file() else {}
        row: dict[str, object] = {
            **item,
            "factor_value": _factor_value(item),
            "target_machine": ",".join(json.loads(item["test_machines"])),
            "best_epoch": training.get("best_epoch", np.nan),
            "best_val_mIoU": training.get("best_val_mIoU", np.nan),
            "completed_epochs": training.get("completed_epochs", np.nan),
            "training_time_seconds": training.get("total_epoch_time_seconds", np.nan),
        }
        row.update(metrics)
        per_image_path = result_dir / "metrics/per_image.csv"
        if per_image_path.is_file():
            per_image = pd.read_csv(per_image_path)
            row["all_background_rate"] = float((per_image["foreground_ratio"] <= 1e-6).mean())
            row["all_foreground_rate"] = float((per_image["foreground_ratio"] >= 1 - 1e-6).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_bootstrap(values: pd.DataFrame, metric: str, repeats: int = 2000) -> tuple[float, float]:
    by_target = values.groupby("target_machine")[metric].mean()
    array = by_target.to_numpy(dtype=float)
    if len(array) < 2:
        return float(array.mean()), float(array.mean())
    rng = np.random.default_rng(42)
    estimates = [rng.choice(array, size=len(array), replace=True).mean() for _ in range(repeats)]
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model", "suite", "factor_value"]
    for key, group in runs.groupby(keys, sort=True):
        row: dict[str, object] = dict(zip(keys, key, strict=True))
        row["runs"] = len(group)
        row["target_machines"] = group["target_machine"].nunique()
        for metric in METRICS:
            if metric not in group:
                continue
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
            low, high = _cluster_bootstrap(group, metric)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(runs: pd.DataFrame) -> pd.DataFrame:
    reference = runs[runs["suite"] == "reference"]
    output = []
    for (model, suite, factor), group in runs[runs["suite"] != "reference"].groupby(
        ["model", "suite", "factor_value"]
    ):
        reduced = group.groupby(["fold", "seed", "target_machine"], as_index=False)[
            list(METRICS)
        ].mean()
        baseline = reference[reference["model"] == model][
            ["fold", "seed", "target_machine", *METRICS]
        ]
        merged = reduced.merge(
            baseline,
            on=["fold", "seed", "target_machine"],
            suffixes=("", "_reference"),
        )
        if merged.empty:
            continue
        row: dict[str, object] = {
            "model": model,
            "suite": suite,
            "factor_value": factor,
            "paired_targets": len(merged),
        }
        for metric in METRICS:
            delta = merged[metric] - merged[f"{metric}_reference"]
            row[f"{metric}_delta_mean"] = delta.mean()
            rng = np.random.default_rng(42)
            estimates = [
                rng.choice(delta.to_numpy(), size=len(delta), replace=True).mean()
                for _ in range(2000)
            ]
            row[f"{metric}_delta_ci95_low"] = np.quantile(estimates, 0.025)
            row[f"{metric}_delta_ci95_high"] = np.quantile(estimates, 0.975)
        output.append(row)
    return pd.DataFrame(output)


def _save_factor_plot(
    aggregates: pd.DataFrame,
    suite: str,
    output: Path,
    title: str,
    categorical: bool = False,
    reference_label: str | None = None,
) -> None:
    frame = aggregates[aggregates["suite"] == suite].copy()
    if reference_label is not None:
        reference = aggregates[aggregates["suite"] == "reference"].copy()
        reference["suite"] = suite
        reference["factor_value"] = reference_label
        frame = pd.concat([frame, reference], ignore_index=True)
    if frame.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    for model, values in frame.groupby("model"):
        if categorical:
            orders = {
                "images_per_machine": ["10", "25", "50", "all"],
                "augmentation": [
                    "none",
                    "brightness_contrast",
                    "color",
                    "noise",
                    "blur",
                    "combined",
                ],
            }
            order = {name: index for index, name in enumerate(orders.get(suite, []))}
            values["category_order"] = (
                values["factor_value"].astype(str).map(order).fillna(len(order))
            )
            values = values.sort_values("category_order")
            x = np.arange(len(values))
            labels = values["factor_value"].tolist()
        else:
            values["numeric"] = pd.to_numeric(values["factor_value"], errors="coerce")
            values = values.sort_values("numeric")
            x = values["numeric"]
            labels = None
        mean = values["IoU_mean"].to_numpy()
        lower = mean - values["IoU_ci95_low"].to_numpy()
        upper = values["IoU_ci95_high"].to_numpy() - mean
        axis.errorbar(x, mean, yerr=[lower, upper], marker="o", capsize=4, label=model)
        if labels is not None:
            axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title(title)
    axis.set_ylabel("Unknown-machine IoU")
    axis.set_xlabel("Condition")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _save_heatmaps(runs: pd.DataFrame, output: Path) -> None:
    frame = runs[runs["suite"] == "machine_diversity"].copy()
    for model, values in frame.groupby("model"):
        values["train_subset"] = values["train_machines"].map(
            lambda text: "+".join(json.loads(text))
        )
        matrix = values.pivot_table(
            index="train_subset", columns="target_machine", values="IoU", aggfunc="mean"
        )
        if matrix.empty:
            continue
        figure, axis = plt.subplots(
            figsize=(max(7, len(matrix.columns) * 1.4), max(5, len(matrix) * 0.35))
        )
        image = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
        axis.set_xlabel("Unknown test machine")
        axis.set_ylabel("Training-machine subset")
        axis.set_title(f"{model}: transfer IoU")
        for row_index in range(len(matrix.index)):
            for column_index in range(len(matrix.columns)):
                value = matrix.iloc[row_index, column_index]
                if not np.isnan(value):
                    color = "white" if value < 0.45 else "black"
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=7,
                    )
        figure.colorbar(image, ax=axis, label="IoU")
        figure.tight_layout()
        path = output / f"machine_heatmap_{model}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
        plt.close(figure)
        matrix.to_csv(output / f"machine_heatmap_{model}.csv")


def _dataset_statistics(plan_path: Path, output: Path) -> pd.DataFrame:
    with plan_path.open("r", encoding="utf-8", newline="") as handle:
        first = next(csv.DictReader(handle))
    with (ROOT / first["config_path"]).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = ROOT / config["data"]["root"]
    mapping = {int(key): int(value) for key, value in config["data"]["mask_value_map"].items()}
    rows = []
    for record in read_manifest(ROOT / config["data"]["manifest"]):
        with Image.open(root / record.image_path) as image_handle:
            rgb = np.asarray(image_handle.convert("RGB"), dtype=np.float32) / 255.0
        with Image.open(root / record.mask_path) as mask_handle:
            mask = np.asarray(mask_handle)
        gray = rgb.mean(axis=2)
        saturation = rgb.max(axis=2) - rgb.min(axis=2)
        sharpness = (np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
        mapped = np.full(mask.shape[:2], 255, dtype=np.int64)
        source = mask[..., 0] if mask.ndim == 3 else mask
        for raw_value, target in mapping.items():
            mapped[source == raw_value] = target
        valid = mapped != 255
        rows.append(
            {
                "sample_id": record.sample_id,
                "machine_id": record.machine_id,
                "delay": record.delay,
                "red_mean": rgb[..., 0].mean(),
                "green_mean": rgb[..., 1].mean(),
                "blue_mean": rgb[..., 2].mean(),
                "luminance_mean": gray.mean(),
                "contrast": gray.std(),
                "saturation_mean": saturation.mean(),
                "sharpness": sharpness,
                "foreground_ratio": (mapped[valid] > 0).mean() if valid.any() else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_image_domain_statistics.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number]).columns
    grouped = frame.groupby("machine_id")[numeric].agg(["mean", "std"])
    grouped.to_csv(output / "per_machine_domain_statistics.csv")
    features = [
        "luminance_mean",
        "contrast",
        "saturation_mean",
        "sharpness",
        "foreground_ratio",
    ]
    normalized = frame.groupby("machine_id")[features].mean()
    normalized = (normalized - normalized.mean()) / normalized.std(ddof=0).replace(0, 1)
    figure, axis = plt.subplots(figsize=(9, max(4, len(normalized) * 0.7)))
    image = axis.imshow(normalized.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    axis.set_xticks(np.arange(len(features)), features, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(normalized.index)), normalized.index)
    axis.set_title("Per-machine input-domain profile (z-score)")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "machine_domain_profile.png", dpi=180)
    plt.close(figure)
    return normalized


def _domain_shift_performance(runs: pd.DataFrame, profiles: pd.DataFrame, output: Path) -> None:
    rows = []
    for _, run in runs.iterrows():
        train_machines = json.loads(run["train_machines"])
        test_machines = json.loads(run["test_machines"])
        if not set(train_machines + test_machines).issubset(profiles.index):
            continue
        train_centroid = profiles.loc[train_machines].mean(axis=0).to_numpy()
        test_centroid = profiles.loc[test_machines].mean(axis=0).to_numpy()
        rows.append(
            {
                "run_id": run["run_id"],
                "model": run["model"],
                "suite": run["suite"],
                "factor_value": run["factor_value"],
                "target_machine": run["target_machine"],
                "input_domain_distance": np.linalg.norm(train_centroid - test_centroid),
                "IoU": run["IoU"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame.to_csv(output / "domain_shift_vs_performance.csv", index=False)
    correlations = (
        frame.groupby("model")[["input_domain_distance", "IoU"]]
        .corr(method="spearman")
        .iloc[0::2, -1]
        .reset_index()
        .rename(columns={"IoU": "spearman_distance_vs_iou"})
    )
    correlations.to_csv(output / "domain_shift_correlations.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    for model, values in frame.groupby("model"):
        axis.scatter(
            values["input_domain_distance"], values["IoU"], alpha=0.65, label=model
        )
    axis.set_xlabel("Input-domain distance: train centroid to test machine")
    axis.set_ylabel("Unknown-machine IoU")
    axis.set_title("Input-domain shift versus segmentation performance")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "domain_shift_vs_iou.png", dpi=180)
    plt.close(figure)


def _write_report(runs: pd.DataFrame, output: Path) -> None:
    completed = len(runs)
    lines = [
        "# Machine generalization factor analysis",
        "",
        f"Completed runs: **{completed}**",
        "",
        "## Generated tables",
        "",
        "- `tables/all_runs.csv`",
        "- `tables/aggregate_results.csv`",
        "- `tables/paired_effects.csv`",
        "- `domain_statistics/per_machine_domain_statistics.csv`",
        "",
        "## Generated figures",
        "",
        "- `figures/machine_count_iou.png`",
        "- `figures/images_per_machine_iou.png`",
        "- `figures/augmentation_iou.png`",
        "- `figures/machine_heatmap_<model>.png`",
        "- `domain_statistics/machine_domain_profile.png`",
        "",
        (
            "Confidence intervals use a target-machine cluster bootstrap. With five machines, "
            "interpret interval widths and rankings as exploratory rather than definitive."
        ),
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="results/generalization_analysis/plan/experiment_plan.csv"
    )
    args = parser.parse_args()
    plan_path = ROOT / args.plan
    output = plan_path.parent.parent / "analysis"
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(plan_path)
    if runs.empty:
        raise ValueError("No completed study runs were found")
    aggregates = aggregate_runs(runs)
    effects = paired_effects(runs)
    runs.to_csv(tables / "all_runs.csv", index=False)
    aggregates.to_csv(tables / "aggregate_results.csv", index=False)
    effects.to_csv(tables / "paired_effects.csv", index=False)
    _save_factor_plot(
        aggregates,
        "machine_diversity",
        figures / "machine_count_iou.png",
        "Effect of training-machine diversity at fixed image budget",
    )
    _save_factor_plot(
        aggregates,
        "images_per_machine",
        figures / "images_per_machine_iou.png",
        "Effect of images per training machine",
        categorical=True,
        reference_label="all",
    )
    _save_factor_plot(
        aggregates,
        "augmentation",
        figures / "augmentation_iou.png",
        "Effect of image augmentation",
        categorical=True,
        reference_label="none",
    )
    _save_heatmaps(runs, figures)
    profiles = _dataset_statistics(plan_path, output / "domain_statistics")
    _domain_shift_performance(runs, profiles, output / "domain_statistics")
    _write_report(runs, output)
    print(f"analysis report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
