"""Extract and visualize model feature distributions by machine domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fm2edge.config import load_config
from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.splits import load_split
from fm2edge.data.transforms import SegmentationTransform
from fm2edge.engine.utils import resolve_device
from fm2edge.models.registry import build_student

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _masked_pool(
    feature: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor
) -> torch.Tensor:
    if not mask.any():
        return torch.full_like(fallback, torch.nan)
    return feature[:, mask].mean(dim=1)


@torch.inference_mode()
def _extract_role(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    role: str,
    ignore_index: int,
) -> tuple[list[dict[str, str]], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    metadata: list[dict[str, str]] = []
    global_vectors: list[np.ndarray] = []
    foreground_vectors: list[np.ndarray] = []
    background_vectors: list[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["mask"].to(device)
        features = model(images).features["kd"]
        resized = F.interpolate(
            targets[:, None].float(), size=features.shape[-2:], mode="nearest"
        )[:, 0].long()
        for index in range(features.shape[0]):
            feature = features[index]
            target = resized[index]
            global_vector = feature.mean(dim=(1, 2))
            valid = target != ignore_index
            foreground = (target > 0) & valid
            background = (target == 0) & valid

            metadata.append(
                {
                    "sample_id": str(batch["sample_id"][index]),
                    "machine_id": str(batch["machine_id"][index]),
                    "delay": str(batch["delay"][index]),
                    "role": role,
                }
            )
            global_vectors.append(global_vector.cpu().numpy())
            foreground_vectors.append(
                _masked_pool(feature, foreground, global_vector).cpu().numpy()
            )
            background_vectors.append(
                _masked_pool(feature, background, global_vector).cpu().numpy()
            )
    return metadata, global_vectors, foreground_vectors, background_vectors


def _plot_pca(metadata: list[dict[str, str]], vectors: np.ndarray, path: Path, title: str) -> float:
    scaled = StandardScaler().fit_transform(vectors)
    coordinates = PCA(n_components=2, random_state=42).fit_transform(scaled)
    machines = np.array([row["machine_id"] for row in metadata])
    roles = np.array([row["role"] for row in metadata])
    figure, axis = plt.subplots(figsize=(8, 6))
    role_markers = {"train": "o", "validation": "^", "test": "X"}
    for machine in sorted(set(machines)):
        for role, marker in role_markers.items():
            selected = (machines == machine) & (roles == role)
            if selected.any():
                axis.scatter(
                    coordinates[selected, 0],
                    coordinates[selected, 1],
                    marker=marker,
                    alpha=0.65,
                    label=f"{machine} ({role})",
                )
    axis.set_title(title)
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    if len(set(machines)) > 1 and all(np.sum(machines == value) > 1 for value in set(machines)):
        return float(silhouette_score(scaled, machines))
    return float("nan")


def _process_run(row: dict[str, str], output: Path, allow_cpu: bool) -> dict[str, object]:
    config = load_config(ROOT / row["config_path"])
    split = load_split(ROOT / row["split_path"])
    result_dir = ROOT / row["result_dir"]
    checkpoint_path = result_dir / "checkpoints/best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = resolve_device("cpu" if allow_cpu and not torch.cuda.is_available() else config.train.device)
    model = build_student(config.model.name, config.data.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    transform = SegmentationTransform(
        config.data.image_size,
        config.data.mean,
        config.data.std,
        config.data.ignore_index,
        config.data.mask_value_map,
    )
    role_machines = {
        "train": split.train_machines,
        "validation": split.val_machines,
        "test": split.test_machines,
    }
    metadata: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    foreground: list[np.ndarray] = []
    background: list[np.ndarray] = []
    for role, machines in role_machines.items():
        dataset = ManifestSegmentationDataset(
            config.data.manifest,
            config.data.root,
            machines,
            transform,
            sample_ids=split.train_sample_ids if role == "train" and split.train_sample_ids else None,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_workers,
        )
        part = _extract_role(model, loader, device, role, config.data.ignore_index)
        metadata.extend(part[0])
        vectors.extend(part[1])
        foreground.extend(part[2])
        background.extend(part[3])
    vector_array = np.stack(vectors)
    run_output = output / row["run_id"]
    run_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_output / "embeddings.npz",
        global_features=vector_array,
        foreground_features=np.stack(foreground),
        background_features=np.stack(background),
    )
    with (run_output / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    score = _plot_pca(
        metadata,
        vector_array,
        run_output / "feature_pca.png",
        f"{row['model']} fold {row['fold']} ({row['suite']}: {row['augmentation']})",
    )
    roles = np.array([item["role"] for item in metadata])
    standardized = StandardScaler().fit_transform(vector_array)
    train_centroid = standardized[roles == "train"].mean(axis=0)
    test_centroid = standardized[roles == "test"].mean(axis=0)
    metrics = json.loads((result_dir / "metrics/summary.json").read_text(encoding="utf-8"))
    return {
        "run_id": row["run_id"],
        "model": row["model"],
        "fold": row["fold"],
        "suite": row["suite"],
        "augmentation": row["augmentation"],
        "machine_silhouette": score,
        "train_test_centroid_distance": float(np.linalg.norm(train_centroid - test_centroid)),
        "test_IoU": metrics["IoU"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", default="results/generalization_analysis/plan/experiment_plan.csv"
    )
    parser.add_argument("--include-combined", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable; use --allow-cpu only for a small smoke dataset")
    with (ROOT / args.plan).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["suite"] == "reference"]
    if args.include_combined:
        selected.extend(
            row
            for row in rows
            if row["suite"] == "augmentation" and row["augmentation"] == "combined"
        )
    selected = [
        row
        for row in selected
        if (ROOT / row["result_dir"] / "metrics/summary.json").is_file()
    ]
    plan_path = ROOT / args.plan
    output = plan_path.parent.parent / "analysis/features"
    summaries = [_process_run(row, output, args.allow_cpu) for row in selected]
    if not summaries:
        raise ValueError("No completed reference/combined runs were found")
    with (output / "feature_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    summary_frame = np.array(
        [
            [float(row["train_test_centroid_distance"]), float(row["test_IoU"])]
            for row in summaries
        ]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    for model in sorted({str(row["model"]) for row in summaries}):
        selected = [row for row in summaries if row["model"] == model]
        axis.scatter(
            [row["train_test_centroid_distance"] for row in selected],
            [row["test_IoU"] for row in selected],
            alpha=0.7,
            label=model,
        )
    axis.set_xlabel("Train-to-test feature centroid distance")
    axis.set_ylabel("Unknown-machine IoU")
    axis.set_title("Learned feature shift versus segmentation performance")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "feature_shift_vs_iou.png", dpi=180)
    plt.close(figure)
    if len(summary_frame) > 1:
        correlation = np.corrcoef(summary_frame[:, 0], summary_frame[:, 1])[0, 1]
        (output / "feature_shift_correlation.json").write_text(
            json.dumps({"pearson_distance_vs_iou": float(correlation)}, indent=2),
            encoding="utf-8",
        )
    print(f"feature analysis: {output}")


if __name__ == "__main__":
    main()
