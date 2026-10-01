"""Visualize cached DINO feature domains and foreground/background separation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from fm2edge.config import load_config
from fm2edge.data.foundation_cache import cache_file, cache_identity, validate_cache_metadata
from fm2edge.data.records import read_manifest
from fm2edge.data.splits import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Any generated config for the teacher")
    parser.add_argument("--output", default="results/foundation_probe/analysis/features")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError('Install analysis dependencies: pip install -e ".[analysis]"') from exc
    config = load_config(args.config)
    if not config.model.cache_dir:
        raise ValueError("model.cache_dir is required")
    identity = cache_identity(config)
    validate_cache_metadata(config.model.cache_dir, identity)
    records = read_manifest(config.data.manifest)
    vectors: list[np.ndarray] = []
    foreground: list[np.ndarray] = []
    background: list[np.ndarray] = []
    machines: list[str] = []
    sample_ids: list[str] = []
    for record in records:
        path = cache_file(
            config.model.cache_dir, str(identity["fingerprint"]), record.sample_id
        )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        feature = payload["features"][-1].float()
        with Image.open(Path(config.data.root) / record.mask_path) as handle:
            mask = np.asarray(handle.resize(config.data.image_size[::-1], Image.Resampling.NEAREST))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_tensor = torch.from_numpy(mask.copy()).long()
        mapped = torch.full_like(mask_tensor, config.data.ignore_index)
        for source, target in config.data.mask_value_map.items():
            mapped[mask_tensor == source] = target
        resized = F.interpolate(
            mapped[None, None].float(), size=feature.shape[-2:], mode="nearest"
        )[0, 0].long()
        global_vector = feature.mean(dim=(1, 2))
        fg = resized == 1
        bg = resized == 0
        vectors.append(global_vector.numpy())
        foreground.append((feature[:, fg].mean(1) if fg.any() else global_vector).numpy())
        background.append((feature[:, bg].mean(1) if bg.any() else global_vector).numpy())
        machines.append(record.machine_id)
        sample_ids.append(record.sample_id)
    array = np.stack(vectors)
    projected = PCA(n_components=2).fit_transform(array)
    output = Path(args.output) / f"{config.model.family}_{config.model.variant}_{config.model.initialization}"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "feature_vectors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "machine_id", "pca_1", "pca_2"])
        writer.writerows(
            (sample_id, machine, float(point[0]), float(point[1]))
            for sample_id, machine, point in zip(sample_ids, machines, projected, strict=True)
        )
    figure, axis = plt.subplots(figsize=(8, 6))
    for machine in sorted(set(machines)):
        indices = [index for index, value in enumerate(machines) if value == machine]
        axis.scatter(projected[indices, 0], projected[indices, 1], s=16, label=machine, alpha=0.75)
    axis.set_title("Frozen DINO feature distribution by machine")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "feature_pca.png", dpi=180)
    plt.close(figure)
    fg_array = np.stack(foreground)
    bg_array = np.stack(background)
    cosine_distance = 1.0 - np.sum(fg_array * bg_array, axis=1) / (
        np.linalg.norm(fg_array, axis=1) * np.linalg.norm(bg_array, axis=1) + 1e-8
    )
    split_rows = []
    for fold in range(1, 6):
        split_path = Path(config.data.split).parent / f"fold_{fold:02d}.yaml"
        if not split_path.is_file():
            continue
        split = load_split(split_path)
        train_indices = [i for i, machine in enumerate(machines) if machine in split.train_machines]
        test_indices = [i for i, machine in enumerate(machines) if machine in split.test_machines]
        centroid = array[train_indices].mean(0)
        distance = np.linalg.norm(array[test_indices] - centroid, axis=1).mean()
        split_rows.append({"fold": fold, "train_to_test_centroid_distance": float(distance)})
    unique_machines = set(machines)
    silhouette = (
        float(silhouette_score(array, machines))
        if 1 < len(unique_machines) < len(machines)
        else None
    )
    summary = {
        "num_images": len(records),
        "num_machines": len(unique_machines),
        "machine_silhouette": silhouette,
        "foreground_background_cosine_distance_mean": float(cosine_distance.mean()),
        "fold_distances": split_rows,
    }
    (output / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"feature analysis: {output}")


if __name__ == "__main__":
    main()
