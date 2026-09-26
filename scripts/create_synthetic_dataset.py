"""Create a tiny deterministic dataset for local and Colab smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--machines", type=int, default=5)
    parser.add_argument("--samples-per-machine", type=int, default=4)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    root = Path(args.output)
    rng = np.random.default_rng(7)
    for machine_index in range(args.machines):
        machine = f"machine_{machine_index:03d}"
        for sample_index in range(args.samples_per_machine):
            delay = str(90 + sample_index % 2)
            image_dir = root / "images" / machine / "delay" / delay
            mask_dir = root / "masks" / machine / "delay" / delay
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            yy, xx = np.mgrid[: args.size, : args.size]
            center_x = args.size // 2 + machine_index - args.machines // 2
            center_y = args.size // 2 + sample_index - args.samples_per_machine // 2
            radius = max(4, args.size // 6)
            mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2 < radius**2).astype(np.uint8)
            image = rng.normal(35 + machine_index * 8, 5, (args.size, args.size, 3))
            image[..., 1] += mask * 170
            image = np.clip(image, 0, 255).astype(np.uint8)
            filename = f"sample_{sample_index:03d}.png"
            Image.fromarray(image).save(image_dir / filename)
            Image.fromarray(mask).save(mask_dir / filename)
    print(f"synthetic dataset written to {root}")


if __name__ == "__main__":
    main()
