"""Create a canonical manifest from Cityscapes or the company-style layout."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from fm2edge.data.adapters import scan_cityscapes, scan_machine_folders
from fm2edge.data.records import write_manifest


def _validate_pairs(
    root: Path,
    records: list,
    allowed_mask_values: set[int] | None,
    required_mask_values: set[int] | None,
) -> None:
    """Fail early on malformed masks before a long training run starts."""
    observed_values: set[int] = set()
    for record in records:
        image_path = root / record.image_path
        mask_path = root / record.mask_path
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Image/mask size mismatch: {image_path} is {image.size}, "
                    f"but {mask_path} is {mask.size}"
                )
            mask_array = np.asarray(mask)
        if mask_array.ndim != 2:
            raise ValueError(
                f"Mask must be a single-channel PNG, got shape {mask_array.shape}: {mask_path}"
            )
        values = {int(value) for value in np.unique(mask_array)}
        observed_values.update(values)
        if allowed_mask_values is not None:
            unexpected = values - allowed_mask_values
            if unexpected:
                raise ValueError(
                    f"Unexpected mask values {sorted(unexpected)} in {mask_path}; "
                    f"allowed values are {sorted(allowed_mask_values)}"
                )
    if required_mask_values is not None:
        missing = required_mask_values - observed_values
        if missing:
            raise ValueError(
                f"Required mask values {sorted(missing)} were not found; "
                f"observed values are {sorted(observed_values)}"
            )
    print(f"validated {len(records)} image/mask pairs; mask values={sorted(observed_values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cityscapes", "machine"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-folder", default="delay")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--masks-dir", default="masks")
    parser.add_argument(
        "--mask-extension",
        default=None,
        help="Pair each image stem with this mask extension, for example .png",
    )
    parser.add_argument(
        "--allowed-mask-values",
        nargs="+",
        type=int,
        default=None,
        help="Validate raw mask pixels before writing the manifest, for example 0 1 255",
    )
    parser.add_argument(
        "--required-mask-values",
        nargs="+",
        type=int,
        default=None,
        help="Require these values to occur somewhere in the dataset, for example 0 1",
    )
    parser.add_argument("--max-machines", type=int, default=None)
    parser.add_argument(
        "--expected-machines",
        type=int,
        default=None,
        help="Fail unless this many distinct machine folders are present",
    )
    parser.add_argument("--max-samples-per-machine", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = (
        scan_cityscapes(args.root)
        if args.dataset == "cityscapes"
        else scan_machine_folders(
            args.root,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            phase_folder=args.phase_folder,
            mask_extension=args.mask_extension,
        )
    )
    _validate_pairs(
        Path(args.root),
        records,
        set(args.allowed_mask_values) if args.allowed_mask_values is not None else None,
        set(args.required_mask_values) if args.required_mask_values is not None else None,
    )
    rng = random.Random(args.seed)
    machines = sorted({record.machine_id for record in records})
    if args.expected_machines is not None and len(machines) != args.expected_machines:
        raise ValueError(
            f"Expected {args.expected_machines} machines, found {len(machines)}: {machines}"
        )
    rng.shuffle(machines)
    if args.max_machines is not None:
        machines = machines[: args.max_machines]
    selected_machines = set(machines)
    grouped = defaultdict(list)
    for record in records:
        if record.machine_id in selected_machines:
            grouped[record.machine_id].append(record)
    records = []
    for machine in machines:
        machine_records = sorted(grouped[machine], key=lambda item: item.sample_id)
        rng.shuffle(machine_records)
        if args.max_samples_per_machine is not None:
            machine_records = machine_records[: args.max_samples_per_machine]
        records.extend(machine_records)
    write_manifest(records, args.output)
    print(f"wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
