"""Create a canonical manifest from Cityscapes or the company-style layout."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from fm2edge.data.adapters import scan_cityscapes, scan_machine_folders
from fm2edge.data.records import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cityscapes", "machine"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase-folder", default="delay")
    parser.add_argument("--max-machines", type=int, default=None)
    parser.add_argument("--max-samples-per-machine", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = (
        scan_cityscapes(args.root)
        if args.dataset == "cityscapes"
        else scan_machine_folders(args.root, phase_folder=args.phase_folder)
    )
    rng = random.Random(args.seed)
    machines = sorted({record.machine_id for record in records})
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
