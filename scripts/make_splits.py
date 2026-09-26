"""Generate reproducible machine-disjoint YAML folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from fm2edge.data.records import read_manifest
from fm2edge.data.splits import make_machine_folds, save_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    folds = make_machine_folds(records, args.folds, args.seed)
    output = Path(args.output_dir)
    for split in folds:
        save_split(split, output / f"fold_{split.fold:02d}.yaml")
    print(f"wrote {len(folds)} machine-disjoint folds to {output}")


if __name__ == "__main__":
    main()
