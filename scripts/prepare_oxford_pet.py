"""Convert Oxford-IIIT Pet into a breed-domain canonical manifest."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from fm2edge.data.public.oxford_pet import scan_oxford_pet
from fm2edge.data.records import SampleRecord, write_manifest


def _subsample(
    records: list[SampleRecord], max_breeds: int | None, max_samples: int | None, seed: int
) -> list[SampleRecord]:
    rng = random.Random(seed)
    breeds = sorted({record.machine_id for record in records})
    rng.shuffle(breeds)
    if max_breeds is not None:
        breeds = breeds[:max_breeds]
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.machine_id].append(record)
    selected: list[SampleRecord] = []
    for breed in breeds:
        breed_records = sorted(grouped[breed], key=lambda record: record.sample_id)
        rng.shuffle(breed_records)
        selected.extend(breed_records[:max_samples] if max_samples is not None else breed_records)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/oxford_pet")
    parser.add_argument("--output", default="data/oxford_pet/manifest.csv")
    parser.add_argument("--max-breeds", type=int, default=None)
    parser.add_argument("--max-samples-per-breed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = _subsample(
        scan_oxford_pet(args.root),
        args.max_breeds,
        args.max_samples_per_breed,
        args.seed,
    )
    write_manifest(records, args.output)
    print(
        f"wrote {len(records)} samples from {len({record.machine_id for record in records})} "
        f"breed domains to {args.output}"
    )


if __name__ == "__main__":
    main()
