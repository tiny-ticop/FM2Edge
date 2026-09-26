"""Portable sample metadata used by every dataset adapter."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: str
    mask_path: str
    machine_id: str
    delay: str = ""
    dataset_name: str = ""
    height: int = 0
    width: int = 0
    split_source: str = ""


FIELDS = tuple(SampleRecord.__dataclass_fields__)


def write_manifest(records: Iterable[SampleRecord], path: str | Path) -> None:
    """Write records as a deterministic UTF-8 CSV manifest."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda item: (item.machine_id, item.delay, item.sample_id))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for record in ordered:
            writer.writerow(asdict(record))


def read_manifest(path: str | Path) -> list[SampleRecord]:
    """Read a manifest and reject duplicate sample identifiers."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        SampleRecord(
            **{
                **row,
                "height": int(row.get("height") or 0),
                "width": int(row.get("width") or 0),
            }
        )
        for row in rows
    ]
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate sample_id found in manifest: {path}")
    return records
