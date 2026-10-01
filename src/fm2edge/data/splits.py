"""Leakage-safe machine-level split generation."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from fm2edge.data.records import SampleRecord


@dataclass(frozen=True)
class MachineSplit:
    train_machines: tuple[str, ...]
    val_machines: tuple[str, ...]
    test_machines: tuple[str, ...]
    seed: int
    fold: int
    train_sample_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        train, val, test = map(set, (self.train_machines, self.val_machines, self.test_machines))
        if not train or not val or not test:
            raise ValueError("train, val, and test machine sets must all be non-empty")
        if train & val or train & test or val & test:
            raise ValueError("Machine leakage detected between train/val/test")

    def to_dict(self) -> dict[str, object]:
        return {
            "train_machines": list(self.train_machines),
            "val_machines": list(self.val_machines),
            "test_machines": list(self.test_machines),
            "seed": self.seed,
            "fold": self.fold,
            **(
                {"train_sample_ids": list(self.train_sample_ids)}
                if self.train_sample_ids
                else {}
            ),
        }


def make_machine_folds(
    records: Iterable[SampleRecord], n_folds: int = 5, seed: int = 42
) -> list[MachineSplit]:
    """Create outer test folds and a distinct rotating validation fold."""
    machines = sorted({record.machine_id for record in records})
    if len(machines) < 3:
        raise ValueError("At least three machines are required for train/val/test separation")
    n_folds = min(n_folds, len(machines))
    if n_folds < 3:
        raise ValueError("n_folds must be at least 3")
    random.Random(seed).shuffle(machines)
    buckets = [tuple(machines[index::n_folds]) for index in range(n_folds)]
    folds: list[MachineSplit] = []
    for index in range(n_folds):
        test = buckets[index]
        val = buckets[(index + 1) % n_folds]
        excluded = set(test) | set(val)
        train = tuple(machine for machine in machines if machine not in excluded)
        split = MachineSplit(train, val, test, seed=seed, fold=index + 1)
        split.validate()
        folds.append(split)
    return folds


def save_split(split: MachineSplit, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split.to_dict(), handle, sort_keys=False, allow_unicode=True)


def load_split(path: str | Path) -> MachineSplit:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    split = MachineSplit(
        train_machines=tuple(str(value) for value in raw["train_machines"]),
        val_machines=tuple(str(value) for value in raw["val_machines"]),
        test_machines=tuple(str(value) for value in raw["test_machines"]),
        seed=int(raw["seed"]),
        fold=int(raw["fold"]),
        train_sample_ids=tuple(str(value) for value in raw.get("train_sample_ids", ())),
    )
    split.validate()
    return split
