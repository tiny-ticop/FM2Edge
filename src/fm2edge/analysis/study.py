"""Generate immutable, leakage-safe experiment plans from a compact YAML study config."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml

from fm2edge.data.records import SampleRecord, read_manifest
from fm2edge.data.splits import MachineSplit, load_split, save_split


def load_study_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    required = ("name", "output_dir", "data", "models", "folds", "suites")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Study config is missing keys: {missing}")
    return config


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _nested_machine_order(
    records: list[SampleRecord], machine: str, seed: int
) -> list[SampleRecord]:
    """Create one reusable, delay-interleaved order per machine."""
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.machine_id == machine:
            groups[record.delay].append(record)
    queues: dict[str, deque[SampleRecord]] = {}
    for delay, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: record.sample_id)
        random.Random(_stable_seed(seed, machine, delay)).shuffle(ordered)
        queues[delay] = deque(ordered)
    output: list[SampleRecord] = []
    while any(queues.values()):
        for delay in sorted(queues):
            if queues[delay]:
                output.append(queues[delay].popleft())
    return output


def _choose_combinations(
    machines: tuple[str, ...], count: int, maximum: int | None, seed: int
) -> list[tuple[str, ...]]:
    combinations = list(itertools.combinations(sorted(machines), count))
    if maximum is not None and len(combinations) > maximum:
        rng = random.Random(_stable_seed(seed, machines, count))
        combinations = sorted(rng.sample(combinations, maximum))
    return combinations


def _allocate_total(machines: tuple[str, ...], total: int, fold: int) -> dict[str, int]:
    if total < len(machines):
        raise ValueError(f"total_images={total} is smaller than machine count={len(machines)}")
    base, remainder = divmod(total, len(machines))
    rotated = sorted(machines)
    offset = (fold - 1) % len(rotated)
    rotated = rotated[offset:] + rotated[:offset]
    allocation = {machine: base for machine in machines}
    for machine in rotated[:remainder]:
        allocation[machine] += 1
    return allocation


def _select_samples(
    orders: dict[str, list[SampleRecord]], allocation: dict[str, int | str]
) -> tuple[str, ...]:
    selected: list[str] = []
    for machine, requested in sorted(allocation.items()):
        available = orders[machine]
        count = len(available) if requested == "all" else int(requested)
        if count > len(available):
            raise ValueError(
                f"Machine {machine!r} has {len(available)} samples, but {count} were requested"
            )
        selected.extend(record.sample_id for record in available[:count])
    return tuple(selected)


def _condition_label(suite: str, **values: object) -> str:
    parts = [suite]
    for key, value in values.items():
        text = str(value).replace(" ", "").replace("/", "-")
        parts.append(f"{key}-{text}")
    return "__".join(parts)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def _run_record(
    *,
    study: dict[str, Any],
    suite: str,
    condition: str,
    model_name: str,
    base_config_path: Path,
    fold: int,
    seed: int,
    split: MachineSplit,
    train_machines: tuple[str, ...],
    train_sample_ids: tuple[str, ...],
    allocation: dict[str, int | str],
    augmentation: str,
    generated_root: Path,
) -> dict[str, object]:
    fingerprint = json.dumps(
        {
            "condition": condition,
            "train_machines": train_machines,
            "allocation": allocation,
            "augmentation": augmentation,
            "augmentation_parameters": study["suites"].get("augmentation", {}).get(
                "parameters", {}
            ),
            "training": study.get("training", {}),
            "manifest": study["data"]["manifest"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    short_condition = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8]
    run_id = f"{suite}__{model_name}__f{fold:02d}__{short_condition}__s{seed}"
    split_path = generated_root / "splits" / f"{run_id}.yaml"
    config_path = generated_root / "configs" / f"{run_id}.yaml"
    generated_split = MachineSplit(
        train_machines=train_machines,
        val_machines=split.val_machines,
        test_machines=split.test_machines,
        seed=seed,
        fold=fold,
        train_sample_ids=train_sample_ids,
    )
    generated_split.validate()
    save_split(generated_split, split_path)

    experiment = _read_yaml(base_config_path)
    experiment["name"] = run_id
    experiment["output_dir"] = str(Path(study["output_dir"]) / "runs")
    experiment["data"]["manifest"] = str(study["data"]["manifest"])
    experiment["data"]["root"] = str(study["data"]["root"])
    experiment["data"]["split"] = split_path.as_posix()
    training = study.get("training", {})
    for key, value in training.items():
        experiment["train"][key] = value
    experiment["train"]["seed"] = seed
    parameters = dict(study["suites"].get("augmentation", {}).get("parameters", {}))
    experiment["augmentation"] = {"preset": augmentation, **parameters}
    _write_yaml(experiment, config_path)

    result_dir = Path(experiment["output_dir"]) / run_id / f"fold_{fold:02d}"
    return {
        "run_id": run_id,
        "suite": suite,
        "condition": condition,
        "model": model_name,
        "fold": fold,
        "seed": seed,
        "train_machine_count": len(train_machines),
        "train_machines": json.dumps(train_machines, ensure_ascii=False),
        "val_machines": json.dumps(split.val_machines, ensure_ascii=False),
        "test_machines": json.dumps(split.test_machines, ensure_ascii=False),
        "train_image_count": len(train_sample_ids),
        "allocation": json.dumps(allocation, ensure_ascii=False, sort_keys=True),
        "augmentation": augmentation,
        "config_path": config_path.as_posix(),
        "split_path": split_path.as_posix(),
        "result_dir": result_dir.as_posix(),
    }


def build_study_plan(config_path: str | Path) -> Path:
    study = load_study_config(config_path)
    output = Path(study["output_dir"])
    generated_root = output / "generated"
    plan_dir = output / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    records = read_manifest(study["data"]["manifest"])
    seeds = [int(seed) for seed in study.get("seeds", [42])]
    sample_seed = int(study.get("sample_seed", 42))
    machines = sorted({record.machine_id for record in records})
    orders = {
        machine: _nested_machine_order(records, machine, sample_seed) for machine in machines
    }
    models = {str(key): Path(value) for key, value in study["models"].items()}
    suites = study["suites"]
    runs: list[dict[str, object]] = []

    for fold in [int(value) for value in study["folds"]]:
        base_split = load_split(Path(study["data"]["splits_dir"]) / f"fold_{fold:02d}.yaml")
        for seed in seeds:
            conditions: list[tuple[str, str, tuple[str, ...], dict[str, int | str], str]] = []
            if suites.get("reference", {}).get("enabled", True):
                allocation = {machine: "all" for machine in base_split.train_machines}
                conditions.append(
                    (
                        "reference",
                        _condition_label("reference", images="all", aug="none"),
                        base_split.train_machines,
                        allocation,
                        "none",
                    )
                )

            diversity = suites.get("machine_diversity", {})
            if diversity.get("enabled", False):
                total = int(diversity["total_images"])
                maximum = diversity.get("max_combinations_per_count")
                maximum = int(maximum) if maximum is not None else None
                for count in [int(value) for value in diversity["machine_counts"]]:
                    if count > len(base_split.train_machines):
                        raise ValueError(
                            f"Fold {fold} has {len(base_split.train_machines)} train candidates; "
                            f"machine count {count} is invalid"
                        )
                    combos = _choose_combinations(
                        base_split.train_machines, count, maximum, sample_seed + fold
                    )
                    for combo_index, combo in enumerate(combos, start=1):
                        allocation = _allocate_total(combo, total, fold)
                        conditions.append(
                            (
                                "machine_diversity",
                                _condition_label(
                                    "machine_diversity",
                                    k=count,
                                    total=total,
                                    combo=combo_index,
                                ),
                                combo,
                                allocation,
                                "none",
                            )
                        )

            image_suite = suites.get("images_per_machine", {})
            if image_suite.get("enabled", False):
                for value in image_suite["counts"]:
                    count: int | str = "all" if str(value).lower() == "all" else int(value)
                    if count == "all" and suites.get("reference", {}).get("enabled", True):
                        continue
                    allocation = {machine: count for machine in base_split.train_machines}
                    conditions.append(
                        (
                            "images_per_machine",
                            _condition_label("images_per_machine", n=count),
                            base_split.train_machines,
                            allocation,
                            "none",
                        )
                    )

            aug_suite = suites.get("augmentation", {})
            if aug_suite.get("enabled", False):
                for preset in [str(value) for value in aug_suite["presets"]]:
                    allocation = {machine: "all" for machine in base_split.train_machines}
                    conditions.append(
                        (
                            "augmentation",
                            _condition_label("augmentation", preset=preset),
                            base_split.train_machines,
                            allocation,
                            preset,
                        )
                    )

            seen: set[tuple[str, str]] = set()
            for suite, condition, train_machines, allocation, augmentation in conditions:
                deduplication_key = (condition, json.dumps(train_machines))
                if deduplication_key in seen:
                    continue
                seen.add(deduplication_key)
                sample_ids = _select_samples(orders, allocation)
                for model_name, base_config in models.items():
                    runs.append(
                        _run_record(
                            study=study,
                            suite=suite,
                            condition=condition,
                            model_name=model_name,
                            base_config_path=base_config,
                            fold=fold,
                            seed=seed,
                            split=base_split,
                            train_machines=train_machines,
                            train_sample_ids=sample_ids,
                            allocation=allocation,
                            augmentation=augmentation,
                            generated_root=generated_root,
                        )
                    )

    plan_path = plan_dir / "experiment_plan.csv"
    if not runs:
        raise ValueError("The study configuration generated no runs")
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        counts[str(run["suite"])] += 1
    with (plan_dir / "plan_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "study": study["name"],
                "total_runs": len(runs),
                "runs_by_suite": dict(sorted(counts.items())),
                "models": list(models),
                "folds": study["folds"],
                "seeds": seeds,
            },
            handle,
            indent=2,
        )
    return plan_path
