"""Record exact local DINO repository revisions and weight checksums."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from fm2edge.analysis.foundation_study import load_foundation_study
from fm2edge.engine.utils import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analyses/foundation_probe.yaml")
    parser.add_argument("--output", default="configs/foundation_models.lock.yaml")
    args = parser.parse_args()
    study = load_foundation_study(args.config)
    output = Path(args.output)
    existing = (
        yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        if output.is_file()
        else {}
    )
    models: dict[str, object] = dict(existing.get("models", {}))
    for name, teacher in study["teachers"].items():
        repository = Path(teacher["repository_path"])
        if not (repository / "hubconf.py").is_file():
            continue
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        weights = teacher.get("weights_path")
        models[name] = {
            "family": teacher["family"],
            "variant": teacher["variant"],
            "repository_path": repository.as_posix(),
            "repository_commit": commit,
            "weights_path": weights,
            "checkpoint_sha256": file_sha256(weights)
            if weights and Path(weights).is_file()
            else None,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"models": models}, handle, sort_keys=False)
    print(f"foundation model lock: {output}")


if __name__ == "__main__":
    main()
