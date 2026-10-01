"""Check local DINO repositories and weights without loading company images."""

from __future__ import annotations

import argparse
from pathlib import Path

from fm2edge.analysis.foundation_study import load_foundation_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analyses/foundation_probe.yaml")
    args = parser.parse_args()
    study = load_foundation_study(args.config)
    missing = 0
    for name, teacher in study["teachers"].items():
        repository = Path(teacher["repository_path"])
        repository_ok = (repository / "hubconf.py").is_file()
        weights = teacher.get("weights_path")
        weights_ok = teacher.get("initialization") == "random_init" or (
            bool(weights) and Path(weights).is_file()
        )
        status = "READY" if repository_ok and weights_ok else "NOT READY"
        print(
            f"{name}: {status}; repository={repository} ({repository_ok}); "
            f"weights={weights or 'not required'} ({weights_ok})"
        )
        missing += int(status != "READY")
    if missing:
        print(f"{missing} teacher configuration(s) are unavailable and will be skipped.")


if __name__ == "__main__":
    main()
