"""Rebuild machine-baseline comparison files from completed results."""

from __future__ import annotations

import argparse

from run_machine_baselines import CONFIGS, aggregate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    names = {
        model: (
            config.stem
            if args.mode == "full"
            else config.stem.replace("machine_", "machine_smoke_", 1)
        )
        for model, config in CONFIGS.items()
    }
    aggregate_results(args.mode, names)


if __name__ == "__main__":
    main()
