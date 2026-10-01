"""Generate the frozen-DINO probe experiment matrix without running it."""

from __future__ import annotations

import argparse

from fm2edge.analysis.foundation_study import build_foundation_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/analyses/foundation_probe.yaml"
    )
    args = parser.parse_args()
    print(f"experiment plan: {build_foundation_plan(args.config)}")


if __name__ == "__main__":
    main()
