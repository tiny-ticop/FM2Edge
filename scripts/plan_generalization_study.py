"""Generate the configurable machine-generalization experiment plan."""

from __future__ import annotations

import argparse

from fm2edge.analysis.study import build_study_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/analyses/machine_generalization.yaml"
    )
    args = parser.parse_args()
    plan = build_study_plan(args.config)
    print(f"experiment plan: {plan}")


if __name__ == "__main__":
    main()
