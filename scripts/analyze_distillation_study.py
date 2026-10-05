"""Aggregate completed KD runs, optionally paired with historical Baseline results."""

import argparse

from fm2edge.analysis.distillation_study import analyze_distillation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/knowledge_distillation/plan/experiment_plan.csv")
    parser.add_argument("--baseline-root", default=None)
    args = parser.parse_args()
    print(analyze_distillation(args.plan, args.baseline_root))


if __name__ == "__main__":
    main()
