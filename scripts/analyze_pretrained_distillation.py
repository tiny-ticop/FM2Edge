"""Aggregate Phase5.5 alone or compare with random-initialized Phase5 artifacts."""

import argparse

from fm2edge.analysis.pretrained_study import analyze_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--phase5-root")
    parser.add_argument("--output")
    parser.add_argument(
        "--prediction-samples",
        type=int,
        default=0,
        help="Optional CPU inference: first N sorted test sample IDs per fold",
    )
    args = parser.parse_args()
    if args.prediction_samples < 0:
        parser.error("--prediction-samples must be >= 0")
    print(analyze_pretrained(args.plan, args.phase5_root, args.output, args.prediction_samples))


if __name__ == "__main__":
    main()
