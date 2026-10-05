"""Generate KD experiment configurations without loading images or Teacher weights."""

import argparse

from fm2edge.analysis.distillation_study import build_distillation_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analyses/knowledge_distillation.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(build_distillation_plan(args.config, smoke=args.smoke))


if __name__ == "__main__":
    main()
