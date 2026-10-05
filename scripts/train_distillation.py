"""Train KD, export the unchanged Student and evaluate held-out machines."""

import argparse

from fm2edge.engine.distillation import train_distillation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    print(
        train_distillation(
            args.config, resume=not args.no_resume, evaluate_test=not args.train_only
        )
    )


if __name__ == "__main__":
    main()
