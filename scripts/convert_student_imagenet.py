"""Convert official local classification weights without changing Student models."""

import argparse
import json

from fm2edge.models.imagenet_conversion import SOURCES, convert_imagenet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, choices=list(SOURCES))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha256")
    args = parser.parse_args()
    report = convert_imagenet(args.student, args.source, args.output, args.source_sha256)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in {"mapping", "loaded_keys", "head_keys_not_loaded", "missing_feature_keys"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
