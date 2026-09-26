"""Download the isolated Oxford-IIIT Pet PoC dataset."""

from __future__ import annotations

import argparse

from fm2edge.data.public.oxford_pet import download_oxford_pet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/oxford_pet")
    args = parser.parse_args()
    download_oxford_pet(args.root)
    print(f"Oxford-IIIT Pet is ready under {args.root}")


if __name__ == "__main__":
    main()
