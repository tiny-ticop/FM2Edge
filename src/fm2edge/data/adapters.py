"""Adapters that convert external layouts into the canonical manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from fm2edge.data.records import SampleRecord

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _sample_id(dataset: str, relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{dataset}-{digest}"


def scan_machine_folders(
    root: str | Path,
    *,
    images_dir: str = "images",
    masks_dir: str = "masks",
    phase_folder: str = "delay",
    mask_extension: str | None = None,
    dataset_name: str = "machine",
) -> list[SampleRecord]:
    """Scan machine folders and pair images with masks by relative path or stem.

    ``mask_extension`` permits lossless PNG masks to accompany JPEG source images.
    For example, ``frame.jpg`` is paired with ``frame.png`` when it is ``.png``.
    """
    root_path = Path(root)
    image_root = root_path / images_dir
    mask_root = root_path / masks_dir
    if not image_root.is_dir() or not mask_root.is_dir():
        raise FileNotFoundError(f"Expected {image_root} and {mask_root}")

    records: list[SampleRecord] = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(image_root)
        parts = relative.parts
        if len(parts) < 4 or parts[1] != phase_folder:
            continue
        machine_id, delay = parts[0], parts[2]
        mask_relative = relative.with_suffix(mask_extension) if mask_extension else relative
        mask_path = mask_root / mask_relative
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found for {image_path}: expected {mask_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            SampleRecord(
                sample_id=_sample_id(dataset_name, relative),
                image_path=(Path(images_dir) / relative).as_posix(),
                mask_path=(Path(masks_dir) / mask_relative).as_posix(),
                machine_id=machine_id,
                delay=str(delay),
                dataset_name=dataset_name,
                height=height,
                width=width,
            )
        )
    if not records:
        raise ValueError(f"No image/mask pairs found under {image_root}")
    return records


def scan_cityscapes(root: str | Path, dataset_name: str = "cityscapes") -> list[SampleRecord]:
    """Scan labeled Cityscapes train/val images, treating each city as a machine."""
    root_path = Path(root)
    records: list[SampleRecord] = []
    for source_split in ("train", "val"):
        image_root = root_path / "leftImg8bit" / source_split
        for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
            city = image_path.parent.name
            stem = image_path.name.removesuffix("_leftImg8bit.png")
            tokens = stem.split("_")
            delay = "_".join(tokens[1:3]) if len(tokens) >= 3 else ""
            mask_path = root_path / "gtFine" / source_split / city / f"{stem}_gtFine_labelIds.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Cityscapes mask not found: {mask_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            relative_image = image_path.relative_to(root_path)
            records.append(
                SampleRecord(
                    sample_id=_sample_id(dataset_name, relative_image),
                    image_path=relative_image.as_posix(),
                    mask_path=mask_path.relative_to(root_path).as_posix(),
                    machine_id=city,
                    delay=delay,
                    dataset_name=dataset_name,
                    height=height,
                    width=width,
                    split_source=source_split,
                )
            )
    if not records:
        raise ValueError(f"No labeled Cityscapes images found under {root_path}")
    return records
