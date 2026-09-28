from pathlib import Path

import numpy as np
from PIL import Image

from fm2edge.data.adapters import scan_cityscapes, scan_machine_folders


def _save_pair(image_path: Path, mask_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8)).save(image_path)
    Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(mask_path)


def test_machine_adapter_supports_configurable_phase_folder(tmp_path: Path) -> None:
    relative = Path("machine_A") / "acquisition_phase" / "90" / "frame.png"
    _save_pair(tmp_path / "images" / relative, tmp_path / "masks" / relative)
    records = scan_machine_folders(tmp_path, phase_folder="acquisition_phase")
    assert len(records) == 1
    assert records[0].machine_id == "machine_A"
    assert records[0].delay == "90"
    assert records[0].height == 8
    assert records[0].width == 9


def test_machine_adapter_pairs_jpeg_images_with_png_masks(tmp_path: Path) -> None:
    relative_image = Path("machine_01") / "delay" / "90" / "frame.jpg"
    relative_mask = relative_image.with_suffix(".png")
    _save_pair(tmp_path / "raw" / relative_image, tmp_path / "masks" / relative_mask)

    records = scan_machine_folders(
        tmp_path,
        images_dir="raw",
        masks_dir="masks",
        mask_extension=".png",
    )

    assert len(records) == 1
    assert records[0].image_path.endswith("frame.jpg")
    assert records[0].mask_path.endswith("frame.png")


def test_cityscapes_adapter_uses_city_as_machine(tmp_path: Path) -> None:
    image = tmp_path / "leftImg8bit" / "train" / "aachen" / "aachen_000001_000019_leftImg8bit.png"
    mask = tmp_path / "gtFine" / "train" / "aachen" / "aachen_000001_000019_gtFine_labelIds.png"
    _save_pair(image, mask)
    records = scan_cityscapes(tmp_path)
    assert len(records) == 1
    assert records[0].machine_id == "aachen"
    assert records[0].delay == "000001_000019"
    assert records[0].split_source == "train"
