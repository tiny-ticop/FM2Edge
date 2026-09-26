from pathlib import Path

import numpy as np
from PIL import Image

from fm2edge.data.dataset import ManifestSegmentationDataset
from fm2edge.data.records import SampleRecord, write_manifest
from fm2edge.data.transforms import SegmentationTransform


def test_dataset_returns_common_interface(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(tmp_path / "images/a.png")
    Image.fromarray(np.ones((10, 12), dtype=np.uint8) * 255).save(tmp_path / "masks/a.png")
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        [SampleRecord("a", "images/a.png", "masks/a.png", "machine_a", "90", height=10, width=12)],
        manifest,
    )
    transform = SegmentationTransform(
        (8, 8), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), mask_value_map={0: 0, 255: 1}
    )
    dataset = ManifestSegmentationDataset(manifest, tmp_path, ["machine_a"], transform)
    sample = dataset[0]
    assert sample["image"].shape == (3, 8, 8)
    assert sample["mask"].shape == (8, 8)
    assert sample["mask"].unique().tolist() == [1]
    assert sample["machine_id"] == "machine_a"
