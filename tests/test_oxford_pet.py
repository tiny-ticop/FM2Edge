import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fm2edge.data.public.oxford_pet import _safe_extract, scan_oxford_pet


def _write_pair(root: Path, image_id: str) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "trimaps").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((7, 9, 3), dtype=np.uint8)).save(root / "images" / f"{image_id}.jpg")
    Image.fromarray(np.ones((7, 9), dtype=np.uint8)).save(
        root / "annotations" / "trimaps" / f"{image_id}.png"
    )


def test_scan_oxford_pet_uses_breed_as_domain(tmp_path: Path) -> None:
    ids = ("Abyssinian_1", "beagle_2", "pug_3")
    for image_id in ids:
        _write_pair(tmp_path, image_id)
    annotations = tmp_path / "annotations"
    (annotations / "trainval.txt").write_text(
        "Abyssinian_1 1 1 1\nbeagle_2 2 2 1\n", encoding="utf-8"
    )
    (annotations / "test.txt").write_text("pug_3 3 2 1\n", encoding="utf-8")

    records = scan_oxford_pet(tmp_path)
    assert {record.machine_id for record in records} == {"Abyssinian", "beagle", "pug"}
    assert {record.split_source for record in records} == {"trainval", "test"}
    assert all(record.delay == "" for record in records)


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"bad"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="escapes destination"):
        _safe_extract(archive_path, tmp_path / "output")
