"""Dataset implementation backed only by the canonical manifest."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from fm2edge.data.records import SampleRecord, read_manifest
from fm2edge.data.transforms import SegmentationTransform


class ManifestSegmentationDataset(Dataset):
    """Read paired segmentation samples without dataset-specific path logic."""

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        machines: Collection[str],
        transform: SegmentationTransform,
        sample_ids: Collection[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        machine_set = set(machines)
        sample_set = set(sample_ids) if sample_ids is not None else None
        self.records = [
            record
            for record in read_manifest(manifest)
            if record.machine_id in machine_set
            and (sample_set is None or record.sample_id in sample_set)
        ]
        if not self.records:
            raise ValueError(f"No manifest samples matched machines: {sorted(machine_set)}")
        if sample_set is not None:
            found = {record.sample_id for record in self.records}
            missing = sample_set - found
            if missing:
                raise ValueError(f"Requested train sample IDs were not found: {sorted(missing)[:5]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record: SampleRecord = self.records[index]
        image_path = self.root / Path(record.image_path)
        mask_path = self.root / Path(record.mask_path)
        with Image.open(image_path) as image_handle, Image.open(mask_path) as mask_handle:
            image, mask = self.transform(image_handle, mask_handle)
        return {
            "image": image,
            "mask": mask,
            "sample_id": record.sample_id,
            "machine_id": record.machine_id,
            "delay": record.delay,
            "image_path": record.image_path,
            "original_size": (record.height, record.width),
        }
