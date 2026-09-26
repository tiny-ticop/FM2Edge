"""Paired image/mask transforms with explicit interpolation rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class SegmentationTransform:
    """Resize a pair and normalize the image; masks always use nearest neighbor."""

    size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    ignore_index: int = 255
    mask_value_map: dict[int, int] | None = None

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = self.size
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mean = torch.tensor(self.mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(self.std, dtype=torch.float32)[:, None, None]
        image_tensor = (image_tensor - mean) / std

        mask_array = np.asarray(mask, dtype=np.int64)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if self.mask_value_map:
            mapped = np.full(mask_array.shape, self.ignore_index, dtype=np.int64)
            for source, target in self.mask_value_map.items():
                mapped[mask_array == source] = target
            mask_array = mapped
        return image_tensor, torch.from_numpy(mask_array.copy()).long()
