"""Paired image/mask transforms with explicit interpolation rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from fm2edge.config import AugmentationConfig


@dataclass(frozen=True)
class SegmentationTransform:
    """Resize a pair and normalize the image; masks always use nearest neighbor."""

    size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    ignore_index: int = 255
    mask_value_map: dict[int, int] | None = None
    augmentation: AugmentationConfig | None = None

    @staticmethod
    def _factor(magnitude: float) -> float:
        return 1.0 + (float(torch.rand(())) * 2.0 - 1.0) * magnitude

    def _augment(self, image: Image.Image) -> tuple[Image.Image, float]:
        config = self.augmentation
        if config is None or config.preset == "none":
            return image, 0.0
        preset = config.preset
        apply = float(torch.rand(())) < config.probability
        if not apply:
            return image, 0.0
        if preset in {"brightness_contrast", "combined"}:
            image = ImageEnhance.Brightness(image).enhance(self._factor(config.brightness))
            image = ImageEnhance.Contrast(image).enhance(self._factor(config.contrast))
        if preset in {"color", "combined"}:
            image = ImageEnhance.Color(image).enhance(self._factor(config.saturation))
            hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
            shift = round((float(torch.rand(())) * 2.0 - 1.0) * config.hue * 255)
            hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
            image = Image.fromarray(hsv, mode="HSV").convert("RGB")
        if preset in {"blur", "combined"}:
            radius = float(torch.rand(())) * config.blur_radius
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        noise_std = config.noise_std if preset in {"noise", "combined"} else 0.0
        return image, noise_std

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = self.size
        image = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        image, noise_std = self._augment(image)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        if noise_std > 0:
            image_tensor = (image_tensor + torch.randn_like(image_tensor) * noise_std).clamp(0, 1)
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
