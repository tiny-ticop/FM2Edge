"""Lightweight segmentation model registry."""

from fm2edge.models.registry import build_student

__all__ = ["build_student"]
from fm2edge.models.foundation import build_foundation_segmentor, build_teacher

__all__ = ["build_foundation_segmentor", "build_teacher"]
