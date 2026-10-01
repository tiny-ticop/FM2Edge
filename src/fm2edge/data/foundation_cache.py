"""Validated on-disk feature cache for frozen foundation backbones."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fm2edge.config import ExperimentConfig
from fm2edge.engine.utils import file_sha256


def repository_commit(path: str | None) -> str | None:
    if not path or not (Path(path) / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def cache_identity(config: ExperimentConfig) -> dict[str, Any]:
    model = config.model
    identity: dict[str, Any] = {
        "schema_version": 1,
        "family": model.family,
        "variant": model.variant,
        "initialization": model.initialization,
        "random_seed": config.train.seed if model.initialization == "random_init" else None,
        "feature_layers": model.feature_layers,
        "embedding_dim": model.embedding_dim,
        "image_size": list(config.data.image_size),
        "mean": list(config.data.mean),
        "std": list(config.data.std),
        "manifest_sha256": file_sha256(config.data.manifest),
        "weights_sha256": (
            file_sha256(model.weights_path) if model.weights_path else None
        ),
        "repository_commit": repository_commit(model.repository_path),
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["fingerprint"] = hashlib.sha256(serialized.encode()).hexdigest()
    return identity


def cache_file(cache_dir: str | Path, fingerprint: str, sample_id: str) -> Path:
    safe_id = hashlib.sha256(sample_id.encode()).hexdigest()
    return Path(cache_dir) / fingerprint[:16] / "features" / f"{safe_id}.pt"


def image_identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": path.as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_cache_metadata(cache_dir: str | Path, identity: dict[str, Any]) -> Path:
    root = Path(cache_dir) / str(identity["fingerprint"])[:16]
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metadata.json"
    path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return path


def validate_cache_metadata(cache_dir: str | Path, identity: dict[str, Any]) -> None:
    path = Path(cache_dir) / str(identity["fingerprint"])[:16] / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Foundation feature cache metadata is missing: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != identity:
        raise ValueError("Foundation feature cache metadata does not match this experiment")


class CachedFoundationDataset(Dataset):
    """Attach validated frozen-backbone features to a segmentation dataset."""

    def __init__(
        self, base: Dataset, cache_dir: str | Path, identity: dict[str, Any]
    ) -> None:
        self.base = base
        self.cache_dir = Path(cache_dir)
        self.identity = identity
        validate_cache_metadata(cache_dir, identity)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = dict(self.base[index])
        sample_id = str(sample["sample_id"])
        path = cache_file(self.cache_dir, str(self.identity["fingerprint"]), sample_id)
        if not path.is_file():
            raise FileNotFoundError(f"Cached DINO feature is missing for {sample_id}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") != self.identity["fingerprint"]:
            raise ValueError(f"Stale cached DINO feature for {sample_id}")
        image_path = Path(self.base.root) / Path(str(sample["image_path"]))
        if payload.get("image") != image_identity(image_path):
            raise ValueError(f"Source image changed after caching: {image_path}")
        sample["foundation_features"] = payload["features"]
        sample["foundation_padded_size"] = tuple(payload["padded_size"])
        return sample
