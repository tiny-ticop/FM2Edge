"""Oxford-IIIT Pet downloader and manifest adapter.

This module is intentionally isolated from the core data pipeline. Once a
canonical manifest is produced, no Oxford-specific code is used by training or
evaluation.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from fm2edge.data.records import SampleRecord


@dataclass(frozen=True)
class DownloadResource:
    url: str
    filename: str
    md5: str


RESOURCES = (
    DownloadResource(
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
        "images.tar.gz",
        "5c4f3ee8e5d25df40f4fd59a7f44e54c",
    ),
    DownloadResource(
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
        "annotations.tar.gz",
        "95a8c909bbe2e81eed6a22bccdf3f68f",
    ),
)


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(resource: DownloadResource, archive_dir: Path) -> Path:
    destination = archive_dir / resource.filename
    if destination.is_file() and _file_md5(destination) == resource.md5:
        print(f"using verified archive: {destination}")
        return destination
    if destination.exists():
        destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(resource.url, headers={"User-Agent": "FM2Edge/0.1"})
    print(f"downloading {resource.url}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if _file_md5(temporary) != resource.md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"MD5 mismatch after downloading {resource.filename}")
    temporary.replace(destination)
    return destination


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination_resolved):
                raise ValueError(f"Archive member escapes destination: {member.name}")
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive member: {member.name}")
        archive.extractall(destination, members=members)


def download_oxford_pet(root: str | Path) -> None:
    """Download, verify, and safely extract the two official archives."""
    root_path = Path(root)
    archive_dir = root_path / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for resource in RESOURCES:
        archive = _download(resource, archive_dir)
        expected = root_path / (
            "images" if resource.filename.startswith("images") else "annotations"
        )
        if expected.is_dir():
            print(f"using extracted directory: {expected}")
            continue
        print(f"extracting {archive} to {root_path}")
        _safe_extract(archive, root_path)
    if not (root_path / "images").is_dir() or not (root_path / "annotations" / "trimaps").is_dir():
        raise RuntimeError(f"Oxford-IIIT Pet extraction is incomplete under {root_path}")


def _sample_id(relative_image: Path) -> str:
    digest = hashlib.sha1(relative_image.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"oxford_pet-{digest}"


def scan_oxford_pet(root: str | Path, dataset_name: str = "oxford_iiit_pet") -> list[SampleRecord]:
    """Build records from official lists, treating breed as machine/domain."""
    root_path = Path(root)
    images_dir = root_path / "images"
    annotations_dir = root_path / "annotations"
    trimaps_dir = annotations_dir / "trimaps"
    records: list[SampleRecord] = []
    seen_ids: set[str] = set()
    for source_split in ("trainval", "test"):
        list_path = annotations_dir / f"{source_split}.txt"
        if not list_path.is_file():
            raise FileNotFoundError(f"Oxford split list not found: {list_path}")
        for line in list_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            image_id = line.split()[0]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate Oxford image id: {image_id}")
            seen_ids.add(image_id)
            breed = image_id.rsplit("_", 1)[0]
            image_path = images_dir / f"{image_id}.jpg"
            mask_path = trimaps_dir / f"{image_id}.png"
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing Oxford pair: {image_path}, {mask_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            relative_image = image_path.relative_to(root_path)
            records.append(
                SampleRecord(
                    sample_id=_sample_id(relative_image),
                    image_path=relative_image.as_posix(),
                    mask_path=mask_path.relative_to(root_path).as_posix(),
                    machine_id=breed,
                    delay="",
                    dataset_name=dataset_name,
                    height=height,
                    width=width,
                    split_source=source_split,
                )
            )
    breeds = {record.machine_id for record in records}
    if len(breeds) < 3:
        raise ValueError(f"Expected multiple Oxford breeds, found {sorted(breeds)}")
    return records
