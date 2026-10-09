"""Offline ImageNet import; historical Student architectures/loaders stay unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from fm2edge.engine.utils import file_sha256
from fm2edge.models.registry import build_student

SOURCES = {
    "mobilenet_v3_lraspp": "torchvision/mobilenet_v3_large/IMAGENET1K_V2",
    "pidnet_s": "XuJiacong/PIDNet-S/ImageNet",
}
MOBILENET_V2_SHA256 = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"
PID_HEADS = ("final_layer.", "seghead_p.", "seghead_d.")
# Classification pretraining may omit task-specific branches. Never allow holes
# within a module present in the source, or missing shared I-branch tensors.
PID_REQUIRED = ("conv1", "layer1", "layer2", "layer3", "layer4", "layer5")


def feature_key(student, key):
    return (
        key.startswith("backbone.")
        if student == "mobilenet_v3_lraspp"
        else not key.startswith(PID_HEADS)
    )


def official_key(student, key):
    """Convert destination key to official classification key (also used in tests)."""
    if student == "pidnet_s":
        return key
    if key.startswith("backbone.stem."):
        key = "features.0." + key.removeprefix("backbone.stem.")
    elif key.startswith("backbone.final."):
        key = "features.16." + key.removeprefix("backbone.final.")
    elif key.startswith("backbone.blocks."):
        index, suffix = key.removeprefix("backbone.blocks.").split(".", 1)
        key = f"features.{int(index) + 1}.{suffix}"
    else:
        raise ValueError(f"Not a MobileNet feature key: {key}")
    return key.replace(".reduce.", ".fc1.").replace(".expand.", ".fc2.")


def convert_state(student, payload):
    if student not in SOURCES:
        raise ValueError(f"Unsupported ImageNet Student: {student}")
    for wrapper in ("state_dict", "model"):
        if isinstance(payload, dict) and wrapper in payload:
            payload = payload[wrapper]
            break
    if not isinstance(payload, dict):
        raise TypeError("Expected a tensor state_dict")
    source = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Non-tensor state entry: {key}")
        key = key.removeprefix("module.")
        if key in source:
            raise ValueError(f"Duplicate normalized key: {key}")
        source[key] = value
    classification_head = any(
        k in {"final_layer.weight", "final_layer.conv2.weight"}
        and v.ndim >= 1
        and v.shape[0] == 1000
        for k, v in source.items()
    )
    if (
        student == "pidnet_s"
        and any(k.startswith(PID_HEADS) for k in source)
        and not classification_head
    ):
        raise ValueError(
            "Segmentation checkpoint detected; use official ImageNet classification weights"
        )
    with torch.random.fork_rng(devices=[]):
        model = build_student(student, 2)
    destination = model.state_dict()
    features = {k: v for k, v in destination.items() if feature_key(student, k)}
    loaded, mismatch, mapping = {}, [], {}
    for key, reference in features.items():
        origin = official_key(student, key)
        if origin in source:
            value = source[origin]
            if reference.shape != value.shape or reference.dtype != value.dtype:
                mismatch.append(key)
            else:
                loaded[key] = value.detach().cpu().clone()
                mapping[key] = origin
    missing = sorted(set(features) - set(loaded))
    # torchvision classification backbone must be complete, including BN buffers.
    unexpected_missing = list(missing) if student == "mobilenet_v3_lraspp" else []
    if student == "pidnet_s":
        present = {k.split(".", 1)[0] for k in source}
        unexpected_missing = [
            k for k in missing if k.split(".", 1)[0] in present | set(PID_REQUIRED)
        ]
    ignored = sorted(set(source) - set(mapping.values()))
    allowed_extra = (
        ("classifier.",)
        if student == "mobilenet_v3_lraspp"
        else ("classifier.", "fc.", "linear.", "last_layer.")
        + (PID_HEADS if classification_head else ())
    )
    unexpected_source = [k for k in ignored if not k.startswith(allowed_extra)]
    if mismatch or unexpected_missing or unexpected_source or not loaded:
        raise ValueError(
            f"Invalid ImageNet import: shape/dtype={mismatch}; missing={unexpected_missing}; "
            f"unknown_source={unexpected_source}"
        )
    parameters = dict(model.named_parameters())
    total = sum(p.numel() for k, p in parameters.items() if k in features)
    imported = sum(p.numel() for k, p in parameters.items() if k in loaded)
    report = {
        "schema": 1,
        "student": student,
        "source_id": SOURCES[student],
        "loaded_keys": sorted(loaded),
        "mapping": mapping,
        "missing_feature_keys": missing,
        "head_keys_not_loaded": sorted(set(destination) - set(features)),
        "ignored_source_keys": ignored,
        "shape_mismatches": mismatch,
        "feature_parameter_count": total,
        "loaded_feature_parameter_count": imported,
        "feature_parameter_coverage": imported / total,
        "student_parameter_coverage": imported / sum(p.numel() for p in parameters.values()),
        "missing_feature_modules": sorted({k.split(".", 1)[0] for k in missing}),
    }
    return loaded, report


def convert_imagenet(student, source, output, expected_sha256=None):
    source, output = Path(source), Path(output)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite converted weights: {output}")
    digest = file_sha256(source)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError("Source SHA-256 mismatch")
    if student == "mobilenet_v3_lraspp" and digest != MOBILENET_V2_SHA256:
        raise ValueError("Not the official MobileNetV3 IMAGENET1K_V2 weight hash")
    state, report = convert_state(
        student, torch.load(source, map_location="cpu", weights_only=True)
    )
    report.update(source_sha256=digest, converter_sha256=file_sha256(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "imagenet_import": report}, output)
    report["converted_sha256"] = file_sha256(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_converted(student, path):
    path = Path(path)
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if report["student"] != student or report["converted_sha256"] != file_sha256(path):
        raise ValueError("Converted Student identity/SHA-256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("imagenet_import", {}).get("source_sha256") != report["source_sha256"]:
        raise ValueError("Import report mismatch")
    if set(payload["model"]) != set(report["loaded_keys"]):
        raise ValueError("Converted key audit mismatch")
    return report
