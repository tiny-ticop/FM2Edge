from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from fm2edge.analysis.distillation_study import build_distillation_plan
from fm2edge.data.records import SampleRecord, write_manifest
from fm2edge.data.splits import MachineSplit, save_split
from fm2edge.distillation_config import load_distillation_config
from fm2edge.engine.distillation import train_distillation, validate_probe
from fm2edge.losses.distillation import (
    MaskedGenerativeDistillation,
    QuerySoftDistillation,
    align_features,
    heterogeneous_knowledge,
)
from fm2edge.models.distillation import set_student_stage, student_features
from fm2edge.models.foundation import FrozenDinoSegmentor
from fm2edge.models.outputs import ModelOutput
from fm2edge.models.registry import build_student


class FakeDino(nn.Module):
    def __init__(self, patch):
        super().__init__()
        self.patch = nn.Conv2d(3, 8, patch, stride=patch)

    def get_intermediate_layers(self, images, n, **kwargs):
        features = self.patch(images)
        return tuple(features for _ in range(n))


def fake_teacher(family="dinov2", mode="online"):
    patch = 14 if family == "dinov2" else 16
    return FrozenDinoSegmentor(
        FakeDino(patch),
        family=family,
        variant="vits14" if patch == 14 else "vits16",
        head="linear",
        num_classes=2,
        feature_layers=1,
        embedding_dim=8,
        patch_size=patch,
        feature_mode=mode,
    )


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())

    def forward(self, images):
        features = self.net(images)
        return features, features


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Conv2d(8, 2, 1)

    def forward(self, low, high):
        return self.classifier(high), high


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = TinyHead()

    def forward(self, images):
        low, high = self.backbone(images)
        logits, features = self.head(low, high)
        return ModelOutput(logits=logits, features={"kd": features})


def make_config(tmp_path, method, family="dinov2", mode="online"):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    records = []
    for machine in ("train", "val", "test"):
        for index in range(2):
            image_path = f"{machine}_{index}.png"
            mask_path = f"{machine}_{index}_mask.png"
            Image.new("RGB", (32, 32), (index * 80, 80, 120)).save(data / image_path)
            Image.new("L", (32, 32), index).save(data / mask_path)
            records.append(
                SampleRecord(
                    f"{machine}-{index}", image_path, mask_path, machine, "90", height=32, width=32
                )
            )
    manifest = data / "manifest.csv"
    write_manifest(records, manifest)
    split = tmp_path / "fold.yaml"
    save_split(MachineSplit(("train",), ("val",), ("test",), 42, 1), split)
    config = {
        "name": method,
        "output_dir": str(tmp_path / "results"),
        "data": {
            "root": str(data),
            "manifest": str(manifest),
            "split": str(split),
            "num_classes": 2,
            "image_size": [32, 32],
        },
        "model": {"name": "mobilenet_v3_lraspp"},
        "train": {"epochs": 1, "batch_size": 2, "num_workers": 0, "device": "cpu", "amp": False},
        "distillation": {
            "method": method,
            "representation_epochs": 1,
            "task_epochs": 1,
            "warmup_epochs": 0,
            "max_tokens": 16,
            "teacher": {
                "family": family,
                "variant": "vits14" if family == "dinov2" else "vits16",
                "embedding_dim": 8,
                "feature_layers": 1,
                "feature_mode": mode,
                "cache_dir": str(tmp_path / "cache"),
            },
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_losses_gradients_and_ignore():
    torch.set_num_threads(1)
    for objective in (MaskedGenerativeDistillation(4, 8, 0.5), QuerySoftDistillation(4, 8, 8)):
        student = torch.randn(2, 4, 3, 4, requires_grad=True)
        teacher = torch.randn(2, 8, 3, 4, requires_grad=True)
        loss = objective(student, teacher, torch.ones(2, 1, 3, 4))
        assert torch.isfinite(loss)
        loss.backward()
        assert student.grad is not None
        assert teacher.grad is None
        assert objective(student, teacher, torch.zeros(2, 1, 3, 4)).item() == 0
    s = torch.randn(2, 2, 4, 5, requires_grad=True)
    t = torch.randn_like(s, requires_grad=True)
    target = torch.randint(0, 2, (2, 4, 5))
    loss = heterogeneous_knowledge(s, t, target, 255, 4)
    loss.backward()
    assert s.grad is not None and t.grad is None
    assert heterogeneous_knowledge(s, t, torch.full_like(target, 255), 255, 4).item() == 0


def test_coordinate_crop_not_stretched_padding():
    student = torch.randn(1, 4, 2, 3)
    teacher = torch.arange(6).reshape(1, 1, 2, 3).float()
    target = torch.zeros(1, 15, 19, dtype=torch.long)
    sf, tf, valid = align_features(student, teacher, target, (28, 42), 255)
    assert tf.shape[-2:] == (2, 3)
    assert not torch.allclose(tf, teacher)
    assert sf.shape[-2:] == tf.shape[-2:] and valid.min() == 1


@pytest.mark.parametrize("name", ["pidnet_s", "mobilenet_v3_lraspp"])
def test_real_student_partition_and_export_contract(name):
    torch.set_num_threads(1)
    student = build_student(name, 2)
    keys = set(student.state_dict())
    set_student_stage(student, name, "representation")
    _, features = student_features(student, torch.randn(2, 3, 64, 64), True)
    features.mean().backward()
    assert any(p.grad is not None for p in student.parameters() if p.requires_grad)
    set_student_stage(student, name, "task")
    before = {
        key: value.clone() for key, value in student.state_dict().items() if "running_" in key
    }
    student_features(student, torch.randn(2, 3, 64, 64), True)
    for key, value in before.items():
        if name == "mobilenet_v3_lraspp" and key.startswith("backbone."):
            assert torch.equal(value, student.state_dict()[key])
        if name == "pidnet_s" and not key.startswith(("final_layer.", "seghead_")):
            assert torch.equal(value, student.state_dict()[key])
    assert set(student.state_dict()) == keys


@pytest.mark.parametrize("method", ["none", "mgd", "gkd_cnn_source_only"])
@pytest.mark.parametrize("family", ["dinov2", "dinov3"])
def test_synthetic_end_to_end(tmp_path, monkeypatch, method, family):
    torch.set_num_threads(1)
    monkeypatch.setattr("fm2edge.engine.distillation.build_student", lambda *args: TinyStudent())
    path = make_config(tmp_path, method, family)
    teacher = fake_teacher(family)
    output = train_distillation(path, teacher_override=teacher)
    assert (output / "completed.json").exists()
    assert (output / "predictions/worst").is_dir()
    exported = torch.load(output / "student.pt", weights_only=True)
    restored = TinyStudent()
    restored.load_state_dict(exported["model"], strict=True)
    assert not any("teacher" in key or "adapter" in key for key in exported["model"])
    assert all(p.grad is None for p in teacher.parameters())
    before = (output / "history/epochs.csv").read_bytes()
    train_distillation(path, teacher_override=teacher)
    assert (output / "history/epochs.csv").read_bytes() == before
    last = torch.load(output / "checkpoints/last.pt", weights_only=True)
    assert "teacher" not in last and "rng" in last and "scaler" in last


def test_cached_teacher_and_resume_final_evaluation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr("fm2edge.engine.distillation.build_student", lambda *args: TinyStudent())
    path = make_config(tmp_path, "mgd", mode="cached")
    teacher = fake_teacher(mode="cached")
    output = train_distillation(path, evaluate_test=False, teacher_override=teacher)
    assert not (output / "metrics/summary.json").exists()
    assert len(list((tmp_path / "cache").rglob("*.pt"))) == 2  # TRAIN ONLY
    train_distillation(path, teacher_override=teacher)
    assert (output / "completed.json").exists()
    content = yaml.safe_load(path.read_text())
    content["train"]["epochs"] = 2
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="identity changed"):
        train_distillation(path, teacher_override=teacher)


def test_probe_provenance_rejects_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("fm2edge.engine.distillation.build_student", lambda *args: TinyStudent())
    path = make_config(tmp_path, "heteroakd")
    with pytest.raises(FileNotFoundError, match="requires probe"):
        train_distillation(path, teacher_override=fake_teacher())


def test_plan_and_old_config_compatibility(tmp_path):
    original = yaml.safe_load(Path("configs/analyses/knowledge_distillation.yaml").read_text())
    original["output_dir"] = str(tmp_path)
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(original))
    plan = build_distillation_plan(source)
    assert json.loads((plan.parent / "summary.json").read_text())["runs"] == 70
    config = next((tmp_path / "generated/configs").glob("*mgd*.yaml"))
    experiment, kd = load_distillation_config(config)
    assert kd.teacher.family in {"dinov2", "dinov3"}
    assert replace(experiment, name="other").model.name == experiment.model.name


def prepare_fake_probe(path, teacher):
    from dataclasses import asdict

    from fm2edge.config import load_config

    config, kd = load_distillation_config(path)
    probe_raw = config.to_dict()
    probe_raw["model"] = asdict(kd.teacher)
    probe_raw["name"] = "probe"
    directory = Path(config.output_dir) / "probe/fold_01"
    directory.mkdir(parents=True)
    probe_path = directory / "config.yaml"
    probe_path.write_text(yaml.safe_dump(probe_raw))
    (directory / "split.yaml").write_bytes(Path(config.data.split).read_bytes())
    (directory / "manifest.csv").write_bytes(Path(config.data.manifest).read_bytes())
    checkpoint = directory / "checkpoints/best.pt"
    checkpoint.parent.mkdir()
    torch.save(
        {"model": teacher.checkpoint_state_dict(), "model_metadata": teacher.checkpoint_metadata()},
        checkpoint,
    )
    assert load_config(probe_path).model.family == "dinov2"
    content = yaml.safe_load(path.read_text())
    content["distillation"].update(probe_config=str(probe_path), probe_checkpoint=str(checkpoint))
    path.write_text(yaml.safe_dump(content))
    return probe_path


def test_heteroakd_end_to_end_and_subset_rejection(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr("fm2edge.engine.distillation.build_student", lambda *args: TinyStudent())
    path = make_config(tmp_path, "heteroakd")
    teacher = fake_teacher()
    probe_path = prepare_fake_probe(path, teacher)
    output = train_distillation(path, teacher_override=teacher)
    assert (output / "completed.json").exists()
    config, kd = load_distillation_config(path)
    validate_probe(config, kd, teacher)
    split = Path(config.data.split)
    raw = yaml.safe_load(split.read_text())
    raw["train_sample_ids"] = ["train-0", "train-1"]
    split.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="subset mismatch"):
        validate_probe(config, kd, teacher)
    assert probe_path.exists()


@pytest.mark.parametrize("method", ["mgd", "gkd_cnn_source_only"])
def test_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, method):
    import fm2edge.engine.distillation as engine

    torch.set_num_threads(1)
    monkeypatch.setattr(engine, "build_student", lambda *args: TinyStudent())
    reference_dir, resumed_dir = tmp_path / "reference", tmp_path / "resumed"
    reference_dir.mkdir()
    resumed_dir.mkdir()
    reference_path = make_config(reference_dir, method)
    resumed_path = make_config(resumed_dir, method)
    for path in (reference_path, resumed_path):
        raw = yaml.safe_load(path.read_text())
        raw["train"]["epochs"] = 2
        path.write_text(yaml.safe_dump(raw))
    torch.manual_seed(18)
    teacher = fake_teacher()
    reference = train_distillation(reference_path, teacher_override=teacher)
    original = engine._save_checkpoint

    def interrupt_after_epoch(state, path):
        original(state, path)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(engine, "_save_checkpoint", interrupt_after_epoch)
    with pytest.raises(RuntimeError, match="simulated"):
        train_distillation(resumed_path, teacher_override=teacher)
    monkeypatch.setattr(engine, "_save_checkpoint", original)
    resumed = train_distillation(resumed_path, teacher_override=teacher)
    full_state = torch.load(reference / "checkpoints/last.pt", weights_only=True)
    resumed_state = torch.load(resumed / "checkpoints/last.pt", weights_only=True)
    for key in full_state["student"]:
        assert torch.equal(full_state["student"][key], resumed_state["student"][key])
    for key in full_state["adapter"]:
        assert torch.equal(full_state["adapter"][key], resumed_state["adapter"][key])


def test_study_aggregation_without_baseline(tmp_path, monkeypatch):
    from fm2edge.analysis.distillation_study import analyze_distillation

    monkeypatch.setattr("fm2edge.engine.distillation.build_student", lambda *args: TinyStudent())
    path = make_config(tmp_path, "mgd")
    output = train_distillation(path, teacher_override=fake_teacher())
    plan = tmp_path / "plan/experiment_plan.csv"
    plan.parent.mkdir()
    import csv

    row = {
        "run_id": "x",
        "teacher": "dinov2",
        "student": "mobilenet_v3_lraspp",
        "method": "mgd",
        "seed": 42,
        "fold": 1,
        "config_path": str(path),
        "result_dir": str(output),
    }
    with plan.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    analysis = analyze_distillation(plan, tmp_path / "results")
    assert (analysis / "machine_heatmap.png").exists()
    assert "no_matching_baseline" in (analysis / "paired_baseline_deltas.csv").read_text()
