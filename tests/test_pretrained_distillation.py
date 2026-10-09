"""Phase5.5 tests; existing tests and trainers intentionally remain unchanged."""

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml
from test_distillation import fake_teacher, make_config

from fm2edge.analysis.distillation_study import write_csv
from fm2edge.analysis.pretrained_study import analyze_pretrained, build_pretrained_plan
from fm2edge.distillation_config import load_distillation_config
from fm2edge.engine.distillation import train_distillation
from fm2edge.engine.utils import file_sha256
from fm2edge.models.imagenet_conversion import (
    convert_imagenet,
    convert_state,
    feature_key,
    official_key,
    validate_converted,
)
from fm2edge.models.registry import build_student


def source_state(name):
    torch.set_num_threads(1)
    model = build_student(name, 2)
    return {official_key(name, k): v for k, v in model.state_dict().items() if feature_key(name, k)}


def fixture_weight(tmp_path, name):
    state, audit = convert_state(name, source_state(name))
    audit["source_sha256"] = "synthetic-not-official"
    path = tmp_path / (name + ".pt")
    torch.save({"model": state, "imagenet_import": audit}, path)
    audit["converted_sha256"] = file_sha256(path)
    path.with_suffix(".json").write_text(json.dumps(audit))
    return path


@pytest.mark.parametrize("name", ["mobilenet_v3_lraspp", "pidnet_s"])
def test_conversion_complete_and_heads_remain_seeded(tmp_path, name):
    source = source_state(name)
    state, audit = convert_state(
        name, {"state_dict": {"module." + k: v for k, v in source.items()}}
    )
    assert audit["feature_parameter_coverage"] == 1
    assert not audit["missing_feature_keys"]
    path = fixture_weight(tmp_path, name)
    state = torch.load(path, weights_only=True)["model"]
    torch.manual_seed(42)
    random = build_student(name, 2)
    torch.manual_seed(42)
    pretrained = build_student(name, 2, str(path))
    for key, value in pretrained.state_dict().items():
        if feature_key(name, key):
            assert torch.equal(value, state[key])
        else:
            assert torch.equal(value, random.state_dict()[key])
    for method in ("none", "mgd", "heteroakd", "gkd_cnn_source_only"):
        torch.manual_seed(42)
        actual = build_student(name, 2, str(path))
        assert all(
            torch.equal(v, actual.state_dict()[k]) for k, v in pretrained.state_dict().items()
        )
    corrupted = dict(source)
    corrupted.pop(next(iter(corrupted)))
    with pytest.raises(ValueError, match="missing"):
        convert_state(name, corrupted)
    corrupted = dict(source)
    corrupted[next(iter(corrupted))] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape"):
        convert_state(name, corrupted)


def test_se_mapping_and_pid_partial_audit_and_wrong_weights(tmp_path):
    state, audit = convert_state("mobilenet_v3_lraspp", source_state("mobilenet_v3_lraspp"))
    se = [k for k in state if ".reduce." in k]
    assert se and all(".fc1." in audit["mapping"][k] for k in se)
    source = source_state("pidnet_s")
    source = {k: v for k, v in source.items() if not k.startswith("pag3.")}
    _, audit = convert_state("pidnet_s", source)
    assert audit["missing_feature_modules"] == ["pag3"]
    with pytest.raises(ValueError, match="Segmentation checkpoint"):
        convert_state(
            "pidnet_s", {**source, "final_layer.conv2.weight": torch.zeros(19, 128, 1, 1)}
        )
    with pytest.raises(ValueError):
        convert_state("pidnet_s", source_state("mobilenet_v3_lraspp"))
    path = fixture_weight(tmp_path, "pidnet_s")
    validate_converted("pidnet_s", path)
    with pytest.raises(ValueError, match="identity"):
        validate_converted("mobilenet_v3_lraspp", path)
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        validate_converted("pidnet_s", path)
    raw = tmp_path / "source.pt"
    torch.save({"state_dict": source}, raw)
    with pytest.raises(ValueError, match="SHA"):
        convert_imagenet("pidnet_s", raw, tmp_path / "out.pt", "wrong")


def study_config(tmp_path):
    raw = yaml.safe_load(Path("configs/analyses/pretrained_distillation.yaml").read_text())
    raw["output_dir"] = str(tmp_path / "phase55")
    raw["student_weights"] = {n: str(fixture_weight(tmp_path, n)) for n in raw["students"]}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_plan_70_and_exact_reference_import(tmp_path):
    source = study_config(tmp_path)
    plan = build_pretrained_plan(source)
    assert json.loads((plan.parent / "summary.json").read_text())["runs"] == 70
    rows = list(csv.DictReader(plan.open()))
    reference = tmp_path / "phase5/plan/experiment_plan.csv"
    row = rows[0]
    raw = yaml.safe_load(Path(row["config_path"]).read_text())
    raw["model"]["pretrained"] = None
    raw["train"]["learning_rate"] = 0.0002
    reference_config = tmp_path / "old.yaml"
    reference_config.write_text(yaml.safe_dump(raw))
    write_csv([{**row, "config_path": str(reference_config)}], reference)
    before = reference_config.read_bytes()
    imported = build_pretrained_plan(source, phase5_plan=reference)
    new = next(csv.DictReader(imported.open()))
    config, _ = load_distillation_config(new["config_path"])
    assert config.train.learning_rate == 0.0002 and config.model.pretrained
    assert reference_config.read_bytes() == before


@pytest.mark.parametrize("name", ["mobilenet_v3_lraspp", "pidnet_s"])
@pytest.mark.parametrize("method", ["none", "mgd", "heteroakd", "gkd_cnn_source_only"])
@pytest.mark.parametrize("family", ["dinov2", "dinov3"])
def test_pretrained_real_students_fake_dino_e2e(tmp_path, name, method, family):
    torch.set_num_threads(1)
    path = make_config(tmp_path, method, family, "cached")
    weight = fixture_weight(tmp_path, name)
    raw = yaml.safe_load(path.read_text())
    raw["model"] = {"name": name, "pretrained": str(weight)}
    raw["data"]["image_size"] = [56, 56]
    path.write_text(yaml.safe_dump(raw))
    teacher = fake_teacher(family, "cached")
    if method == "heteroakd":
        config, kd = load_distillation_config(path)
        from dataclasses import asdict

        probe = {**config.to_dict(), "model": asdict(kd.teacher)}
        directory = tmp_path / "probe"
        directory.mkdir()
        probe_path = directory / "config.yaml"
        probe_path.write_text(yaml.safe_dump(probe))
        (directory / "split.yaml").write_bytes(Path(config.data.split).read_bytes())
        (directory / "manifest.csv").write_bytes(Path(config.data.manifest).read_bytes())
        checkpoint = directory / "checkpoints/best.pt"
        checkpoint.parent.mkdir()
        torch.save(
            {
                "model": teacher.checkpoint_state_dict(),
                "model_metadata": teacher.checkpoint_metadata(),
            },
            checkpoint,
        )
        raw["distillation"].update(probe_config=str(probe_path), probe_checkpoint=str(checkpoint))
        path.write_text(yaml.safe_dump(raw))
    output = train_distillation(path, evaluate_test=False, teacher_override=teacher)
    assert not (output / "completed.json").exists()
    train_distillation(path, teacher_override=teacher)
    assert (output / "predictions/random").is_dir()
    assert (output / "completed.json").is_file()
    model = build_student(name, 2)
    checkpoint = torch.load(output / "student.pt", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    before = (output / "history/epochs.csv").read_bytes()
    train_distillation(path, teacher_override=teacher)
    assert (output / "history/epochs.csv").read_bytes() == before


def write_artifact(path, initialization, method, iou, *, content="same", teacher_hash="teacher"):
    path.mkdir(parents=True)
    raw = {
        "data": {"num_classes": 2},
        "model": {
            "name": "mobilenet_v3_lraspp",
            "pretrained": "imagenet.pt" if initialization == "imagenet" else None,
        },
        "train": {"seed": 42, "epochs": 50},
        "distillation": {"method": method, "teacher": {"family": "dinov2", "head": "linear"}},
    }
    (path / "config.yaml").write_text(yaml.safe_dump(raw))
    (path / "split.yaml").write_text(
        yaml.safe_dump(
            {"fold": 1, "train_machines": ["a"], "val_machines": ["b"], "test_machines": ["c"]}
        )
    )
    (path / "manifest.csv").write_text("identical manifest")
    (path / "identity.json").write_text(
        json.dumps(
            {
                "dataset_content_sha256": content,
                "student_initial_sha256": initialization if initialization == "imagenet" else None,
                "teacher": {"weights_sha256": teacher_hash} if method != "none" else None,
                "implementation_sha256": {"engine": "same"},
            }
        )
    )
    (path / "completed.json").write_text("{}")
    (path / "metrics").mkdir()
    (path / "metrics/summary.json").write_text(json.dumps({"IoU": iou, "Dice": iou}))
    (path / "metrics/per_machine.csv").write_text(f"machine_id,IoU\nc,{iou}\n")
    return path


def test_four_way_deltas_and_no_history_fallback(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    old = tmp_path / "old"
    write_artifact(old / "gt", "random", "none", 0.8)
    write_artifact(old / "kd", "random", "mgd", 0.85)
    new = tmp_path / "new"
    gt = write_artifact(new / "gt", "imagenet", "none", 0.87)
    kd = write_artifact(new / "kd", "imagenet", "mgd", 0.9)
    plan = tmp_path / "plan/experiment_plan.csv"
    write_csv([{"result_dir": str(gt)}, {"result_dir": str(kd)}], plan)
    output = analyze_pretrained(plan, old)
    deltas = list(csv.DictReader((output / "paired_deltas.csv").open()))
    effect = {
        d["effect"]: float(d["IoU_delta"]) for d in deltas if d["initialization"] == "imagenet"
    }
    assert effect["pretraining_gt"] == pytest.approx(0.07)
    assert effect["kd_addition"] == pytest.approx(0.03)
    assert effect["kd_interaction"] == pytest.approx(-0.02)
    assert (output / "machine_heatmap.png").exists()
    analyze_pretrained(plan)
    assert "no_matching_control" in (output / "comparison_status.csv").read_text()
    identity = json.loads((old / "gt/identity.json").read_text())
    identity["dataset_content_sha256"] = "changed"
    (old / "gt/identity.json").write_text(json.dumps(identity))
    analyze_pretrained(plan, old)
    assert "dataset_content_sha256" in (output / "comparison_status.csv").read_text()
    deltas = list(csv.DictReader((output / "paired_deltas.csv").open()))
    assert not any(d["effect"] == "pretraining_gt" for d in deltas)


def test_teacher_mismatch_ambiguity_and_gkd_budget(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    new = tmp_path / "new"
    gt = write_artifact(new / "gt", "imagenet", "none", 0.8)
    kd = write_artifact(new / "kd", "imagenet", "gkd_cnn_source_only", 0.85)
    duplicate = write_artifact(new / "gt2", "imagenet", "none", 0.8)
    old = tmp_path / "old"
    write_artifact(old / "kd", "random", "gkd_cnn_source_only", 0.7, teacher_hash="different")
    plan = tmp_path / "plan/experiment_plan.csv"
    write_csv([{"result_dir": str(gt)}, {"result_dir": str(kd)}], plan)
    output = analyze_pretrained(plan, old)
    deltas = list(csv.DictReader((output / "paired_deltas.csv").open()))
    assert len(deltas) == 1 and deltas[0]["status"] == "budget_different"
    assert "teacher_or_kd_conditions" in (output / "comparison_status.csv").read_text()
    write_csv(
        [{"result_dir": str(gt)}, {"result_dir": str(kd)}, {"result_dir": str(duplicate)}], plan
    )
    analyze_pretrained(plan, old)
    assert "ambiguous_control" in (output / "comparison_status.csv").read_text()


def test_pretrained_interrupted_epoch_resume(tmp_path, monkeypatch):
    import fm2edge.engine.distillation as engine

    path = make_config(tmp_path, "mgd")
    raw = yaml.safe_load(path.read_text())
    raw["model"]["pretrained"] = str(fixture_weight(tmp_path, "mobilenet_v3_lraspp"))
    raw["data"]["image_size"] = [56, 56]
    raw["train"]["epochs"] = 2
    path.write_text(yaml.safe_dump(raw))
    original = engine._save_checkpoint

    def stop(state, target):
        original(state, target)
        raise RuntimeError("interrupt")

    monkeypatch.setattr(engine, "_save_checkpoint", stop)
    with pytest.raises(RuntimeError, match="interrupt"):
        train_distillation(path, teacher_override=fake_teacher())
    monkeypatch.setattr(engine, "_save_checkpoint", original)
    output = train_distillation(path, teacher_override=fake_teacher())
    assert (output / "completed.json").exists()
    history = list(csv.DictReader((output / "history/epochs.csv").open()))
    assert len(history) == 2
