from pathlib import Path

from fm2edge.data.records import SampleRecord, read_manifest, write_manifest
from fm2edge.data.splits import load_split, make_machine_folds, save_split


def test_manifest_roundtrip_and_machine_disjoint_splits(tmp_path: Path) -> None:
    records = [
        SampleRecord(
            sample_id=f"sample-{machine}",
            image_path=f"images/{machine}.png",
            mask_path=f"masks/{machine}.png",
            machine_id=machine,
            delay="90",
        )
        for machine in ("a", "b", "c", "d", "e")
    ]
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    assert read_manifest(manifest) == records

    splits = make_machine_folds(read_manifest(manifest), n_folds=5, seed=11)
    assert len(splits) == 5
    for split in splits:
        train = set(split.train_machines)
        val = set(split.val_machines)
        test = set(split.test_machines)
        assert not train & val
        assert not train & test
        assert not val & test

    path = tmp_path / "fold.yaml"
    save_split(splits[0], path)
    assert load_split(path) == splits[0]
