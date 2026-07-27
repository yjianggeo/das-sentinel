import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from das_sentinel.classifiers.supervised import (
    TrainingGateError,
    load_model_bundle,
    save_model_bundle,
    train_model,
)
from das_sentinel.domain import Label


def _write_sentinel(path: str) -> None:
    Path(path).write_text("executed", encoding="utf-8")


class _MaliciousPickle:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return _write_sentinel, (str(self.sentinel),)


@dataclass(frozen=True)
class TrainingRow:
    event_id: str
    label: Label
    features: dict[str, float]
    day: str


def _rows(per_label: int) -> list[TrainingRow]:
    labels = (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)
    rows = []
    for label_index, label in enumerate(labels):
        for index in range(per_label):
            rows.append(TrainingRow(
                event_id=f"{label}-{index}", label=label,
                features={"duration_s": label_index * 100.0 + index, "rms": label_index + index / 100.0},
                day=f"day-{index % 5}",
            ))
    return rows


def test_training_rejects_less_than_twenty_per_class():
    rows = _rows(15)

    with pytest.raises(TrainingGateError, match="20 reviewed events"):
        train_model(rows, groups=[row.day for row in rows])


def test_training_rejects_unknown_label():
    rows = _rows(20)
    rows[0] = TrainingRow("unknown", Label.UNKNOWN, rows[0].features, rows[0].day)

    with pytest.raises(TrainingGateError, match="four reviewed labels"):
        train_model(rows, groups=[row.day for row in rows])


def test_training_rejects_nonfinite_features():
    rows = _rows(20)
    rows[0] = TrainingRow(rows[0].event_id, rows[0].label, {"duration_s": float("nan"), "rms": 1.0}, rows[0].day)

    with pytest.raises(TrainingGateError, match="finite"):
        train_model(rows, groups=[row.day for row in rows])


def test_publishable_report_meets_all_gates_and_bundle_schema_is_checked(tmp_path: Path):
    rows = _rows(25)

    bundle, report = train_model(rows, groups=[row.day for row in rows])

    assert report.macro_f1 >= 0.70
    assert min(report.recall_by_label.values()) >= 0.50
    assert set(report.precision_by_label) == {Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE}
    assert report.confusion_matrix.shape == (4, 4)
    assert 0.0 <= report.rule_baseline_macro_f1 <= 1.0
    assert bundle.feature_names == tuple(sorted(rows[0].features))
    path = tmp_path / "model.joblib"
    save_model_bundle(bundle, path)
    assert load_model_bundle(path, expected_feature_names=bundle.feature_names).feature_schema_version == bundle.feature_schema_version
    with pytest.raises(ValueError, match="feature schema"):
        load_model_bundle(path, expected_feature_names=("rms",))


def test_load_rejects_pickle_without_executing_it(tmp_path: Path):
    path = tmp_path / "malicious.bundle"
    sentinel = tmp_path / "sentinel.txt"
    path.write_bytes(pickle.dumps(_MaliciousPickle(sentinel)))

    with pytest.raises((ValueError, TypeError)):
        load_model_bundle(path, expected_feature_names=("duration_s", "rms"))

    assert not sentinel.exists()


@pytest.mark.parametrize("member", ("metadata.json", "model.skops"))
def test_load_rejects_tampered_bundle_members(tmp_path: Path, member: str):
    rows = _rows(20)
    bundle, _ = train_model(rows, groups=[row.day for row in rows])
    path = tmp_path / "model.bundle"
    save_model_bundle(bundle, path)
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents[member] = b"{}" if member == "metadata.json" else b"not a skops model"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in contents.items():
            archive.writestr(name, value)

    with pytest.raises((ValueError, TypeError)):
        load_model_bundle(path, expected_feature_names=bundle.feature_names)


def test_load_rejects_tampered_sklearn_version(tmp_path: Path):
    rows = _rows(20)
    bundle, _ = train_model(rows, groups=[row.day for row in rows])
    path = tmp_path / "model.bundle"
    save_model_bundle(bundle, path)
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        model = archive.read("model.skops")
    metadata["sklearn_version"] = "0.0"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("model.skops", model)

    with pytest.raises(ValueError, match="sklearn version"):
        load_model_bundle(path, expected_feature_names=bundle.feature_names)


def test_snapshot_hash_is_stable_when_rows_and_groups_are_reordered():
    rows = _rows(20)
    groups = [row.day for row in rows]

    first, _ = train_model(rows, groups=groups)
    second, _ = train_model(list(reversed(rows)), groups=list(reversed(groups)))

    assert first.training_snapshot_hash == second.training_snapshot_hash
