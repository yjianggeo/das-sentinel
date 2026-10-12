import json
import pickle
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from das_sentinel.classifiers.supervised import (
    TrainingGateError,
    load_model_bundle,
    save_model_bundle,
    train_model,
)
from das_sentinel.domain import Label
from sklearn.ensemble import RandomForestClassifier
from skops.io import dumps as skops_dumps


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


@pytest.fixture
def trained_bundle_bytes(tmp_path: Path):
    rows = _rows(20)
    bundle, _ = train_model(rows, groups=[row.day for row in rows])
    path = tmp_path / "model.bundle"
    save_model_bundle(bundle, path)
    return path.read_bytes(), bundle.feature_names


def test_load_model_bundle_bytes_verifies_expected_digest(
    trained_bundle_bytes,
):
    from das_sentinel.classifiers import supervised

    raw, feature_names = trained_bundle_bytes
    digest = sha256(raw).hexdigest()

    loaded = supervised.load_model_bundle_bytes(raw, feature_names, digest)

    assert loaded.feature_names == tuple(feature_names)
    with pytest.raises(ValueError, match="SHA-256"):
        supervised.load_model_bundle_bytes(raw, feature_names, "0" * 64)


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


def test_training_uses_feasible_whole_group_folds_when_greedy_stratification_fails():
    distributions = [
        [6, 6, 2, 1, 4, 5, 0, 6],
        [0, 2, 0, 12, 4, 3, 1, 8],
        [0, 6, 2, 3, 7, 7, 4, 1],
        [17, 2, 3, 0, 3, 0, 4, 1],
    ]
    rows = _rows(30)
    groups = [f"day-{day}" for counts in distributions
              for day, count in enumerate(counts) for _ in range(count)]

    bundle, report = train_model(rows, groups)

    assert report.confusion_matrix.sum() == 120
    assert list(report.confusion_matrix.sum(axis=1)) == [30, 30, 30, 30]
    assert np.isfinite(report.macro_f1)
    assert sum(bundle.predict_probabilities(rows[0].features).values()) == pytest.approx(1.0)


def test_training_rejects_class_coverage_that_cannot_be_grouped():
    rows = _rows(20)
    groups = ["blast-day" if row.label == Label.BLAST else row.day for row in rows]

    with pytest.raises(TrainingGateError, match="class.*groups"):
        train_model(rows, groups)


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
    digest = save_model_bundle(bundle, path)
    assert load_model_bundle(path, expected_feature_names=bundle.feature_names, expected_sha256=digest).feature_schema_version == bundle.feature_schema_version
    with pytest.raises(ValueError, match="feature schema"):
        load_model_bundle(path, expected_feature_names=("rms",), expected_sha256=digest)


def test_load_rejects_pickle_without_executing_it(tmp_path: Path):
    path = tmp_path / "malicious.bundle"
    sentinel = tmp_path / "sentinel.txt"
    path.write_bytes(pickle.dumps(_MaliciousPickle(sentinel)))

    with pytest.raises((ValueError, TypeError)):
        load_model_bundle(path, expected_feature_names=("duration_s", "rms"), expected_sha256="0" * 64)

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
        load_model_bundle(path, expected_feature_names=bundle.feature_names, expected_sha256=sha256(path.read_bytes()).hexdigest())


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
        load_model_bundle(path, expected_feature_names=bundle.feature_names, expected_sha256=sha256(path.read_bytes()).hexdigest())


def test_snapshot_hash_is_stable_when_rows_and_groups_are_reordered():
    rows = _rows(20)
    groups = [row.day for row in rows]

    first, _ = train_model(rows, groups=groups)
    second, _ = train_model(list(reversed(rows)), groups=list(reversed(groups)))

    assert first.training_snapshot_hash == second.training_snapshot_hash


def _replace_model(path: Path, model_bytes: bytes) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("metadata.json")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", metadata)
        archive.writestr("model.skops", model_bytes)
    return sha256(path.read_bytes()).hexdigest()


def _approved_model(feature_count: int) -> RandomForestClassifier:
    rows = np.arange(16 * feature_count, dtype=float).reshape(16, feature_count)
    labels = np.array(["earthquake", "blast", "traffic", "noise"] * 4)
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=20260726, n_jobs=1).fit(rows, labels)


def test_load_rejects_legitimate_model_replacement_before_parsing(tmp_path: Path):
    rows = _rows(20)
    bundle, _ = train_model(rows, groups=[row.day for row in rows])
    path = tmp_path / "model.bundle"
    digest = save_model_bundle(bundle, path)
    _replace_model(path, skops_dumps(_approved_model(2)))

    with pytest.raises(ValueError, match="SHA-256"):
        load_model_bundle(path, expected_feature_names=bundle.feature_names, expected_sha256=digest)


def test_load_rejects_replacement_with_wrong_feature_dimension_after_hash_check(tmp_path: Path):
    rows = _rows(20)
    bundle, _ = train_model(rows, groups=[row.day for row in rows])
    path = tmp_path / "model.bundle"
    save_model_bundle(bundle, path)
    replacement_digest = _replace_model(path, skops_dumps(_approved_model(3)))

    with pytest.raises(ValueError, match="feature count"):
        load_model_bundle(path, expected_feature_names=bundle.feature_names, expected_sha256=replacement_digest)


def test_load_rejects_inner_skops_zip_bomb_before_skops_parsing(tmp_path: Path):
    rows = _rows(20)
    bundle, _ = train_model(rows, groups=[row.day for row in rows])
    path = tmp_path / "model.bundle"
    save_model_bundle(bundle, path)
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("schema.json", b"0" * (1024 * 1024))
    digest = _replace_model(path, payload.getvalue())

    with pytest.raises(ValueError, match="skops archive"):
        load_model_bundle(path, expected_feature_names=bundle.feature_names, expected_sha256=digest)


def test_rule_profile_changes_baseline_and_snapshot_binding():
    rows = [
        TrainingRow(row.event_id, row.label, row.features | {"apparent_velocity_mps": 10.0 if row.label is Label.TRAFFIC else 0.0, "continuity": 0.5}, row.day)
        for row in _rows(20)
    ]
    groups = [row.day for row in rows]

    default_bundle, default_report = train_model(rows, groups=groups)
    custom_bundle, custom_report = train_model(
        rows, groups=groups, rule_profile={"traffic_velocity_mps": (100.0, 200.0)}, rule_profile_version="traffic-off",
    )

    assert default_report.rule_baseline_macro_f1 != custom_report.rule_baseline_macro_f1
    assert default_bundle.training_snapshot_hash != custom_bundle.training_snapshot_hash
    assert custom_report.rule_profile == {"traffic_velocity_mps": [100.0, 200.0]}
    assert custom_report.rule_profile_version == "traffic-off"
