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
    assert bundle.feature_names == tuple(sorted(rows[0].features))
    path = tmp_path / "model.joblib"
    save_model_bundle(bundle, path)
    assert load_model_bundle(path, expected_feature_names=bundle.feature_names).feature_schema_version == bundle.feature_schema_version
    with pytest.raises(ValueError, match="feature schema"):
        load_model_bundle(path, expected_feature_names=("rms",))
