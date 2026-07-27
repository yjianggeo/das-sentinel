from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from das_sentinel.classifiers.similarity import FeatureVectorizer
from das_sentinel.domain import Label

_LABELS = (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)
_SEED = 20260726


class TrainingGateError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingReport:
    macro_f1: float
    recall_by_label: dict[Label, float]
    reviewed_count_by_label: dict[Label, int]


@dataclass(frozen=True)
class ModelBundle:
    model: RandomForestClassifier
    feature_names: tuple[str, ...]
    labels: tuple[Label, ...]
    training_snapshot_hash: str
    sklearn_version: str
    report: TrainingReport
    feature_schema_version: str

    def predict_probabilities(self, features: Mapping[str, Any]) -> dict[Label, float]:
        try:
            vector = FeatureVectorizer(self.feature_names).transform_one(features).reshape(1, -1)
        except ValueError as exc:
            raise TrainingGateError("prediction features do not match the model schema") from exc
        probabilities = self.model.predict_proba(vector)[0]
        if probabilities.shape != (len(self.model.classes_),) or not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise TrainingGateError("model returned invalid probabilities")
        known = {Label(label): float(value) for label, value in zip(self.model.classes_, probabilities, strict=True)}
        return {label: known.get(label, 0.0) for label in self.labels}


def _label(value: Any) -> Label:
    try:
        label = Label(value)
    except ValueError as exc:
        raise TrainingGateError("training rows must use the four reviewed labels") from exc
    if label not in _LABELS:
        raise TrainingGateError("training rows must use the four reviewed labels")
    return label


def _snapshot_hash(rows: Sequence[Any], feature_names: tuple[str, ...]) -> str:
    snapshot = [
        {
            "event_id": str(row.event_id),
            "label": str(_label(row.label)),
            "features": {name: float(row.features.get(name, 0.0)) for name in feature_names},
        }
        for row in rows
    ]
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def train_model(rows: Iterable[Any], groups: Sequence[Any]) -> tuple[ModelBundle, TrainingReport]:
    reviewed_rows = tuple(rows)
    if not reviewed_rows:
        raise TrainingGateError("at least one reviewed event is required")
    if len(groups) != len(reviewed_rows):
        raise TrainingGateError("groups must match reviewed event count")
    labels = np.asarray([_label(row.label) for row in reviewed_rows])
    counts = Counter(Label(label) for label in labels)
    if any(counts[label] < 20 for label in _LABELS):
        raise TrainingGateError("each supervised label requires 20 reviewed events")
    try:
        feature_sets = [{name for name in row.features if name not in {"site_id", "cable_type", "device_config"}} for row in reviewed_rows]
    except AttributeError as exc:
        raise TrainingGateError("training rows must contain feature mappings") from exc
    if any(features != feature_sets[0] for features in feature_sets[1:]):
        raise TrainingGateError("reviewed events must share one feature schema")
    feature_names = tuple(sorted(feature_sets[0]))
    try:
        vectorizer = FeatureVectorizer(feature_names)
    except ValueError as exc:
        raise TrainingGateError("training features do not match the approved schema") from exc
    for row in reviewed_rows:
        for name in feature_names:
            value = row.features.get(name)
            if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(float(value)):
                raise TrainingGateError("training features must be finite numeric values")
    try:
        features = vectorizer.transform(row.features for row in reviewed_rows)
    except ValueError as exc:
        raise TrainingGateError("training features do not match the approved schema") from exc
    if len(set(groups)) < 5:
        raise TrainingGateError("five distinct review groups are required")

    classifier = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=_SEED, n_jobs=1,
    )
    predicted = np.empty(labels.shape, dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_SEED)
    try:
        for train_index, test_index in splitter.split(features, labels, groups):
            predicted[test_index] = clone(classifier).fit(features[train_index], labels[train_index]).predict(features[test_index])
    except ValueError as exc:
        raise TrainingGateError("review groups cannot support five-fold stratified validation") from exc
    recalls = recall_score(labels, predicted, labels=list(_LABELS), average=None, zero_division=0.0)
    report = TrainingReport(
        macro_f1=float(f1_score(labels, predicted, labels=list(_LABELS), average="macro", zero_division=0.0)),
        recall_by_label=dict(zip(_LABELS, (float(value) for value in recalls), strict=True)),
        reviewed_count_by_label={label: counts[label] for label in _LABELS},
    )
    trained = classifier.fit(features, labels)
    return ModelBundle(
        model=trained,
        feature_names=feature_names,
        labels=_LABELS,
        training_snapshot_hash=_snapshot_hash(reviewed_rows, feature_names),
        sklearn_version=sklearn.__version__,
        report=report,
        feature_schema_version=vectorizer.feature_schema_version,
    ), report


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> None:
    joblib.dump(bundle, path)


def load_model_bundle(path: str | Path, expected_feature_names: Sequence[str]) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError("invalid model bundle")
    expected = tuple(expected_feature_names)
    if bundle.feature_names != expected or bundle.feature_schema_version != FeatureVectorizer.feature_schema_version:
        raise ValueError("feature schema does not match model bundle")
    return bundle
