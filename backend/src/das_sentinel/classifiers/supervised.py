from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from io import BytesIO
from numbers import Real
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from skops.io import dumps as skops_dumps
from skops.io import get_untrusted_types
from skops.io import loads as skops_loads

from das_sentinel.classifiers.fusion import fuse_classification
from das_sentinel.classifiers.group_folds import make_group_folds
from das_sentinel.classifiers.rules import score_rules
from das_sentinel.classifiers.similarity import FeatureVectorizer
from das_sentinel.domain import Label

_LABELS = (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)
_SEED = 20260726
_MEMBERS = frozenset({"metadata.json", "model.skops"})
_MAX_BUNDLE_BYTES = 64 * 1024 * 1024
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_MODEL_BYTES = 48 * 1024 * 1024
_MAX_SKOPS_MEMBERS = 4096
_MAX_SKOPS_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_SKOPS_TOTAL_BYTES = 48 * 1024 * 1024
_MAX_SKOPS_COMPRESSION_RATIO = 100


class TrainingGateError(ValueError):
    pass


@dataclass(frozen=True)
class TrainingReport:
    macro_f1: float
    recall_by_label: dict[Label, float]
    precision_by_label: dict[Label, float]
    confusion_matrix: np.ndarray
    rule_baseline_macro_f1: float
    rule_profile_version: str
    rule_profile_hash: str
    rule_profile: dict[str, Any]
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
        except (TypeError, ValueError) as exc:
            raise TrainingGateError("prediction features do not match the model schema") from exc
        probabilities = np.asarray(self.model.predict_proba(vector)[0], dtype=np.float64)
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


def _finite_number(value: Any) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not np.isfinite(float(value)):
        raise TrainingGateError("training features must be finite numeric values")
    return float(value)


def _canonical_rule_profile(rule_profile: Mapping[str, Any] | None, rule_profile_version: str) -> tuple[dict[str, Any], str, str]:
    if not isinstance(rule_profile_version, str) or not rule_profile_version:
        raise TrainingGateError("rule profile version must be non-empty")
    if rule_profile is not None and not isinstance(rule_profile, Mapping):
        raise TrainingGateError("rule profile must be a mapping")
    try:
        encoded = json.dumps(dict(rule_profile or {}), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TrainingGateError("rule profile must be canonical JSON data") from exc
    return json.loads(encoded), encoded, sha256(encoded.encode()).hexdigest()


def _snapshot_hash(rows: Sequence[Any], groups: Sequence[Any], feature_names: tuple[str, ...], rule_profile_version: str, canonical_rule_profile: str) -> str:
    records = [
        {
            "event_id": str(row.event_id),
            "label": str(_label(row.label)),
            "group": str(group),
            "features": [[name, _finite_number(row.features[name])] for name in feature_names],
        }
        for row, group in zip(rows, groups, strict=True)
    ]
    records.sort(key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":")))
    payload = {"records": records, "rule_profile_version": rule_profile_version, "rule_profile": json.loads(canonical_rule_profile)}
    return sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _metric_map(values: np.ndarray) -> dict[Label, float]:
    return dict(zip(_LABELS, (float(value) for value in values), strict=True))


def train_model(rows: Iterable[Any], groups: Sequence[Any], *, rule_profile: Mapping[str, Any] | None = None, rule_profile_version: str = "default") -> tuple[ModelBundle, TrainingReport]:
    reviewed_rows = tuple(rows)
    groups = tuple(groups)
    if not reviewed_rows:
        raise TrainingGateError("at least one reviewed event is required")
    if len(groups) != len(reviewed_rows):
        raise TrainingGateError("groups must match reviewed event count")
    profile, canonical_profile, profile_hash = _canonical_rule_profile(rule_profile, rule_profile_version)
    labels = np.asarray([_label(row.label) for row in reviewed_rows])
    counts = Counter(Label(label) for label in labels)
    if any(counts[label] < 20 for label in _LABELS):
        raise TrainingGateError("each supervised label requires 20 reviewed events")
    try:
        feature_sets = [{name for name in row.features if name not in {"site_id", "cable_type", "device_config"}} for row in reviewed_rows]
    except AttributeError as exc:
        raise TrainingGateError("training rows must contain feature mappings") from exc
    if any(feature_set != feature_sets[0] for feature_set in feature_sets[1:]):
        raise TrainingGateError("reviewed events must share one feature schema")
    feature_names = tuple(sorted(feature_sets[0]))
    try:
        vectorizer = FeatureVectorizer(feature_names)
    except ValueError as exc:
        raise TrainingGateError("training features do not match the approved schema") from exc
    for row in reviewed_rows:
        for name in feature_names:
            _finite_number(row.features.get(name))
    try:
        features = vectorizer.transform(row.features for row in reviewed_rows)
    except (TypeError, ValueError) as exc:
        raise TrainingGateError("training features do not match the approved schema") from exc
    if len(set(groups)) < 5:
        raise TrainingGateError("five distinct review groups are required")

    classifier = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=_SEED, n_jobs=1)
    predicted = np.empty(labels.shape, dtype=object)
    try:
        splits = make_group_folds(labels, groups, n_splits=5, seed=_SEED)
    except ValueError as exc:
        raise TrainingGateError(str(exc)) from exc
    for train_index, test_index in splits:
        if set(labels[train_index]) != set(_LABELS) or set(labels[test_index]) != set(_LABELS):
            raise TrainingGateError("each validation fold must cover the four reviewed labels")
        if set(np.asarray(groups)[train_index]) & set(np.asarray(groups)[test_index]):
            raise TrainingGateError("validation train and test groups must not overlap")
        predicted[test_index] = clone(classifier).fit(features[train_index], labels[train_index]).predict(features[test_index])

    precision = precision_score(labels, predicted, labels=list(_LABELS), average=None, zero_division=0.0)
    recall = recall_score(labels, predicted, labels=list(_LABELS), average=None, zero_division=0.0)
    rules_only = np.asarray([fuse_classification(score_rules(row.features, profile).scores, None, ()).label for row in reviewed_rows])
    report = TrainingReport(
        macro_f1=float(f1_score(labels, predicted, labels=list(_LABELS), average="macro", zero_division=0.0)),
        recall_by_label=_metric_map(recall),
        precision_by_label=_metric_map(precision),
        confusion_matrix=confusion_matrix(labels, predicted, labels=list(_LABELS)),
        rule_baseline_macro_f1=float(f1_score(labels, rules_only, labels=list(_LABELS), average="macro", zero_division=0.0)),
        rule_profile_version=rule_profile_version,
        rule_profile_hash=profile_hash,
        rule_profile=profile,
        reviewed_count_by_label={label: counts[label] for label in _LABELS},
    )
    trained = classifier.fit(features, labels)
    return ModelBundle(
        model=trained,
        feature_names=feature_names,
        labels=_LABELS,
        training_snapshot_hash=_snapshot_hash(reviewed_rows, groups, feature_names, rule_profile_version, canonical_profile),
        sklearn_version=sklearn.__version__,
        report=report,
        feature_schema_version=vectorizer.feature_schema_version,
    ), report


def _metadata(bundle: ModelBundle) -> bytes:
    report = bundle.report
    payload = {
        "feature_schema_version": bundle.feature_schema_version,
        "feature_names": list(bundle.feature_names),
        "labels": [str(label) for label in bundle.labels],
        "training_snapshot_hash": bundle.training_snapshot_hash,
        "sklearn_version": bundle.sklearn_version,
        "rule_profile_version": report.rule_profile_version,
        "rule_profile_hash": report.rule_profile_hash,
        "rule_profile": report.rule_profile,
        "model": {"type": "RandomForestClassifier", "n_estimators": 300, "class_weight": "balanced", "random_state": _SEED, "n_jobs": 1},
        "report": {
            "macro_f1": report.macro_f1,
            "recall_by_label": {str(label): value for label, value in report.recall_by_label.items()},
            "precision_by_label": {str(label): value for label, value in report.precision_by_label.items()},
            "confusion_matrix": report.confusion_matrix.tolist(),
            "rule_baseline_macro_f1": report.rule_baseline_macro_f1,
            "rule_profile_version": report.rule_profile_version,
            "rule_profile_hash": report.rule_profile_hash,
            "rule_profile": report.rule_profile,
            "reviewed_count_by_label": {str(label): value for label, value in report.reviewed_count_by_label.items()},
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save_model_bundle(bundle: ModelBundle, path: str | Path) -> str:
    if type(bundle.model) is not RandomForestClassifier:
        raise TypeError("model bundle must contain a RandomForestClassifier")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", _metadata(bundle))
        archive.writestr("model.skops", skops_dumps(bundle.model))
    return sha256(Path(path).read_bytes()).hexdigest()


def _read_bundle(raw_bundle: bytes) -> tuple[dict[str, Any], bytes]:
    try:
        with ZipFile(BytesIO(raw_bundle)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(infos) != len(_MEMBERS) or set(names) != _MEMBERS or len(set(names)) != len(names):
                raise ValueError("model bundle has invalid members")
            if any(
                info.is_dir()
                or info.file_size > (_MAX_METADATA_BYTES if info.filename == "metadata.json" else _MAX_MODEL_BYTES)
                for info in infos
            ):
                raise ValueError("model bundle member is too large")
            metadata_bytes = archive.read("metadata.json")
            model_bytes = archive.read("model.skops")
    except (OSError, BadZipFile) as exc:
        raise ValueError("model bundle must be a valid ZIP archive") from exc
    if len(metadata_bytes) > _MAX_METADATA_BYTES or len(model_bytes) > _MAX_MODEL_BYTES:
        raise ValueError("model bundle member is too large")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"), object_pairs_hook=_json_object, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model bundle metadata is invalid JSON") from exc
    if not isinstance(metadata, dict):
        raise TypeError("model bundle metadata must be an object")
    return metadata, model_bytes


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("model bundle metadata has duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"model bundle metadata has invalid constant: {value}")


def _report_from_metadata(value: Any) -> TrainingReport:
    if not isinstance(value, dict) or set(value) != {"macro_f1", "recall_by_label", "precision_by_label", "confusion_matrix", "rule_baseline_macro_f1", "rule_profile_version", "rule_profile_hash", "rule_profile", "reviewed_count_by_label"}:
        raise ValueError("model bundle report schema is invalid")
    def labels_map(name: str, integer: bool = False) -> dict[Label, float] | dict[Label, int]:
        mapping = value[name]
        if not isinstance(mapping, dict) or set(mapping) != {str(label) for label in _LABELS}:
            raise ValueError("model bundle report labels are invalid")
        result: dict[Label, float] | dict[Label, int] = {}
        for label in _LABELS:
            number = mapping[str(label)]
            if integer:
                if type(number) is not int or number < 20:
                    raise ValueError("model bundle report counts are invalid")
            elif not isinstance(number, Real) or not np.isfinite(number) or not 0.0 <= float(number) <= 1.0:
                raise ValueError("model bundle report metrics are invalid")
            result[label] = number
        return result
    matrix = np.asarray(value["confusion_matrix"])
    if matrix.shape != (4, 4) or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError("model bundle confusion matrix is invalid")
    metrics = (value["macro_f1"], value["rule_baseline_macro_f1"])
    if any(not isinstance(item, Real) or not np.isfinite(item) or not 0.0 <= float(item) <= 1.0 for item in metrics):
        raise ValueError("model bundle report metrics are invalid")
    if not isinstance(value["rule_profile_version"], str) or not value["rule_profile_version"] or not isinstance(value["rule_profile_hash"], str) or len(value["rule_profile_hash"]) != 64:
        raise ValueError("model bundle rule profile is invalid")
    profile, _, profile_hash = _canonical_rule_profile(value["rule_profile"], value["rule_profile_version"])
    if profile_hash != value["rule_profile_hash"]:
        raise ValueError("model bundle rule profile binding is invalid")
    return TrainingReport(float(metrics[0]), labels_map("recall_by_label"), labels_map("precision_by_label"), matrix, float(metrics[1]), value["rule_profile_version"], value["rule_profile_hash"], profile, labels_map("reviewed_count_by_label", integer=True))


def _read_bundle_path_bytes(path: str | Path) -> bytes:
    try:
        bundle_path = Path(path)
        if bundle_path.stat().st_size > _MAX_BUNDLE_BYTES:
            raise ValueError("model bundle is too large")
        raw_bundle = bundle_path.read_bytes()
    except OSError as exc:
        raise ValueError("model bundle cannot be read") from exc
    return raw_bundle


def _validate_skops_archive(model_bytes: bytes) -> None:
    try:
        with ZipFile(BytesIO(model_bytes)) as archive:
            infos = archive.infolist()
    except BadZipFile as exc:
        raise ValueError("model bundle skops archive is invalid") from exc
    names = [info.filename for info in infos]
    if not infos or len(infos) > _MAX_SKOPS_MEMBERS or len(names) != len(set(names)):
        raise ValueError("model bundle skops archive members are invalid")
    total_size = 0
    for info in infos:
        if info.is_dir() or info.flag_bits & 1 or info.file_size > _MAX_SKOPS_MEMBER_BYTES:
            raise ValueError("model bundle skops archive member is invalid")
        total_size += info.file_size
        if total_size > _MAX_SKOPS_TOTAL_BYTES or (info.compress_size and info.file_size / info.compress_size > _MAX_SKOPS_COMPRESSION_RATIO) or (not info.compress_size and info.file_size):
            raise ValueError("model bundle skops archive exceeds safety limits")


def _load_verified_components(
    metadata: dict[str, Any],
    model_bytes: bytes,
    expected_feature_names: Sequence[str],
) -> ModelBundle:
    expected = tuple(expected_feature_names)
    if set(metadata) != {"feature_schema_version", "feature_names", "labels", "training_snapshot_hash", "sklearn_version", "rule_profile_version", "rule_profile_hash", "rule_profile", "model", "report"}:
        raise ValueError("model bundle metadata schema is invalid")
    if metadata["feature_schema_version"] != FeatureVectorizer.feature_schema_version or metadata["feature_names"] != list(expected):
        raise ValueError("feature schema does not match model bundle")
    try:
        vectorizer = FeatureVectorizer(expected)
    except ValueError as exc:
        raise ValueError("feature schema does not match model bundle") from exc
    if metadata["labels"] != [str(label) for label in _LABELS] or metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("model bundle labels or sklearn version are invalid")
    if not isinstance(metadata["training_snapshot_hash"], str) or len(metadata["training_snapshot_hash"]) != 64 or any(char not in "0123456789abcdef" for char in metadata["training_snapshot_hash"]):
        raise ValueError("model bundle snapshot hash is invalid")
    if metadata["model"] != {"type": "RandomForestClassifier", "n_estimators": 300, "class_weight": "balanced", "random_state": _SEED, "n_jobs": 1}:
        raise ValueError("model bundle model declaration is invalid")
    report = _report_from_metadata(metadata["report"])
    profile, _, profile_hash = _canonical_rule_profile(metadata["rule_profile"], metadata["rule_profile_version"])
    if profile_hash != metadata["rule_profile_hash"] or metadata["rule_profile_version"] != report.rule_profile_version or metadata["rule_profile_hash"] != report.rule_profile_hash or profile != report.rule_profile:
        raise ValueError("model bundle rule profile binding is invalid")
    try:
        if get_untrusted_types(data=model_bytes):
            raise ValueError("model bundle contains untrusted model types")
        model = skops_loads(model_bytes)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("model bundle model payload is invalid") from exc
    if type(model) is not RandomForestClassifier or tuple(map(str, model.classes_)) != tuple(sorted(map(str, _LABELS))):
        raise TypeError("model bundle does not contain the approved classifier")
    params = model.get_params(deep=False)
    if (params["n_estimators"], params["class_weight"], params["random_state"], params["n_jobs"]) != (300, "balanced", _SEED, 1):
        raise ValueError("model bundle classifier parameters are invalid")
    if not all(hasattr(model, name) for name in ("n_features_in_", "estimators_", "classes_", "n_classes_", "n_outputs_")) or model.n_features_in_ != len(expected) or len(model.estimators_) != 300 or model.n_outputs_ != 1:
        raise ValueError("model bundle classifier feature count or fitted state is invalid")
    return ModelBundle(model, expected, _LABELS, metadata["training_snapshot_hash"], sklearn.__version__, report, vectorizer.feature_schema_version)


def load_model_bundle_bytes(
    raw: bytes,
    expected_feature_names: Sequence[str],
    expected_sha256: str,
) -> ModelBundle:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in expected_sha256
        )
    ):
        raise ValueError("expected SHA-256 must be 64 hexadecimal characters")
    if len(raw) > _MAX_BUNDLE_BYTES:
        raise ValueError("model bundle is too large")
    if not compare_digest(sha256(raw).hexdigest(), expected_sha256.lower()):
        raise ValueError("model bundle SHA-256 does not match expected digest")
    metadata, model_bytes = _read_bundle(raw)
    _validate_skops_archive(model_bytes)
    return _load_verified_components(
        metadata,
        model_bytes,
        expected_feature_names,
    )


def load_model_bundle(
    path: str | Path,
    expected_feature_names: Sequence[str],
    expected_sha256: str,
) -> ModelBundle:
    return load_model_bundle_bytes(
        _read_bundle_path_bytes(path),
        expected_feature_names,
        expected_sha256,
    )
