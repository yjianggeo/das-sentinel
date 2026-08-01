from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from das_sentinel.classifiers.supervised import (
    TrainingGateError,
    save_model_bundle,
    train_model,
)
from das_sentinel.domain import Label
from das_sentinel.persistence.models import (
    Event,
    ModelVersion,
    ProcessingProfile,
    ProcessingRun,
    Review,
    Site,
    SourceFile,
    utc_now,
)
from das_sentinel.persistence.repositories import ModelRepository

LABEL_GATES = {
    "per_class_reviewed": 20,
    "macro_f1": 0.70,
    "min_recall": 0.50,
}
SUPERVISED_LABELS = (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)


class TrainingServiceError(RuntimeError):
    """Base class for training lifecycle errors exposed at the API boundary."""


class TrainingDataError(TrainingServiceError):
    """The reviewed dataset does not satisfy the training gates."""


class PublishGateError(TrainingServiceError):
    """Publishing or rollback would violate the model lifecycle rules."""


@dataclass(frozen=True)
class TrainingRow:
    event_id: str
    label: str
    features: dict[str, float]
    group: str


class TrainingService:
    def __init__(self, session: Session, artifact_root: Path) -> None:
        self.session = session
        self.artifact_root = Path(artifact_root).resolve()
        self.models = ModelRepository(session)

    def get(self, model_id: str) -> ModelVersion:
        model = self.models.get(model_id)
        if model is None:
            raise LookupError(f"model not found: {model_id}")
        return model

    def list_for_site(self, site_id: str) -> list[ModelVersion]:
        return self.models.list_for_site(site_id)

    def active_model(self, site_id: str) -> ModelVersion | None:
        statement = (
            select(ModelVersion)
            .where(
                ModelVersion.site_id == site_id,
                ModelVersion.status == "active",
            )
            .order_by(ModelVersion.published_at.desc(), ModelVersion.id.desc())
        )
        return self.session.scalars(statement).first()

    def train(self, *, site_id: str) -> ModelVersion:
        site = self.session.get(Site, site_id)
        if site is None:
            raise LookupError(f"site not found: {site_id}")
        profile = self._current_profile(site_id)
        rows = self._reviewed_rows(site_id)
        if not rows:
            raise TrainingDataError("no reviewed events are available for training")
        try:
            bundle, report = train_model(
                rows,
                groups=[row.group for row in rows],
                rule_profile=dict(profile.payload),
                rule_profile_version=str(profile.version),
            )
        except TrainingGateError as error:
            raise TrainingDataError(str(error)) from error

        version = self._next_version(site_id)
        bundle_path = self._bundle_path(site_id, version)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_sha256 = save_model_bundle(bundle, bundle_path)
        return self.models.add(
            site_id=site_id,
            version=version,
            artifact_path=str(bundle_path),
            bundle_sha256=bundle_sha256,
            status="candidate",
            dataset_sha256=bundle.training_snapshot_hash,
            metrics=_metrics_payload(bundle, report),
        )

    def publish(self, model_id: str) -> ModelVersion:
        model = self.get(model_id)
        if model.status != "candidate":
            raise PublishGateError("only a candidate model can be published")
        metrics = json.loads(model.metrics_json)
        _verify_publish_gates(metrics)
        active = self.active_model(model.site_id)
        if active is not None and active.id != model.id:
            active_metrics = json.loads(active.metrics_json)
            if metrics["macro_f1"] < active_metrics["macro_f1"]:
                raise PublishGateError(
                    "candidate macro-F1 is below the active model"
                )
            self.models.set_status(
                active.id,
                status="retired",
                published_at=active.published_at,
            )
        return self.models.set_status(
            model.id,
            status="active",
            published_at=utc_now(),
        )

    def rollback(self, site_id: str) -> ModelVersion:
        active = self.active_model(site_id)
        if active is None:
            raise PublishGateError("no active model to roll back from")
        statement = (
            select(ModelVersion)
            .where(
                ModelVersion.site_id == site_id,
                ModelVersion.status == "retired",
                ModelVersion.published_at.is_not(None),
                ModelVersion.id != active.id,
            )
            .order_by(ModelVersion.published_at.desc(), ModelVersion.id.desc())
        )
        for candidate in self.session.scalars(statement):
            try:
                _verify_publish_gates(json.loads(candidate.metrics_json))
            except PublishGateError:
                continue
            self.models.set_status(
                active.id,
                status="retired",
                published_at=active.published_at,
            )
            return self.models.set_status(
                candidate.id,
                status="active",
                published_at=candidate.published_at,
            )
        raise PublishGateError("no previous valid model to roll back to")

    def _current_profile(self, site_id: str) -> ProcessingProfile:
        statement = select(ProcessingProfile).where(
            ProcessingProfile.site_id == site_id,
            ProcessingProfile.is_current.is_(True),
        )
        profile = self.session.scalars(statement).first()
        if profile is None:
            raise TrainingDataError("site has no current processing profile")
        return profile

    def _reviewed_rows(self, site_id: str) -> list[TrainingRow]:
        latest_label = (
            select(Review.label)
            .where(Review.event_id == Event.id)
            .order_by(Review.created_at.desc(), Review.id.desc())
            .limit(1)
            .correlate(Event)
            .scalar_subquery()
        )
        statement = (
            select(Event, latest_label)
            .join(ProcessingRun, ProcessingRun.id == Event.processing_run_id)
            .join(SourceFile, SourceFile.id == ProcessingRun.source_file_id)
            .where(SourceFile.site_id == site_id)
            .where(latest_label.is_not(None))
            .order_by(Event.id)
        )
        rows = []
        for event, label in self.session.execute(statement):
            rows.append(
                TrainingRow(
                    event_id=event.id,
                    label=str(label),
                    features={
                        str(name): float(value)
                        for name, value in json.loads(event.features_json).items()
                    },
                    group=event.occurred_at.date().isoformat(),
                )
            )
        return rows

    def _next_version(self, site_id: str) -> str:
        highest = 0
        for model in self.models.list_for_site(site_id):
            if model.version.isdigit():
                highest = max(highest, int(model.version))
        return str(highest + 1)

    def _bundle_path(self, site_id: str, version: str) -> Path:
        candidate = (
            self.artifact_root / "models" / site_id / version / "model.bundle"
        ).resolve()
        if not candidate.is_relative_to(self.artifact_root):
            raise TrainingServiceError("model bundle path escapes the artifact root")
        return candidate


def _metrics_payload(bundle, report) -> dict[str, Any]:
    return {
        "macro_f1": report.macro_f1,
        "recall_by_label": {
            str(label): value for label, value in report.recall_by_label.items()
        },
        "precision_by_label": {
            str(label): value for label, value in report.precision_by_label.items()
        },
        "confusion_matrix": report.confusion_matrix.tolist(),
        "rule_baseline_macro_f1": report.rule_baseline_macro_f1,
        "rule_profile_version": report.rule_profile_version,
        "rule_profile_hash": report.rule_profile_hash,
        "reviewed_count_by_label": {
            str(label): value
            for label, value in report.reviewed_count_by_label.items()
        },
        "feature_names": list(bundle.feature_names),
        "feature_schema_version": bundle.feature_schema_version,
        "sklearn_version": bundle.sklearn_version,
        "training_snapshot_hash": bundle.training_snapshot_hash,
    }


def _verify_publish_gates(metrics: dict[str, Any]) -> None:
    counts = metrics.get("reviewed_count_by_label") or {}
    per_class = LABEL_GATES["per_class_reviewed"]
    for label in SUPERVISED_LABELS:
        if int(counts.get(str(label), 0)) < per_class:
            raise PublishGateError(
                f"each supervised label requires {per_class} reviewed events"
            )
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    if macro_f1 < LABEL_GATES["macro_f1"]:
        raise PublishGateError(
            f"macro-F1 {macro_f1:.3f} is below the {LABEL_GATES['macro_f1']:.2f} gate"
        )
    recalls = metrics.get("recall_by_label") or {}
    lowest = min((float(recalls.get(str(label), 0.0)) for label in SUPERVISED_LABELS), default=0.0)
    if lowest < LABEL_GATES["min_recall"]:
        raise PublishGateError(
            f"class recall {lowest:.3f} is below the {LABEL_GATES['min_recall']:.2f} gate"
        )
