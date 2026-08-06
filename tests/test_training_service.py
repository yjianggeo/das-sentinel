from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from das_sentinel.classifiers.similarity import FeatureVectorizer
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import EventRecord
from das_sentinel.features import extract_features
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Event
from das_sentinel.services.training import (
    PublishGateError,
    TrainingDataError,
    TrainingService,
)
from sqlalchemy import select

from tests.synthetic import preprocessed_block

LABELS = ("earthquake", "blast", "traffic", "noise")


def _real_training_features(site_id: str, label_index: int, index: int):
    block = preprocessed_block(
        np.tile(np.linspace(-1.0, 1.0, 80, dtype=np.float32)[:, None], (1, 4))
    )
    event = EventRecord(
        event_id=f"feature-{label_index}-{index}",
        start_sample=1_010,
        end_sample=1_070,
        peak_sample=1_040,
        channel_start=0,
        channel_stop=4,
        evidence=(),
    )
    features = extract_features(
        block,
        event,
        context={
            "site_id": site_id,
            "cable_type": "urban",
            "device_config": "cpu",
        },
    )
    features["duration_s"] = label_index * 100.0 + index
    features["rms"] = label_index + index / 100.0
    return features


def _sessions(tmp_path: Path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'das.sqlite3'}"
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    return session_factory(engine)


def _seed_site(
    sessions,
    *,
    per_label: int = 25,
    separable: bool = True,
) -> str:
    """One site, one current profile, reviewed events across five days."""
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash="not-used")
        site = repositories.sites.add(
            name="site-a",
            data_path="/data",
            timezone="UTC",
            cable_type="urban",
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path="/data/a.h5",
            fingerprint="a" * 64,
            size_bytes=1,
            mtime_ns=1,
        )
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="pipeline-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        base = datetime(2026, 7, 1, tzinfo=UTC)
        for label_index, label in enumerate(LABELS):
            for index in range(per_label):
                features = _real_training_features(site.id, label_index, index)
                if not separable:
                    features["duration_s"] = 1.0
                    features["rms"] = 1.0
                event = repositories.events.add(
                    run_id=run.id,
                    event_uid=f"{label}-{index}",
                    start_sample=0,
                    end_sample=10,
                    channel_start=0,
                    channel_stop=2,
                    occurred_at=base + timedelta(days=index % 5, minutes=index),
                    features=features,
                )
                repositories.reviews.add(
                    event.id,
                    label=label,
                    note="seeded",
                    admin_id=1,
                )
        return site.id


@pytest.fixture
def training_env(tmp_path: Path):
    sessions = _sessions(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    return sessions, artifact_root


def test_training_creates_candidate_not_active(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions.begin() as session:
        service = TrainingService(session, artifact_root)
        model = service.train(site_id=site_id)
        assert model.status == "candidate"
        assert service.active_model(site_id) is None

    with sessions() as session:
        stored = TrainingService(session, artifact_root).get(model.id)
        assert stored.dataset_sha256 is not None
        assert len(stored.bundle_sha256) == 64
        assert Path(stored.artifact_path).resolve().is_relative_to(
            artifact_root.resolve()
        )
        assert Path(stored.artifact_path).exists()


def test_reviewed_rows_keep_extract_features_string_context(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions() as session:
        rows = TrainingService(session, artifact_root)._reviewed_rows(site_id)

    assert rows[0].features["site_id"] == site_id
    assert rows[0].features["cable_type"] == "urban"
    FeatureVectorizer().transform(row.features for row in rows)


def test_training_snapshot_is_deterministic_and_tracks_reviews(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions.begin() as session:
        first = TrainingService(session, artifact_root).train(site_id=site_id)
    with sessions.begin() as session:
        second = TrainingService(session, artifact_root).train(site_id=site_id)
    assert first.dataset_sha256 == second.dataset_sha256

    with sessions.begin() as session:
        repositories = Repositories(session)
        event = session.scalars(
            select(Event).where(Event.event_uid == "earthquake-0")
        ).one()
        repositories.reviews.add(event.id, label="noise", admin_id=1)
    with sessions.begin() as session:
        third = TrainingService(session, artifact_root).train(site_id=site_id)
    assert third.dataset_sha256 != first.dataset_sha256


def test_training_requires_twenty_reviewed_per_class(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions, per_label=15)

    with sessions.begin() as session, pytest.raises(
        TrainingDataError, match="20 reviewed events"
    ):
        TrainingService(session, artifact_root).train(site_id=site_id)


def test_publish_is_atomic_and_rollback_restores_previous(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions.begin() as session:
        service = TrainingService(session, artifact_root)
        first = service.train(site_id=site_id)
        second = service.train(site_id=site_id)
        service.publish(first.id)
        assert service.active_model(site_id).id == first.id

        service.publish(second.id)
        assert service.active_model(site_id).id == second.id
        assert service.get(first.id).status == "retired"

        rolled_back = service.rollback(site_id)
        assert rolled_back.id == first.id
        assert service.active_model(site_id).id == first.id
        assert service.get(second.id).status == "retired"

        assert Path(service.get(second.id).artifact_path).exists()


def test_rollback_without_previous_model_raises(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions.begin() as session:
        service = TrainingService(session, artifact_root)
        model = service.train(site_id=site_id)
        service.publish(model.id)
        with pytest.raises(PublishGateError):
            service.rollback(site_id)


def test_publish_rejects_candidate_below_metric_gate(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions.begin() as session:
        service = TrainingService(session, artifact_root)
        model = service.train(site_id=site_id)
        metrics = dict(json.loads(model.metrics_json))
        metrics["macro_f1"] = 0.40
        stored = Repositories(session).models.get(model.id)
        stored.metrics_json = json.dumps(metrics)
        with pytest.raises(PublishGateError, match="0.70"):
            service.publish(model.id)
        assert service.active_model(site_id) is None


def test_publish_rejects_candidate_below_active_model(training_env):
    sessions, artifact_root = training_env
    site_id = _seed_site(sessions)

    with sessions.begin() as session:
        service = TrainingService(session, artifact_root)
        strong = service.train(site_id=site_id)
        weak = service.train(site_id=site_id)
        service.publish(strong.id)
        strong_metrics = json.loads(
            Repositories(session).models.get(strong.id).metrics_json
        )
        weak_stored = Repositories(session).models.get(weak.id)
        weak_metrics = dict(json.loads(weak_stored.metrics_json))
        weak_metrics["macro_f1"] = max(0.70, strong_metrics["macro_f1"] - 0.05)
        weak_stored.metrics_json = json.dumps(weak_metrics)
        with pytest.raises(PublishGateError, match="active"):
            service.publish(weak.id)
        assert service.active_model(site_id).id == strong.id
