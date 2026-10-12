from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from das_sentinel.persistence import Repositories
from das_sentinel.persistence.models import Event, ProcessingProfile, Review
from sqlalchemy import select, text

from tests.test_models_api import model_api  # noqa: F401
from tests.test_training_service import LABELS, _sessions


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        pytest.fail("readiness must never fit or train a model")

    monkeypatch.setattr("sklearn.ensemble.RandomForestClassifier.fit", fail)
    monkeypatch.setattr("das_sentinel.services.training.TrainingService.train", fail)
    monkeypatch.setattr("das_sentinel.services.training.train_model", fail)
    monkeypatch.setattr("das_sentinel.classifiers.supervised.train_model", fail)


@pytest.fixture
def readiness_env(tmp_path: Path):
    return _sessions(tmp_path), tmp_path / "artifacts"


def _seed(sessions, *, name="site-a", per_label=20, days=5, blast_days=None):
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name=name, data_path="/data", timezone="Asia/Shanghai", cable_type="urban"
        )
        profile = repositories.profiles.add(
            site_id=site.id, version=1, payload={}, is_current=True
        )
        source = repositories.source_files.add(
            site_id=site.id, path="/data/a.h5", fingerprint="a" * 64,
            size_bytes=1, mtime_ns=1,
        )
        run = repositories.runs.add(
            source_file_id=source.id, processing_profile_id=profile.id,
            pipeline_version="v1", requested_backend="cpu", actual_backend="cpu",
            status="succeeded",
        )
        for label in LABELS:
            label_days = blast_days if label == "blast" and blast_days else days
            for index in range(per_label):
                event = repositories.events.add(
                    run_id=run.id, event_uid=f"{label}-{index}",
                    start_sample=0, end_sample=10, channel_start=0, channel_stop=2,
                    occurred_at=datetime(2026, 7, 1, tzinfo=UTC)
                    + timedelta(days=index % label_days, minutes=index),
                    features={"rms": 1.0, "duration_s": 2.0},
                )
                event.id = f"{name}-{label}-{index:03d}"
                session.flush()
                repositories.reviews.add(event.id, label=label)
        return site.id, run.id


def _get(env, site_id):
    from das_sentinel.services.training_readiness import TrainingReadinessService

    sessions, artifact_root = env
    with sessions() as session:
        return TrainingReadinessService(session, artifact_root).get(site_id)


def test_ready_is_structural_and_does_not_write_or_train(readiness_env):
    sessions, artifact_root = readiness_env
    site_id, _ = _seed(sessions)
    with sessions() as session:
        before = list(session.connection().connection.iterdump())

    result = _get(readiness_env, site_id)

    assert result["ready"] is True
    assert result["blockers"] == []
    assert result["reviewed_events"] == 80
    assert result["unreviewed_events"] == 0
    assert result["unsupported_label_count"] == 0
    assert result["classes"] == [
        {"label": label, "reviewed_count": 20, "utc_day_count": 5,
         "missing_count": 0, "minimum_days_missing": 0}
        for label in LABELS
    ]
    assert result["pipeline_counts"] == {"v1": 80}
    assert result["profile_counts"] == {"1": 80}
    assert result["model_count"] == 0
    assert result["active_model_version"] is None
    assert len(result["folds"]) == 5
    for index, fold in enumerate(result["folds"], start=1):
        assert fold == {
            "fold": index, "train_counts": dict.fromkeys(LABELS, 16),
            "test_counts": dict.fromkeys(LABELS, 4),
            "missing_train_labels": [], "missing_test_labels": [],
        }
    assert any("结构门槛" in item for item in result["warnings"])
    assert any("独立物理事件" in item for item in result["warnings"])
    assert not artifact_root.exists()
    with sessions() as session:
        assert list(session.connection().connection.iterdump()) == before


def test_twenty_labels_do_not_hide_missing_blast_folds(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions, per_label=30, blast_days=3)

    result = _get(readiness_env, site_id)

    assert result["ready"] is False
    assert result["classes"][1] == {
        "label": "blast", "reviewed_count": 30, "utc_day_count": 3,
        "missing_count": 0, "minimum_days_missing": 2,
    }
    assert result["folds"] == []
    assert any("折" in item and "爆破" in item for item in result["blockers"])


def test_latest_review_uses_timestamp_then_id_and_is_site_scoped(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions)
    _seed(sessions, name="other", per_label=2)
    with sessions.begin() as session:
        event = session.get(Event, "site-a-earthquake-000")
        original = session.scalars(select(Review).where(Review.event_id == event.id)).one()
        session.add(Review(
            event_id=event.id, label="noise", created_at=original.created_at
        ))
        session.flush()
        session.add(Review(
            event_id=event.id, label="blast",
            created_at=original.created_at - timedelta(days=1),
        ))

    result = _get(readiness_env, site_id)

    assert result["reviewed_events"] == 80
    assert [item["reviewed_count"] for item in result["classes"]] == [19, 20, 20, 21]
    assert result["classes"][0]["missing_count"] == 1
    assert result["ready"] is False


def test_feasible_groups_pass_when_the_initial_stratification_misses_classes(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions, per_label=30)
    distributions = [
        [6, 6, 2, 1, 4, 5, 0, 6],
        [0, 2, 0, 12, 4, 3, 1, 8],
        [0, 6, 2, 3, 7, 7, 4, 1],
        [17, 2, 3, 0, 3, 0, 4, 1],
    ]
    with sessions.begin() as session:
        for label, counts in zip(LABELS, distributions, strict=True):
            days = [day for day, count in enumerate(counts) for _ in range(count)]
            for index, day in enumerate(days):
                session.get(Event, f"site-a-{label}-{index:03d}").occurred_at = (
                    datetime(2026, 7, 1, tzinfo=UTC) + timedelta(days=day)
                )

    result = _get(readiness_env, site_id)

    assert all(item["minimum_days_missing"] == 0 for item in result["classes"])
    assert result["ready"] is True
    assert result["blockers"] == []
    assert len(result["folds"]) == 5
    for fold in result["folds"]:
        assert all(count > 0 for count in fold["train_counts"].values())
        assert all(count > 0 for count in fold["test_counts"].values())
        assert fold["missing_test_labels"] == fold["missing_train_labels"] == []
    assert all(sum(fold["test_counts"][label] for fold in result["folds"]) == 30
               for label in LABELS)


def test_unknown_latest_label_blocks_instead_of_becoming_noise(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions)
    with sessions.begin() as session:
        session.execute(text("PRAGMA ignore_check_constraints = ON"))
        session.add(Review(event_id="site-a-earthquake-000", label="unknown"))

    result = _get(readiness_env, site_id)

    assert result["unsupported_label_count"] == 1
    assert result["reviewed_events"] == 80
    assert [item["reviewed_count"] for item in result["classes"]] == [19, 20, 20, 20]
    assert result["ready"] is False
    assert any("不支持" in item for item in result["blockers"])


@pytest.mark.parametrize("features, message", [
    ({"rms": float("nan"), "duration_s": 2.0}, "有限数值"),
    ({"rms": True, "duration_s": 2.0}, "有限数值"),
    ({"rms": "1", "duration_s": 2.0}, "有限数值"),
    ({"rms": 10**400, "duration_s": 2.0}, "有限数值"),
    ({"rms": 1.0}, "特征字段"),
    ({"rms": 1.0, "duration_s": 2.0, "filename": 1.0}, "特征字段"),
    ([], "特征"),
])
def test_bad_features_block_readiness(readiness_env, features, message):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions)
    with sessions.begin() as session:
        session.get(Event, "site-a-earthquake-000").features_json = json.dumps(features)

    result = _get(readiness_env, site_id)

    assert result["ready"] is False
    assert result["reviewed_events"] == 80
    assert any(message in item for item in result["blockers"])


def test_only_approved_numeric_subset_and_string_context_are_allowed(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions)
    with sessions.begin() as session:
        for event in session.scalars(select(Event)):
            event.features_json = json.dumps({
                "rms": 1.0, "duration_s": 2.0,
                "site_id": site_id, "cable_type": "urban", "device_config": "cpu",
            })

    assert _get(readiness_env, site_id)["ready"] is True


def test_empty_site_and_missing_current_profile_report_gaps(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions, per_label=0)
    with sessions.begin() as session:
        session.scalars(select(ProcessingProfile)).one().is_current = False

    result = _get(readiness_env, site_id)

    assert result["ready"] is False
    assert result["reviewed_events"] == 0
    assert result["folds"] == []
    assert all(item["missing_count"] == 20 for item in result["classes"])
    assert all(item["minimum_days_missing"] == 5 for item in result["classes"])
    assert any("当前处理配置" in item for item in result["blockers"])


def test_fewer_than_five_utc_days_cannot_split(readiness_env):
    sessions, _ = readiness_env
    site_id, _ = _seed(sessions, days=4)
    with sessions.begin() as session:
        # This is July 1 in UTC, despite a July 2 local timestamp.
        session.get(Event, "site-a-earthquake-000").occurred_at = datetime(
            2026, 7, 2, 0, 30, tzinfo=timezone(timedelta(hours=8))
        )

    result = _get(readiness_env, site_id)

    assert result["ready"] is False
    assert result["folds"] == []
    assert all(item["utc_day_count"] == 4 for item in result["classes"])
    assert any("五" in item for item in result["blockers"])


def test_mixed_versions_warn_and_model_metadata_is_site_scoped(readiness_env):
    sessions, _ = readiness_env
    site_id, run_id = _seed(sessions)
    other_id, _ = _seed(sessions, name="other", per_label=0)
    with sessions.begin() as session:
        repositories = Repositories(session)
        old_profile = session.scalars(select(ProcessingProfile).where(
            ProcessingProfile.site_id == site_id
        )).one()
        profile = repositories.profiles.add(
            site_id=site_id, version=2, payload={}, is_current=False
        )
        from das_sentinel.persistence.models import ProcessingRun

        run = session.get(ProcessingRun, run_id)
        second_run = repositories.runs.add(
            source_file_id=run.source_file_id, processing_profile_id=profile.id,
            pipeline_version="v2", requested_backend="cpu", actual_backend="cpu",
            status="succeeded",
        )
        session.get(Event, "site-a-noise-000").processing_run_id = second_run.id
        unreviewed = repositories.events.add(
            run_id=run_id, event_uid="unreviewed", start_sample=0, end_sample=10,
            channel_start=0, channel_stop=2, occurred_at=datetime(2026, 7, 1, tzinfo=UTC),
            features={},
        )
        assert unreviewed.id
        assert old_profile.version == 1
        for model_site, version, status in [
            (site_id, "1", "retired"), (site_id, "2", "active"),
            (other_id, "99", "active"),
        ]:
            repositories.models.add(
                site_id=model_site, version=version, status=status,
                artifact_path="unused", bundle_sha256="b" * 64,
            )

    result = _get(readiness_env, site_id)

    assert result["ready"] is True
    assert result["unreviewed_events"] == 1
    assert result["pipeline_counts"] == {"v1": 79, "v2": 1}
    assert result["profile_counts"] == {"1": 79, "2": 1}
    assert result["model_count"] == 2
    assert result["active_model_version"] == "2"
    assert any("流水线版本" in item for item in result["warnings"])
    assert any("处理配置版本" in item for item in result["warnings"])


def test_nonexistent_site_raises_lookup_error(readiness_env):
    with pytest.raises(LookupError, match="site not found"):
        _get(readiness_env, "missing")


def test_training_readiness_api_is_authenticated_and_read_only(model_api):  # noqa: F811
    client, sessions, _, site_id = model_api
    with sessions() as session:
        before = list(session.connection().connection.iterdump())

    response = client.get(f"/api/v1/sites/{site_id}/training-readiness")

    assert response.status_code == 200
    assert response.json()["site_id"] == site_id
    assert response.json()["ready"] is True
    assert client.get("/api/v1/sites/missing/training-readiness").status_code == 404
    with sessions() as session:
        assert list(session.connection().connection.iterdump()) == before
    client.cookies.clear()
    assert client.get(f"/api/v1/sites/{site_id}/training-readiness").status_code == 401
