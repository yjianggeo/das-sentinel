from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Job
from das_sentinel.services.jobs import (
    JobConflictError,
    JobNotFoundError,
    JobService,
    JobValidationError,
)
from sqlalchemy import func, select


@pytest.fixture
def job_database(tmp_path: Path):
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    database_url = f"sqlite+pysqlite:///{app_data / 'jobs.sqlite3'}"
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    sessions = session_factory(engine)
    try:
        yield sessions, app_data / "artifacts"
    finally:
        engine.dispose()


@contextmanager
def seeded_service(job_database, *, enabled: bool = True):
    sessions, artifact_root = job_database
    artifact_root.mkdir(exist_ok=True)
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name="urban-a",
            data_path="C:/das/raw",
            timezone="Asia/Shanghai",
            cable_type="urban",
            enabled=enabled,
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=7,
            payload={"device": "auto", "bands": [[1, 4], [4, 20]]},
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path="C:/das/raw/a.h5",
            fingerprint="a" * 64,
            size_bytes=123,
            mtime_ns=456,
        )
        yield JobService(session, artifact_root), repositories, site, profile, source


def test_enqueue_is_idempotent_for_source_profile_pipeline(job_database):
    with seeded_service(job_database) as (service, _, _, profile, source):
        first = service.enqueue(source.id, profile.id, "pipeline-v1")
        second = service.enqueue(source.id, profile.id, "pipeline-v1")

        assert second.id == first.id
        assert service.list() == [first]


def test_enqueue_requires_enabled_matching_site(job_database):
    with seeded_service(job_database, enabled=False) as (
        service,
        repositories,
        disabled_site,
        profile,
        source,
    ):
        with pytest.raises(JobValidationError, match="disabled"):
            service.enqueue(source.id, profile.id, "pipeline-v1")
        disabled_site.enabled = True
        service.session.flush()
        other = repositories.sites.add(
            name="subsea-b",
            data_path="C:/das/other",
            timezone="UTC",
            cable_type="subsea",
        )
        wrong_profile = repositories.profiles.add(
            site_id=other.id,
            version=1,
            payload={},
            is_current=True,
        )
        with pytest.raises(JobValidationError, match="different sites"):
            service.enqueue(source.id, wrong_profile.id, "pipeline-v1")
        assert service.list() == []

        with pytest.raises(JobNotFoundError, match="source"):
            service.enqueue("missing", profile.id, "pipeline-v1")


def test_service_flushes_but_caller_owns_rollback(job_database):
    sessions, artifact_root = job_database
    with seeded_service(job_database) as (_, _, _, profile, source):
        profile_id = profile.id
        source_id = source.id

    session = sessions()
    try:
        service = JobService(session, artifact_root)
        created = service.enqueue(source_id, profile_id, "rollback-pipeline")
        assert session.get(Job, created.id) is not None
        session.rollback()
    finally:
        session.close()

    with sessions() as verification:
        count = verification.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.pipeline_version == "rollback-pipeline")
        )
        assert count == 0


def test_claim_selects_oldest_and_returns_deeply_immutable_snapshot(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        oldest = service.enqueue(source.id, profile.id, "pipeline-old")
        oldest_row = service.session.get(Job, oldest.id)
        assert oldest_row is not None
        oldest_row.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        service.session.flush()
        other_source = repositories.source_files.add(
            site_id=site.id,
            path="C:/das/raw/b.h5",
            fingerprint="b" * 64,
            size_bytes=456,
            mtime_ns=789,
        )
        service.enqueue(other_source.id, profile.id, "pipeline-new")

        claim = service.claim("worker-a", now=datetime(2026, 7, 30, tzinfo=UTC))

        assert claim is not None
        assert claim.id == oldest.id
        assert claim.attempt_count == 1
        assert claim.worker_id == "worker-a"
        assert claim.source_path == "C:/das/raw/a.h5"
        assert claim.source_fingerprint == "a" * 64
        assert claim.site_timezone == "Asia/Shanghai"
        assert claim.cable_type == "urban"
        assert claim.profile_version == 7
        assert claim.pipeline_version == "pipeline-old"
        assert isinstance(claim.profile_payload, MappingProxyType)
        assert claim.profile_payload["bands"] == ((1, 4), (4, 20))
        with pytest.raises(TypeError):
            claim.profile_payload["device"] = "cpu"


def test_only_one_worker_claims_job(job_database):
    with seeded_service(job_database) as (service, _, _, profile, source):
        queued = service.enqueue(source.id, profile.id, "pipeline-v1")

        first = service.claim("worker-a")
        second = service.claim("worker-b")

        assert first is not None
        assert first.id == queued.id
        assert second is None


def test_heartbeat_requires_owner_and_valid_progress_without_partial_change(
    job_database,
):
    with seeded_service(job_database) as (service, _, _, profile, source):
        queued = service.enqueue(source.id, profile.id, "pipeline-v1")
        claim = service.claim("worker-a")
        assert claim is not None

        with pytest.raises(JobValidationError, match="progress"):
            service.heartbeat(queued.id, "worker-a", 4, 3)
        unchanged = service.get(queued.id)
        assert (unchanged.progress_numerator, unchanged.progress_denominator) == (0, 0)

        with pytest.raises(JobConflictError, match="owner"):
            service.heartbeat(queued.id, "worker-b", 1, 3)

        heartbeat = datetime(2026, 7, 30, 1, 2, tzinfo=UTC)
        updated = service.heartbeat(queued.id, "worker-a", 2, 3, now=heartbeat)
        assert (updated.progress_numerator, updated.progress_denominator) == (2, 3)
        assert updated.heartbeat_at == heartbeat
        assert updated.heartbeat_at.utcoffset() == timedelta(0)


def test_complete_requires_owner_and_exact_versions(job_database, tmp_path: Path):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        artifact = service.artifact_root / "run" / "waterfall.png"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"png")

        with pytest.raises(JobConflictError, match="owner"):
            service.complete(
                job.id,
                "worker-b",
                profile.version,
                "pipeline-v1",
                [artifact],
            )
        with pytest.raises(JobConflictError, match="profile version"):
            service.complete(
                job.id,
                "worker-a",
                profile.version + 1,
                "pipeline-v1",
                [artifact],
            )
        with pytest.raises(JobConflictError, match="pipeline version"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v2",
                [artifact],
            )

        completed = service.complete(
            job.id,
            "worker-a",
            profile.version,
            "pipeline-v1",
            [artifact],
        )
        assert completed.status == "succeeded"
        assert completed.retry_eligible is False


@pytest.mark.parametrize(
    "artifact_builder",
    [
        lambda root, outside: Path("relative.png"),
        lambda root, outside: root / ".." / "outside.png",
        lambda root, outside: outside,
        lambda root, outside: root / "missing.png",
    ],
)
def test_complete_rejects_unsafe_artifacts_without_changing_job(
    job_database,
    tmp_path: Path,
    artifact_builder,
):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        unsafe = artifact_builder(service.artifact_root, outside)

        with pytest.raises(JobValidationError, match="artifact"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v1",
                [unsafe],
            )

        assert service.get(job.id).status == "running"


def test_fail_retry_limit_and_cancel_transitions(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        failed_job = service.enqueue(source.id, profile.id, "pipeline-fail")
        service.claim("worker-a")
        failed = service.fail(
            failed_job.id,
            "worker-a",
            error_code="analysis_failed",
            error_message="detector failed",
            retry_eligible=True,
        )
        assert failed.status == "failed"
        assert failed.retry_eligible is True
        assert failed.error_code == "analysis_failed"

        retried = service.retry(failed_job.id)
        assert retried.status == "queued"
        assert retried.error_code is None
        service.claim("worker-b")
        service.fail(
            failed_job.id,
            "worker-b",
            error_code="again",
            error_message="again",
            retry_eligible=True,
        )
        service.retry(failed_job.id)
        service.claim("worker-c")
        exhausted = service.fail(
            failed_job.id,
            "worker-c",
            error_code="third",
            error_message="third",
            retry_eligible=True,
        )
        assert exhausted.attempt_count == 3
        assert exhausted.retry_eligible is False
        with pytest.raises(JobConflictError, match="retry"):
            service.retry(failed_job.id)

        other_source = repositories.source_files.add(
            site_id=site.id,
            path="C:/das/raw/c.h5",
            fingerprint="c" * 64,
            size_bytes=1,
            mtime_ns=1,
        )
        cancelled = service.enqueue(other_source.id, profile.id, "pipeline-cancel")
        assert service.cancel(cancelled.id).status == "cancelled"
        with pytest.raises(JobConflictError, match="cancel"):
            service.cancel(cancelled.id)


def test_failure_fields_are_bounded_before_state_changes(job_database):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")

        with pytest.raises(JobValidationError, match="error code"):
            service.fail(
                job.id,
                "worker-a",
                error_code="x" * 129,
                error_message="bounded",
            )
        with pytest.raises(JobValidationError, match="error message"):
            service.fail(
                job.id,
                "worker-a",
                error_code="bounded",
                error_message="x" * 4097,
            )
        assert service.get(job.id).status == "running"


def test_stale_recovery_requeues_then_exhausts_attempts(job_database):
    now = datetime(2026, 7, 30, 12, tzinfo=UTC)
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a", now=now - timedelta(minutes=3))

        assert service.recover_stale(now=now) == [job.id]
        recovered = service.get(job.id)
        assert recovered.status == "queued"
        assert recovered.worker_id is None

        service.claim("worker-b", now=now)
        service.fail(
            job.id,
            "worker-b",
            error_code="again",
            error_message="again",
        )
        service.retry(job.id)
        service.claim("worker-c", now=now)
        service.heartbeat(
            job.id,
            "worker-c",
            0,
            0,
            now=now - timedelta(minutes=3),
        )

        assert service.recover_stale(now=now) == [job.id]
        exhausted = service.get(job.id)
        assert exhausted.status == "failed"
        assert exhausted.retry_eligible is False
        assert exhausted.error_code == "stale_worker"


def test_missing_job_uses_explicit_not_found_exception(job_database):
    with (
        seeded_service(job_database) as (service, _, _, _, _),
        pytest.raises(JobNotFoundError),
    ):
        service.get("missing")
