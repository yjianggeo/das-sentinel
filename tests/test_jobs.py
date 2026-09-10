from __future__ import annotations

import os
import subprocess
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
from das_sentinel.persistence.models import (
    Artifact,
    Event,
    Job,
    ProcessingRun,
    Review,
    SourceFile,
)
from das_sentinel.services.jobs import (
    MAX_SQLITE_INTEGER,
    STALE_AFTER,
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


@pytest.mark.parametrize("delete_all", [False, True])
def test_delete_cancelled_preserves_analysis_history_and_files(job_database, tmp_path, delete_all):
    with seeded_service(job_database) as (service, repositories, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.cancel(job.id)
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="pipeline-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            job_id=job.id,
            status="failed",
        )
        event = repositories.events.add(
            run_id=run.id,
            event_uid="preserved-event",
            start_sample=0,
            end_sample=20,
            channel_start=0,
            channel_stop=2,
            occurred_at=datetime(2026, 9, 9, tzinfo=UTC),
        )
        review = repositories.reviews.add(event.id, label="earthquake", note="keep")
        raw_path = tmp_path / "raw.h5"
        raw_path.write_bytes(b"source-data")
        source.path = str(raw_path)
        artifact_path = service.artifact_root / job.id / "waterfall.png"
        artifact_path.parent.mkdir()
        artifact_path.write_bytes(b"retained-image")
        artifact = repositories.artifacts.add(
            event_id=event.id,
            kind="waterfall",
            path=str(artifact_path),
            media_type="image/png",
            sha256="b" * 64,
        )
        queued = service.enqueue(source.id, profile.id, "pipeline-v2")
        queued_run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="pipeline-v2",
            requested_backend="cpu",
            actual_backend="cpu",
            job_id=queued.id,
        )

        if delete_all:
            assert service.delete_cancelled() == 1
        else:
            service.delete(job.id)

        service.session.expire_all()
        assert service.session.get(Job, job.id) is None
        assert service.get(queued.id).status == "queued"
        assert service.session.get(ProcessingRun, run.id).job_id is None
        assert service.session.get(ProcessingRun, queued_run.id).job_id == queued.id
        assert service.session.get(SourceFile, source.id).path == str(raw_path)
        assert service.session.get(Event, event.id).processing_run_id == run.id
        assert service.session.get(Review, review.id).note == "keep"
        assert service.session.get(Artifact, artifact.id).path == str(artifact_path)
        assert raw_path.read_bytes() == b"source-data"
        assert artifact_path.read_bytes() == b"retained-image"


@pytest.mark.parametrize("delete_all", [False, True])
def test_delete_cancelled_remains_in_callers_transaction(job_database, delete_all):
    sessions, artifact_root = job_database
    with seeded_service(job_database) as (service, repositories, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.cancel(job.id)
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="pipeline-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            job_id=job.id,
        )
        job_id, run_id = job.id, run.id

    with sessions() as session:
        service = JobService(session, artifact_root)
        if delete_all:
            service.delete_cancelled()
        else:
            service.delete(job_id)
        session.rollback()

    with sessions() as session:
        assert session.get(Job, job_id).status == "cancelled"
        assert session.get(ProcessingRun, run_id).job_id == job_id


def test_enqueue_creates_new_job_after_nonretryable_failure(job_database):
    with seeded_service(job_database) as (service, _, _, profile, source):
        original = service.enqueue(source.id, profile.id, "pipeline-v1")
        claim = service.claim("worker-a")
        assert claim is not None
        service.fail(
            claim.id,
            "worker-a",
            error_code="invalid_model",
            error_message="bundle digest mismatch",
            retry_eligible=False,
        )

        replacement = service.enqueue(source.id, profile.id, "pipeline-v1")

        assert replacement.id != original.id
        assert replacement.attempt_count == 0


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
        assert claim.source_size_bytes == 123
        assert claim.source_mtime_ns == 456
        source.size_bytes = 999
        source.mtime_ns = 999
        service.session.flush()
        assert claim.source_size_bytes == 123
        assert claim.source_mtime_ns == 456
        assert claim.site_timezone == "Asia/Shanghai"
        assert claim.cable_type == "urban"
        assert claim.profile_version == 7
        assert claim.pipeline_version == "pipeline-old"
        assert isinstance(claim.profile_payload, MappingProxyType)
        assert claim.profile_payload["bands"] == ((1, 4), (4, 20))
        with pytest.raises(TypeError):
            claim.profile_payload["device"] = "cpu"


def test_first_claim_binds_active_model_and_stale_reclaim_keeps_it(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        active = repositories.models.add(
            site_id=site.id,
            version="7",
            artifact_path=str(
                service.artifact_root
                / "models"
                / site.id
                / "7"
                / "model.bundle"
            ),
            bundle_sha256="7" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        queued = service.enqueue(source.id, profile.id, "pipeline-v1")
        first = service.claim("worker-a")
        assert first is not None
        assert first.id == queued.id
        assert first.model_version_id == active.id
        assert first.model_version == "7"
        assert first.model_artifact_path == active.artifact_path
        assert first.model_bundle_sha256 == active.bundle_sha256
        assert first.model_feature_names == ("duration_s", "rms")
        service.recover_stale(
            now=first.heartbeat_at + STALE_AFTER + timedelta(seconds=1)
        )
        repositories.models.set_status(active.id, status="retired")
        repositories.models.add(
            site_id=site.id,
            version="8",
            artifact_path=str(
                service.artifact_root
                / "models"
                / site.id
                / "8"
                / "model.bundle"
            ),
            bundle_sha256="8" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        second = service.claim("worker-b")
        assert second is not None
        assert second.model_version_id == active.id


def test_first_claim_without_active_model_stays_rules_only_after_reclaim(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        service.enqueue(source.id, profile.id, "pipeline-v1")
        first = service.claim("worker-a")
        assert first is not None
        assert first.model_version_id is None
        assert first.model_version is None
        assert first.model_artifact_path is None
        assert first.model_bundle_sha256 is None
        assert first.model_feature_names is None
        service.recover_stale(
            now=first.heartbeat_at + STALE_AFTER + timedelta(seconds=1)
        )
        repositories.models.add(
            site_id=site.id,
            version="1",
            artifact_path=str(
                service.artifact_root
                / "models"
                / site.id
                / "1"
                / "model.bundle"
            ),
            bundle_sha256="1" * 64,
            status="active",
            metrics={"feature_names": ["duration_s", "rms"]},
        )
        second = service.claim("worker-b")
        assert second is not None
        assert second.model_version_id is None


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


@pytest.mark.parametrize(
    ("numerator", "denominator"),
    [
        ("1", 1),
        (1.0, 1),
        (True, 1),
        (0, MAX_SQLITE_INTEGER + 1),
    ],
)
def test_heartbeat_requires_strict_sqlite_range_integers_without_partial_change(
    job_database,
    numerator,
    denominator,
):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")

        with pytest.raises(JobValidationError, match="progress"):
            service.heartbeat(job.id, "worker-a", numerator, denominator)

        unchanged = service.get(job.id)
        assert unchanged.status == "running"
        assert (unchanged.progress_numerator, unchanged.progress_denominator) == (0, 0)


@pytest.mark.parametrize(
    ("numerator", "denominator", "completed_progress"),
    [(0, 0, (1, 1)), (0, 1, (1, 1)), (4, 10, (10, 10))],
)
def test_complete_records_full_progress(
    job_database, numerator, denominator, completed_progress
):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        service.heartbeat(job.id, "worker-a", numerator, denominator)

        completed = service.complete(
            job.id, "worker-a", profile.version, "pipeline-v1"
        )

        assert completed.status == "succeeded"
        assert (
            completed.progress_numerator, completed.progress_denominator
        ) == completed_progress
        service.session.expire_all()
        persisted = service.get(job.id)
        assert (
            persisted.progress_numerator, persisted.progress_denominator
        ) == completed_progress


def test_complete_requires_owner_and_exact_versions(job_database, tmp_path: Path):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        artifact = service.artifact_root / "run" / "waterfall.png"
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"png")
        before = service.heartbeat(job.id, "worker-a", 4, 10)

        with pytest.raises(JobConflictError, match="owner"):
            service.complete(
                job.id,
                "worker-b",
                profile.version,
                "pipeline-v1",
                [artifact],
            )
        assert service.get(job.id) == before
        with pytest.raises(JobConflictError, match="profile version"):
            service.complete(
                job.id,
                "worker-a",
                profile.version + 1,
                "pipeline-v1",
                [artifact],
            )
        assert service.get(job.id) == before
        with pytest.raises(JobConflictError, match="pipeline version"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v2",
                [artifact],
            )
        assert service.get(job.id) == before

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
        before = service.heartbeat(job.id, "worker-a", 4, 10)

        with pytest.raises(JobValidationError, match="artifact"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v1",
                [unsafe],
            )

        assert service.get(job.id) == before


def test_complete_rejects_symlinked_artifact_root_even_for_resolved_file(
    job_database,
    tmp_path: Path,
):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        outside = tmp_path / "outside-root"
        outside.mkdir()
        artifact = outside / "waterfall.png"
        artifact.write_bytes(b"png")
        service.artifact_root.rmdir()
        try:
            service.artifact_root.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlink unavailable: {error}")

        with pytest.raises(JobValidationError, match="artifact root"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v1",
                [artifact],
            )

        assert service.get(job.id).status == "running"


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_complete_rejects_junction_artifact_root_even_for_resolved_file(
    job_database,
    tmp_path: Path,
):
    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        outside = tmp_path / "outside-junction"
        outside.mkdir()
        artifact = outside / "waterfall.png"
        artifact.write_bytes(b"png")
        service.artifact_root.rmdir()
        created = subprocess.run(
            [
                "cmd",
                "/c",
                "mklink",
                "/J",
                str(service.artifact_root),
                str(outside),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if created.returncode != 0:
            pytest.skip(f"junction unavailable: {created.stderr or created.stdout}")

        with pytest.raises(JobValidationError, match="artifact root"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v1",
                [artifact],
            )

        assert service.get(job.id).status == "running"


def test_complete_parent_replacement_cannot_redirect_validation(
    job_database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import das_sentinel.services.jobs as job_module

    with seeded_service(job_database) as (service, _, _, profile, source):
        job = service.enqueue(source.id, profile.id, "pipeline-v1")
        service.claim("worker-a")
        parent = service.artifact_root / "run"
        parent.mkdir()
        artifact = parent / "waterfall.png"
        artifact.write_bytes(b"trusted")
        moved = service.artifact_root / "run-original"
        outside = tmp_path / "attacker"
        outside.mkdir()
        (outside / "waterfall.png").write_bytes(b"outside")
        original_open = job_module._open_artifact_component
        replaced = False

        def replace_before_open(path, parent_handle, *, expect_directory):
            nonlocal replaced
            if path == parent and not replaced:
                replaced = True
                parent.rename(moved)
                path.symlink_to(outside, target_is_directory=True)
            return original_open(
                path,
                parent_handle,
                expect_directory=expect_directory,
            )

        monkeypatch.setattr(
            job_module,
            "_open_artifact_component",
            replace_before_open,
        )

        with pytest.raises(JobValidationError, match="artifact"):
            service.complete(
                job.id,
                "worker-a",
                profile.version,
                "pipeline-v1",
                [artifact],
            )

        assert service.get(job.id).status == "running"


def test_fail_retry_limit_and_cancel_transitions(job_database):
    with seeded_service(job_database) as (service, repositories, site, profile, source):
        failed_job = service.enqueue(source.id, profile.id, "pipeline-fail")
        service.claim("worker-a")
        service.heartbeat(failed_job.id, "worker-a", 4, 10)
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
        assert (failed.progress_numerator, failed.progress_denominator) == (4, 10)

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
