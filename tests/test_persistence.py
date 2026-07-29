from __future__ import annotations

import asyncio
import warnings
from datetime import UTC, datetime, timedelta, timezone

import pytest
from das_sentinel.persistence.db import (
    AsyncWriteCoordinator,
    SQLiteRuntimeError,
    check_database_startup,
    create_local_engine,
    session_factory,
    upgrade_database,
    validate_local_sqlite_url,
    validate_sqlite_runtime,
)
from das_sentinel.persistence.models import Base
from das_sentinel.persistence.repositories import (
    Repositories,
    canonical_json,
)
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

EXPECTED_TABLES = {
    "admins",
    "sites",
    "processing_profiles",
    "source_files",
    "jobs",
    "processing_runs",
    "events",
    "detection_evidence",
    "classifications",
    "reviews",
    "model_versions",
    "artifacts",
}


@pytest.fixture
def database_url(tmp_path):
    return f"sqlite+pysqlite:///{tmp_path / 'sentinel.sqlite3'}"


@pytest.fixture
def migrated_engine(database_url):
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    try:
        yield engine
    finally:
        engine.dispose()


def _seed_run(repositories):
    site = repositories.sites.add(
        name="well-a",
        data_path="/data/well-a",
        timezone="Asia/Shanghai",
        cable_type="downhole",
    )
    profile = repositories.profiles.add(
        site_id=site.id,
        version=1,
        payload={"device": "cpu"},
    )
    source = repositories.source_files.add(
        site_id=site.id,
        path="/data/well-a/a.h5",
        fingerprint="f" * 64,
        size_bytes=100,
        mtime_ns=1,
    )
    run = repositories.runs.add(
        source_file_id=source.id,
        processing_profile_id=profile.id,
        pipeline_version="v1",
        requested_backend="cpu",
        actual_backend="cpu",
    )
    return site, profile, source, run


def test_initial_migration_creates_all_tables_and_is_idempotent(tmp_path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'sentinel.sqlite3'}"

    upgrade_database(database_url)
    upgrade_database(database_url)

    engine = create_local_engine(database_url)
    try:
        assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES
    finally:
        engine.dispose()


def test_schema_has_no_foreign_key_dependency_cycle():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(Base.metadata.sorted_tables)

    assert not [
        warning for warning in caught if "unresolvable cycles" in str(warning.message)
    ]


def test_review_does_not_overwrite_machine_classification(migrated_engine):
    engine = migrated_engine
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        _, _, _, run = _seed_run(repositories)
        event = repositories.events.add(
            run_id=run.id,
            event_uid="event-a",
            start_sample=10,
            end_sample=20,
            channel_start=1,
            channel_stop=3,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
        first = repositories.events.add_classification(
            event.id,
            label="unknown",
            decision_score=0.4,
            method="fusion",
        )
        review = repositories.reviews.add(
            event.id,
            label="earthquake",
            note="clear P/S",
        )

        stored = repositories.events.get(event.id)
        assert stored.machine_label == "unknown"
        assert stored.latest_review.id == review.id
        assert stored.latest_review.label == "earthquake"
        assert stored.classifications == [first]


def test_machine_classification_is_latest_append_only_value(migrated_engine):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        _, _, _, run = _seed_run(repositories)
        event = repositories.events.add(
            run_id=run.id,
            event_uid="event-a",
            start_sample=10,
            end_sample=20,
            channel_start=1,
            channel_stop=3,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
        repositories.events.add_classification(
            event.id,
            label="noise",
            decision_score=0.7,
            method="rules",
            created_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
        )
        repositories.events.add_classification(
            event.id,
            label="traffic",
            decision_score=0.8,
            method="model",
            created_at=datetime(2026, 7, 30, 1, tzinfo=UTC),
        )
        repositories.events.add_classification(
            event.id,
            label="blast",
            decision_score=0.6,
            method="import",
            created_at=datetime(2026, 7, 30, tzinfo=UTC),
        )

        stored = repositories.events.get(event.id)
        assert stored.machine_label == "traffic"
        assert len(stored.classifications) == 3


def test_connection_pragmas_are_enforced(migrated_engine):
    with migrated_engine.connect() as connection:
        assert (
            connection.exec_driver_sql("PRAGMA journal_mode").scalar().lower() == "wal"
        )
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
        assert connection.exec_driver_sql("PRAGMA busy_timeout").scalar() == 5000


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://localhost/das",
        "sqlite+pysqlite://server/share/das.sqlite3",
        "sqlite+pysqlite://///server/share/das.sqlite3",
        r"sqlite+pysqlite:///\\server\share\das.sqlite3",
        "sqlite+pysqlite:///file://server/share/das.sqlite3",
        "sqlite+pysqlite:///relative.sqlite3",
    ],
)
def test_remote_or_non_absolute_database_urls_are_rejected(database_url):
    with pytest.raises(ValueError):
        validate_local_sqlite_url(database_url)


def test_sqlite_runtime_gate_requires_fixed_version_or_explicit_patch():
    with pytest.raises(SQLiteRuntimeError, match="3.51.3"):
        validate_sqlite_runtime("3.42.0")

    assert validate_sqlite_runtime("3.51.3") == (3, 51, 3)
    assert validate_sqlite_runtime(
        "3.50.4",
        patched_versions={"3.50.4"},
    ) == (3, 50, 4)


def test_application_startup_check_does_not_silently_accept_old_sqlite(
    migrated_engine,
):
    with pytest.raises(SQLiteRuntimeError):
        check_database_startup(migrated_engine)


def test_async_write_coordinator_exposes_one_shared_lock():
    coordinator = AsyncWriteCoordinator()
    assert isinstance(coordinator.lock, asyncio.Lock)
    assert coordinator.lock is coordinator.lock


def test_profile_payload_is_a_canonical_deep_snapshot(migrated_engine):
    payload = {"z": [2, {"b": 1}], "a": 3}
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name="city-a",
            data_path="/data/city-a",
            timezone="UTC",
            cable_type="urban",
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=payload,
        )
        payload["z"][1]["b"] = 99

        assert profile.payload_json == '{"a":3,"z":[2,{"b":1}]}'
        assert profile.payload == {"a": 3, "z": (2, {"b": 1})}
        with pytest.raises(TypeError):
            profile.payload["a"] = 4
        with pytest.raises(TypeError):
            profile.payload["z"][1]["b"] = 4
        assert not hasattr(repositories.profiles, "update")

    assert canonical_json({"z": 2, "a": 1}) == '{"a":1,"z":2}'


def test_run_key_is_unique_by_source_profile_and_pipeline(migrated_engine):
    sessions = session_factory(migrated_engine)
    with pytest.raises(IntegrityError), sessions.begin() as session:
        repositories = Repositories(session)
        _, profile, source, _ = _seed_run(repositories)
        repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="v1",
            requested_backend="cuda",
            actual_backend="cpu",
        )


def test_foreign_keys_and_critical_indexes_exist(migrated_engine):
    inspector = inspect(migrated_engine)
    run_uniques = {
        tuple(item["column_names"])
        for item in inspector.get_unique_constraints("processing_runs")
    }
    assert (
        "source_file_id",
        "processing_profile_id",
        "pipeline_version",
    ) in run_uniques
    event_indexes = {
        tuple(item["column_names"]) for item in inspector.get_indexes("events")
    }
    assert ("processing_run_id", "occurred_at") in event_indexes

    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO reviews "
                "(event_id, label, created_at) "
                "VALUES ('missing-event', 'noise', "
                "'2026-07-30T00:00:00.000000Z')"
            )
        )


def test_model_bundle_digest_is_validated_and_separate_from_path(
    migrated_engine,
):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        site = repositories.sites.add(
            name="subsea-a",
            data_path="/data/subsea-a",
            timezone="UTC",
            cable_type="subsea",
        )
        with pytest.raises(ValueError, match="SHA-256"):
            repositories.models.add(
                site_id=site.id,
                version="model-1",
                artifact_path="/app-data/models/model-1.skops",
                bundle_sha256="A" * 64,
                status="candidate",
            )
        model = repositories.models.add(
            site_id=site.id,
            version="model-1",
            artifact_path="/app-data/models/model-1.skops",
            bundle_sha256="a" * 64,
            status="candidate",
        )
        assert model.bundle_sha256 == "a" * 64
        assert model.artifact_path.endswith("model-1.skops")


def test_utc_values_are_normalized_and_naive_values_are_rejected(
    migrated_engine,
):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        _, _, _, run = _seed_run(repositories)
        event = repositories.events.add(
            run_id=run.id,
            event_uid="event-a",
            start_sample=10,
            end_sample=20,
            channel_start=1,
            channel_stop=3,
            occurred_at=datetime(
                2026,
                7,
                30,
                8,
                tzinfo=timezone(timedelta(hours=8)),
            ),
        )
        assert event.occurred_at == datetime(2026, 7, 30, tzinfo=UTC)
        with pytest.raises(ValueError, match="timezone-aware"):
            repositories.events.add(
                run_id=run.id,
                event_uid="event-b",
                start_sample=30,
                end_sample=40,
                channel_start=1,
                channel_stop=3,
                occurred_at=datetime(2026, 7, 30),  # noqa: DTZ001
            )


def test_session_can_write_after_a_rolled_back_transaction(migrated_engine):
    sessions = session_factory(migrated_engine)
    session = sessions()
    try:
        with pytest.raises(IntegrityError), session.begin():
            repositories = Repositories(session)
            repositories.sites.add(
                name="duplicate",
                data_path="/data/a",
                timezone="UTC",
                cable_type="urban",
            )
            repositories.sites.add(
                name="duplicate",
                data_path="/data/b",
                timezone="UTC",
                cable_type="urban",
            )

        with session.begin():
            stored = Repositories(session).admins.add(password_hash="hash-b")
            assert stored.id is not None
    finally:
        session.close()


def test_all_repository_groups_offer_transaction_scoped_append_crud(
    migrated_engine,
):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        assert {
            "admins",
            "sites",
            "profiles",
            "source_files",
            "jobs",
            "runs",
            "events",
            "reviews",
            "models",
            "artifacts",
        } <= set(vars(repositories))

        site, profile, source, run = _seed_run(repositories)
        job = repositories.jobs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="v1",
        )
        event = repositories.events.add(
            run_id=run.id,
            event_uid="event-a",
            start_sample=10,
            end_sample=20,
            channel_start=1,
            channel_stop=3,
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
        )
        repositories.events.add_evidence(
            event.id,
            detector="stalta",
            score=0.9,
            payload={"peak_sample": 15},
        )
        artifact = repositories.artifacts.add(
            event_id=event.id,
            kind="waterfall_raw",
            path="/app-data/artifacts/event-a/raw.png",
            media_type="image/png",
            sha256="b" * 64,
        )

        assert repositories.sites.get(site.id) is site
        assert repositories.profiles.get(profile.id) is profile
        assert repositories.source_files.get(source.id) is source
        assert repositories.jobs.get(job.id) is job
        assert repositories.runs.get(run.id) is run
        assert repositories.events.get(event.id) is event
        assert repositories.artifacts.get(artifact.id) is artifact
