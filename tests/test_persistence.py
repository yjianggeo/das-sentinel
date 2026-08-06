from __future__ import annotations

import asyncio
import warnings
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from das_sentinel.persistence.db import (
    AsyncWriteCoordinator,
    SQLiteRuntimeError,
    check_database_startup,
    create_local_engine,
    session_factory,
    upgrade_database,
    validate_local_sqlite_url,
    validate_selected_sqlite_runtime,
    validate_sqlite_runtime,
)
from das_sentinel.persistence.models import Base, ProcessingProfile
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


class _SafeVersionCursor:
    def __init__(self, cursor):
        self._cursor = cursor
        self._sqlite_version_requested = False

    def execute(self, statement, *args, **kwargs):
        self._sqlite_version_requested = statement == "SELECT sqlite_version()"
        return self._cursor.execute(statement, *args, **kwargs)

    def fetchone(self):
        if self._sqlite_version_requested:
            self._sqlite_version_requested = False
            self._cursor.fetchone()
            return ("3.53.3",)
        return self._cursor.fetchone()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _SafeVersionConnection:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self, *args, **kwargs):
        return _SafeVersionCursor(self._connection.cursor(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._connection, name)


class _SafeVersionSQLiteDriver:
    sqlite_version = "3.53.3"

    def __init__(self, driver):
        self._driver = driver

    def connect(self, *args, **kwargs):
        return _SafeVersionConnection(self._driver.connect(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._driver, name)


@pytest.fixture(autouse=True)
def safe_sqlite_driver_for_migration_tests(monkeypatch):
    from das_sentinel.persistence import db

    selected = db._sqlite_driver()
    monkeypatch.setattr(db, "_sqlite_driver", lambda: _SafeVersionSQLiteDriver(selected))


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


def _seed_reviewed_event(repositories):
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
    classification = repositories.events.add_classification(
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
    return event, classification, review


def test_initial_migration_creates_all_tables_and_is_idempotent(tmp_path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'sentinel.sqlite3'}"

    upgrade_database(database_url)
    upgrade_database(database_url)

    engine = create_local_engine(database_url)
    try:
        assert set(inspect(engine).get_table_names()) >= EXPECTED_TABLES
        with engine.connect() as connection:
            triggers = set(
                connection.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                ).scalars()
            )
        assert triggers >= {
            "classifications_append_only_update",
            "classifications_append_only_delete",
            "reviews_append_only_update",
            "reviews_append_only_delete",
            "processing_profiles_immutable",
        }
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


@pytest.mark.parametrize("attribute", ["label", "decision_score"])
def test_orm_cannot_update_machine_classification(
    migrated_engine,
    attribute,
):
    sessions = session_factory(migrated_engine)
    session = sessions()
    try:
        with session.begin():
            _, classification, _ = _seed_reviewed_event(Repositories(session))
            classification_id = classification.id

        with pytest.raises(IntegrityError), session.begin():
            stored = session.get(type(classification), classification_id)
            setattr(stored, attribute, "noise" if attribute == "label" else 0.9)
            session.flush()

        with session.begin():
            stored = session.get(type(classification), classification_id)
            assert stored.label == "unknown"
            assert stored.decision_score == 0.4
    finally:
        session.close()


def test_orm_cannot_update_review_or_remove_audit_rows(migrated_engine):
    sessions = session_factory(migrated_engine)
    session = sessions()
    try:
        with session.begin():
            event, classification, review = _seed_reviewed_event(Repositories(session))
            event_id = event.id
            classification_id = classification.id
            review_id = review.id

        with pytest.raises(IntegrityError), session.begin():
            stored_review = session.get(type(review), review_id)
            stored_review.note = "rewritten"
            session.flush()

        with pytest.raises(IntegrityError), session.begin():
            stored_event = session.get(type(event), event_id)
            stored_classification = session.get(type(classification), classification_id)
            stored_event.classifications.remove(stored_classification)
            session.flush()

        with pytest.raises(IntegrityError), session.begin():
            stored_event = session.get(type(event), event_id)
            stored_review = session.get(type(review), review_id)
            stored_event.reviews.remove(stored_review)
            session.flush()

        with session.begin():
            stored_event = Repositories(session).events.get(event_id)
            assert stored_event.machine_label == "unknown"
            assert stored_event.latest_review.note == "clear P/S"
    finally:
        session.close()


@pytest.mark.parametrize("table", ["classifications", "reviews"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
def test_direct_sql_cannot_rewrite_or_delete_audit_rows(
    migrated_engine,
    table,
    operation,
):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        _, classification, review = _seed_reviewed_event(Repositories(session))
        row_id = classification.id if table == "classifications" else review.id

    statement = (
        f"UPDATE {table} SET created_at = created_at WHERE id = :row_id"
        if operation == "UPDATE"
        else f"DELETE FROM {table} WHERE id = :row_id"
    )
    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(text(statement), {"row_id": row_id})

    with migrated_engine.connect() as connection:
        assert (
            connection.execute(
                text(f"SELECT count(*) FROM {table} WHERE id = :row_id"),
                {"row_id": row_id},
            ).scalar_one()
            == 1
        )


def test_deleting_event_cannot_silently_erase_audit_rows(migrated_engine):
    sessions = session_factory(migrated_engine)
    session = sessions()
    try:
        with session.begin():
            event, _, _ = _seed_reviewed_event(Repositories(session))
            event_id = event.id

        with pytest.raises(IntegrityError), session.begin():
            stored_event = Repositories(session).events.get(event_id)
            session.delete(stored_event)
            session.flush()

        with session.begin():
            stored_event = Repositories(session).events.get(event_id)
            assert stored_event is not None
            assert stored_event.machine_label == "unknown"
            assert stored_event.latest_review.label == "earthquake"
    finally:
        session.close()


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


def test_selected_sqlite_driver_is_the_runtime_being_validated(monkeypatch):
    selected = SimpleNamespace(sqlite_version="3.53.3")
    monkeypatch.setattr("das_sentinel.persistence.db._sqlite_driver", lambda: selected)
    assert validate_selected_sqlite_runtime() == (3, 53, 3)


def test_selected_sqlite_driver_rejects_unsafe_version(monkeypatch):
    selected = SimpleNamespace(sqlite_version="3.51.1")
    monkeypatch.setattr("das_sentinel.persistence.db._sqlite_driver", lambda: selected)
    with pytest.raises(SQLiteRuntimeError, match="3.51.3"):
        validate_selected_sqlite_runtime()


def test_upgrade_rejects_unsafe_driver_before_alembic(monkeypatch, tmp_path):
    selected = SimpleNamespace(sqlite_version="3.51.1")
    migration_called = False

    def record_upgrade(*args, **kwargs):
        nonlocal migration_called
        migration_called = True

    monkeypatch.setattr("das_sentinel.persistence.db._sqlite_driver", lambda: selected)
    monkeypatch.setattr("das_sentinel.persistence.db.command.upgrade", record_upgrade)
    with pytest.raises(SQLiteRuntimeError):
        upgrade_database(f"sqlite+pysqlite:///{tmp_path / 'unsafe.sqlite3'}")
    assert migration_called is False


class _StubResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _StubConnection:
    def __init__(self, sqlite_version):
        self._sqlite_version = sqlite_version

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def exec_driver_sql(self, statement):
        normalized = statement.strip().lower()
        if "sqlite_version" in normalized:
            return _StubResult(self._sqlite_version)
        if "journal_mode" in normalized:
            return _StubResult("wal")
        if "foreign_keys" in normalized:
            return _StubResult(1)
        if "busy_timeout" in normalized:
            return _StubResult(5000)
        raise AssertionError(f"unexpected statement: {statement}")


class _StubEngine:
    def __init__(self, sqlite_version):
        self._sqlite_version = sqlite_version

    def connect(self):
        return _StubConnection(self._sqlite_version)


def test_application_startup_check_does_not_silently_accept_old_sqlite():
    with pytest.raises(SQLiteRuntimeError, match="3.51.3"):
        check_database_startup(_StubEngine("3.42.0"))


def test_application_startup_check_accepts_current_runtime(migrated_engine):
    report = check_database_startup(migrated_engine)
    assert report.sqlite_version >= (3, 51, 3)


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


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"nested": [1, {"value": float("inf")}]},
        {"nested": [float("-inf")]},
    ],
)
def test_canonical_json_rejects_non_finite_numbers(payload):
    with pytest.raises(ValueError, match="JSON"):
        canonical_json(payload)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("version", 99),
        ("payload_json", '{"device":"cuda"}'),
    ],
)
def test_direct_sql_cannot_change_profile_identity_or_payload(
    migrated_engine,
    column,
    value,
):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        _, profile, _, _ = _seed_run(Repositories(session))
        profile_id = profile.id

    statement = text(
        f"UPDATE processing_profiles SET {column} = :value WHERE id = :profile_id"
    )
    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            statement,
            {"value": value, "profile_id": profile_id},
        )


def test_direct_sql_cannot_move_profile_to_another_valid_site(
    migrated_engine,
):
    sessions = session_factory(migrated_engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        _, profile, _, _ = _seed_run(repositories)
        other_site = repositories.sites.add(
            name="other-site",
            data_path="/data/other",
            timezone="UTC",
            cable_type="urban",
        )
        profile_id = profile.id
        other_site_id = other_site.id

    with pytest.raises(IntegrityError), migrated_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE processing_profiles SET site_id = :site_id "
                "WHERE id = :profile_id"
            ),
            {"site_id": other_site_id, "profile_id": profile_id},
        )


def test_orm_cannot_change_profile_payload_but_can_change_current_flag(
    migrated_engine,
):
    sessions = session_factory(migrated_engine)
    session = sessions()
    try:
        with session.begin():
            _, profile, _, _ = _seed_run(Repositories(session))
            profile_id = profile.id

        with pytest.raises(IntegrityError), session.begin():
            stored = session.get(ProcessingProfile, profile_id)
            stored.payload_json = '{"device":"cuda"}'
            session.flush()

        with session.begin():
            stored = session.get(ProcessingProfile, profile_id)
            stored.is_current = True
            session.flush()

        with session.begin():
            stored = session.get(ProcessingProfile, profile_id)
            assert stored.payload_json == '{"device":"cpu"}'
            assert stored.is_current is True
    finally:
        session.close()


def test_current_profile_switch_orders_updates_and_rolls_back_as_one_unit(
    migrated_engine,
):
    sessions = session_factory(migrated_engine)
    session = sessions()
    try:
        with session.begin():
            repositories = Repositories(session)
            site = repositories.sites.add(
                name="ordered-site",
                data_path="/data/ordered",
                timezone="UTC",
                cable_type="urban",
            )
            old = ProcessingProfile(
                id="z" * 36,
                site_id=site.id,
                version=1,
                payload_json='{"device":"cpu"}',
                is_current=True,
            )
            target = ProcessingProfile(
                id="a" * 36,
                site_id=site.id,
                version=2,
                payload_json='{"device":"cpu"}',
                is_current=False,
            )
            session.add_all([old, target])
            session.flush()
            site_id = site.id

        with (
            pytest.raises(RuntimeError, match="force rollback"),
            session.begin(),
        ):
            Repositories(session).sites.set_current_profile(site_id, target.id)
            raise RuntimeError("force rollback")

        with session.begin():
            assert session.get(ProcessingProfile, old.id).is_current is True
            assert session.get(ProcessingProfile, target.id).is_current is False

        with session.begin():
            Repositories(session).sites.set_current_profile(site_id, target.id)

        with session.begin():
            assert session.get(ProcessingProfile, old.id).is_current is False
            assert session.get(ProcessingProfile, target.id).is_current is True
    finally:
        session.close()


def test_programmatic_migration_url_beats_ambient_environment(
    tmp_path,
    monkeypatch,
):
    ambient_url = f"sqlite+pysqlite:///{tmp_path / 'ambient.sqlite3'}"
    explicit_url = f"sqlite+pysqlite:///{tmp_path / 'explicit.sqlite3'}"
    monkeypatch.setenv("DAS_DATABASE_URL", ambient_url)

    upgrade_database(explicit_url)

    ambient_engine = create_local_engine(ambient_url)
    explicit_engine = create_local_engine(explicit_url)
    try:
        assert "events" not in inspect(ambient_engine).get_table_names()
        assert set(inspect(explicit_engine).get_table_names()) >= EXPECTED_TABLES
    finally:
        ambient_engine.dispose()
        explicit_engine.dispose()


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
