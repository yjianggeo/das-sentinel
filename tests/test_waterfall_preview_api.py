from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.domain import DasBlock
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import SourceFile
from das_sentinel.services.discovery import fingerprint
from fastapi.testclient import TestClient
from sqlalchemy import func, select

PASSWORD = "correct horse"


def _settings(tmp_path: Path) -> Settings:
    data_root = tmp_path / "raw"
    app_data_dir = tmp_path / "app-data"
    data_root.mkdir()
    app_data_dir.mkdir()
    return Settings(
        data_root=data_root,
        app_data_dir=app_data_dir,
        database_url=f"sqlite+pysqlite:///{app_data_dir / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token="w" * 32,
        secure_cookies=False,
    )


def _write_source(path: Path) -> None:
    sample_rate_hz = 100.0
    sample_count = 200
    samples = np.arange(sample_count, dtype=np.float32) / sample_rate_hz
    channels = np.arange(5, dtype=np.float32)
    data = np.sin(2 * np.pi * 8 * samples[:, None] + channels[None, :])
    data[:, 2] = 0.0
    start = datetime(2026, 7, 26, tzinfo=UTC)
    timestamps = start.timestamp() * 1_000_000 + np.arange(sample_count) * 10_000
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=data.astype(np.float32))
        raw.create_dataset("RawDataTime", data=timestamps.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


@pytest.fixture
def waterfall_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    source_path = settings.data_root / "input.h5"
    _write_source(source_path)
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site = repositories.sites.add(
            name="preview-site",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="urban",
        )
        run_profile = repositories.profiles.add(
            site_id=site.id,
            version=4,
            payload=ProcessingProfile(
                device="cpu", stalta_freq_low_hz=3.5, stalta_freq_high_hz=27.25
            ).model_dump(mode="json"),
            is_current=False,
        )
        repositories.profiles.add(
            site_id=site.id,
            version=5,
            payload=ProcessingProfile(
                device="cpu", stalta_freq_low_hz=1.0, stalta_freq_high_hz=2.0
            ).model_dump(mode="json"),
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path=str(source_path.resolve()),
            fingerprint=fingerprint(source_path),
            size_bytes=source_path.stat().st_size,
            mtime_ns=source_path.stat().st_mtime_ns,
        )
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=run_profile.id,
            pipeline_version="pipeline-v1",
            requested_backend="cuda",
            actual_backend="cpu",
            status="succeeded",
        )
        event = repositories.events.add(
            run_id=run.id,
            event_uid="preview-event",
            start_sample=20,
            end_sample=40,
            channel_start=1,
            channel_stop=4,
            occurred_at=datetime(2026, 7, 26, 0, 0, 0, 200_000, tzinfo=UTC),
        )
        repositories.events.add_evidence(
            event.id,
            detector="stalta",
            score=0.9,
            payload={
                "start_sample": 20,
                "end_sample": 40,
                "peak_sample": 30,
                "channel_start": 1,
                "channel_stop": 4,
            },
        )
        ids = {"event": event.id, "source": source.id}
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        assert login.status_code == 204
        yield client, settings, sessions, login.headers["x-csrf-token"], ids, source_path
    engine.dispose()


def test_waterfall_routes_require_admin_and_preview_requires_csrf(waterfall_api):
    client, _, _, csrf_token, ids, _ = waterfall_api
    client.cookies.clear()

    assert client.get(f"/api/v1/events/{ids['event']}/waterfall-options").status_code == 401
    assert client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
    ).status_code == 401

    assert client.post("/api/v1/session", json={"password": PASSWORD}).status_code == 204
    assert client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
    ).status_code == 403
    assert csrf_token


def test_waterfall_options_use_historical_run_profile_and_metadata(waterfall_api):
    client, _, _, _, ids, _ = waterfall_api

    response = client.get(f"/api/v1/events/{ids['event']}/waterfall-options")

    assert response.status_code == 200
    assert response.json() == {
        "automatic_distance_start_m": 12.0,
        "automatic_distance_stop_m": 20.0,
        "available_distance_start_m": 8.0,
        "available_distance_stop_m": 24.0,
        "primary_detector": "stalta",
        "primary_detector_display": "STA/LTA",
        "filter_low_hz": 3.5,
        "filter_high_hz": 27.25,
        "distance_unit": "m",
    }
    assert client.get("/api/v1/events/missing/waterfall-options").status_code == 404


@pytest.mark.parametrize("kind", ["missing", "outside_root", "mismatch"])
def test_waterfall_options_reject_conflicted_source_without_disclosing_path(
    waterfall_api, tmp_path: Path, kind: str
):
    client, _, sessions, _, ids, source_path = waterfall_api
    outside = tmp_path / "outside.h5"
    if kind == "missing":
        source_path.unlink()
    with sessions.begin() as session:
        source = session.get(SourceFile, ids["source"])
        assert source is not None
        if kind == "outside_root":
            _write_source(outside)
            source.path = str(outside.resolve())
        elif kind == "mismatch":
            source.fingerprint = "0" * 64

    response = client.get(f"/api/v1/events/{ids['event']}/waterfall-options")

    assert response.status_code == 409
    assert str(source_path) not in response.text
    assert str(outside) not in response.text


def test_waterfall_preview_returns_two_pngs_after_one_read_without_writes(waterfall_api):
    client, settings, sessions, csrf_token, ids, _ = waterfall_api
    service = client.app.state.waterfall_preview_service
    original_reader = service.reader

    class CountingReader:
        def __init__(self):
            self.read_block_calls = 0

        def read_metadata(self, path):
            return original_reader.read_metadata(path)

        def read_block(self, path, start, stop, channels):
            self.read_block_calls += 1
            return original_reader.read_block(path, start, stop, channels)

    counting_reader = CountingReader()
    service.reader = counting_reader
    with sessions() as session:
        before_rows = {
            table.__tablename__: session.scalar(select(func.count()).select_from(table))
            for table in (SourceFile,)
        }
    before_artifacts = sorted(
        path.relative_to(settings.app_data_dir).as_posix()
        for path in settings.app_data_dir.rglob("*")
    )

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["raw_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert base64.b64decode(body["filtered_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert body["channel_count"] == 2
    assert counting_reader.read_block_calls == 1
    with sessions() as session:
        assert before_rows == {
            table.__tablename__: session.scalar(select(func.count()).select_from(table))
            for table in (SourceFile,)
        }
    assert before_artifacts == sorted(
        path.relative_to(settings.app_data_dir).as_posix()
        for path in settings.app_data_dir.rglob("*")
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"distance_start_m": 20.0, "distance_stop_m": 12.0},
        {"distance_start_m": 8.0, "distance_stop_m": 12.0},
        {"distance_start_m": 12.0, "distance_stop_m": 24.0},
    ],
)
def test_waterfall_preview_rejects_invalid_or_non_strictly_interior_ranges(
    waterfall_api, payload
):
    client, _, _, csrf_token, ids, _ = waterfall_api

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json=payload,
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


def test_waterfall_preview_rejects_non_finite_range(waterfall_api):
    client, _, _, csrf_token, ids, _ = waterfall_api

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        content=b'{"distance_start_m":NaN,"distance_stop_m":20.0}',
        headers={"Content-Type": "application/json", "X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


def test_waterfall_preview_rejects_no_selected_or_usable_channels(
    waterfall_api, monkeypatch: pytest.MonkeyPatch
):
    client, _, _, csrf_token, ids, _ = waterfall_api
    service = client.app.state.waterfall_preview_service
    reader = service.reader
    read_block = reader.read_block

    assert client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 13.0, "distance_stop_m": 15.0},
        headers={"X-CSRF-Token": csrf_token},
    ).status_code == 422

    def all_bad_block(path, start, stop, channels):
        block = read_block(path, start, stop, channels)
        return DasBlock(np.zeros_like(block.data), block.metadata, block.sample_offset)

    monkeypatch.setattr(reader, "read_block", all_bad_block)
    assert client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    ).status_code == 422


def test_waterfall_preview_rechecks_fingerprint_after_render(
    waterfall_api, monkeypatch: pytest.MonkeyPatch
):
    import das_sentinel.services.waterfall_previews as previews

    client, _, _, csrf_token, ids, source_path = waterfall_api
    expected = fingerprint(source_path)
    calls = 0

    def changed_after_initial_check(path: Path) -> str:
        nonlocal calls
        calls += 1
        return expected if calls == 1 else "0" * 64

    monkeypatch.setattr(previews, "fingerprint", changed_after_initial_check)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 409
    assert str(source_path) not in response.text
