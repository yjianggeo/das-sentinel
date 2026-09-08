from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.domain import DasBlock, EventRecord
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import (
    Base,
    Event,
    ProcessingRun,
    SourceFile,
)
from das_sentinel.services.discovery import fingerprint
from das_sentinel.services.waterfall_previews import WaterfallPreviewService
from fastapi.testclient import TestClient
from sqlalchemy import select

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


def _write_source(path: Path, sample_count: int = 200) -> None:
    sample_rate_hz = 100.0
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
                device="cpu",
                artifact_pad_seconds=0.05,
                stalta_freq_low_hz=3.5,
                stalta_freq_high_hz=27.25,
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


def test_waterfall_options_use_candidate_distance_extrema_for_nonmonotonic_metadata(
    waterfall_api, monkeypatch: pytest.MonkeyPatch
):
    client, _, _, _, ids, _ = waterfall_api
    reader = client.app.state.waterfall_preview_service.reader
    original_metadata = reader.read_metadata

    def nonmonotonic_metadata(path):
        metadata = original_metadata(path)
        return replace(
            metadata,
            channel_distance_m=np.array([8.0, 20.0, 12.0, 16.0, 24.0]),
        )

    monkeypatch.setattr(reader, "read_metadata", nonmonotonic_metadata)
    response = client.get(f"/api/v1/events/{ids['event']}/waterfall-options")

    assert response.status_code == 200
    assert response.json()["automatic_distance_start_m"] == 12.0
    assert response.json()["automatic_distance_stop_m"] == 20.0


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


@pytest.mark.parametrize("automatic", [False, True])
def test_waterfall_preview_returns_three_pngs_after_one_read_without_writes(
    waterfall_api, automatic: bool
):
    client, settings, sessions, csrf_token, ids, _ = waterfall_api
    service = client.app.state.waterfall_preview_service
    original_reader = service.reader

    class CountingReader:
        def __init__(self):
            self.read_block_calls = 0

        def read_metadata(self, path):
            return original_reader.read_metadata(path)

        def sample_count(self, path):
            return original_reader.sample_count(path)

        def read_block(self, path, start, stop, channels):
            self.read_block_calls += 1
            return original_reader.read_block(path, start, stop, channels)

    counting_reader = CountingReader()
    service.reader = counting_reader
    original_artifact = settings.app_data_dir / "waveform.png"
    original_artifact.write_bytes(b"registered original waveform")
    with sessions.begin() as session:
        Repositories(session).artifacts.add(
            event_id=ids["event"], kind="waveform", path=str(original_artifact),
            media_type="image/png",
            sha256=hashlib.sha256(original_artifact.read_bytes()).hexdigest(),
        )
    with sessions() as session:
        before_rows = {
            table.name: session.execute(select(table)).all()
            for table in Base.metadata.sorted_tables
        }
    before_data_root = _tree_snapshot(settings.data_root)
    before_artifacts = sorted(
        path.relative_to(settings.app_data_dir).as_posix()
        for path in settings.app_data_dir.rglob("*")
    )
    before_artifact_bytes = original_artifact.read_bytes()

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={} if automatic else {"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    body = response.json()
    assert base64.b64decode(body["raw_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert base64.b64decode(body["filtered_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert base64.b64decode(body["waveform_png_base64"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert body["channel_count"] == 2
    assert counting_reader.read_block_calls == 1
    with sessions() as session:
        assert before_rows == {
            table.name: session.execute(select(table)).all()
            for table in Base.metadata.sorted_tables
        }
    assert before_data_root == _tree_snapshot(settings.data_root)
    assert original_artifact.read_bytes() == before_artifact_bytes
    assert before_artifacts == sorted(
        path.relative_to(settings.app_data_dir).as_posix()
        for path in settings.app_data_dir.rglob("*")
    )


def test_waterfall_preview_rejects_reversed_range(waterfall_api):
    client, _, _, csrf_token, ids, _ = waterfall_api

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 20.0, "distance_stop_m": 12.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422


def test_waterfall_preview_accepts_range_outside_automatic_but_inside_available(
    waterfall_api,
):
    """Regresses validation against event bounds instead of source availability."""
    client, _, _, csrf_token, ids, _ = waterfall_api

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 8.0, "distance_stop_m": 12.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.json()["distance_start_m"] == 8.0
    assert response.json()["distance_stop_m"] == 12.0


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ((11.0, 21.0), (12.0, 20.0, 2)),
        ((11.0, 13.0), (12.0, 12.0, 1)),
    ],
)
def test_waterfall_preview_reports_actual_usable_distance_extrema(
    waterfall_api, requested, expected
):
    """Regresses echoing off-grid request bounds rather than selected channels."""
    client, _, _, csrf_token, ids, _ = waterfall_api

    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": requested[0], "distance_stop_m": requested[1]},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["distance_start_m"], body["distance_stop_m"], body["channel_count"]) == expected


@pytest.mark.parametrize(
    ("event_bounds", "expected_read_bounds"),
    [((2, 10), (0, 200)), ((190, 198), (0, 200))],
)
def test_waterfall_preview_reads_full_short_source_at_both_event_edges(
    waterfall_api,
    monkeypatch: pytest.MonkeyPatch,
    event_bounds,
    expected_read_bounds,
):
    """Regresses reading the bare event interval or passing padded bounds past EOF."""
    client, _, sessions, csrf_token, ids, _ = waterfall_api
    reader = client.app.state.waterfall_preview_service.reader
    original_read_block = reader.read_block
    captured: list[tuple[int, int]] = []
    with sessions.begin() as session:
        event = session.get(Event, ids["event"])
        event.start_sample, event.end_sample = event_bounds
        event.occurred_at = datetime(2026, 7, 26, tzinfo=UTC) + timedelta(
            seconds=event.start_sample / 100.0
        )
        payload = json.loads(event.evidence[0].payload_json)
        payload.update(
            start_sample=event_bounds[0],
            end_sample=event_bounds[1],
            peak_sample=(event_bounds[0] + event_bounds[1]) // 2,
        )
        event.evidence[0].payload_json = json.dumps(payload)

    def capture_read(path, start, stop, channels):
        captured.append((start, stop))
        return original_read_block(path, start, stop, channels)

    monkeypatch.setattr(reader, "read_block", capture_read)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert captured == [expected_read_bounds]


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
    rendered = []
    original_waterfall = previews.render_waterfall_png
    original_waveform = previews.render_waveform_png

    def waterfall(*args, **kwargs):
        result = original_waterfall(*args, **kwargs)
        rendered.append("filtered" if kwargs["filtered"] else "raw")
        return result

    def waveform(*args, **kwargs):
        result = original_waveform(*args, **kwargs)
        rendered.append("waveform")
        return result

    def changed_after_initial_check(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls > 1:
            assert rendered == ["raw", "filtered", "waveform"]
        return expected if calls == 1 else "0" * 64

    monkeypatch.setattr(previews, "fingerprint", changed_after_initial_check)
    monkeypatch.setattr(previews, "render_waterfall_png", waterfall)
    monkeypatch.setattr(previews, "render_waveform_png", waveform)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 409
    assert str(source_path) not in response.text


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("source disappeared"),
        OSError("read failed"),
        ValueError("source structure is inconsistent"),
    ],
)
def test_waterfall_preview_controls_read_block_io_errors(
    waterfall_api, monkeypatch: pytest.MonkeyPatch, error: Exception
):
    client, _, _, csrf_token, ids, source_path = waterfall_api
    reader = client.app.state.waterfall_preview_service.reader

    def fail_read_block(path, start, stop, channels):
        raise error

    monkeypatch.setattr(reader, "read_block", fail_read_block)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 409
    assert str(source_path) not in response.text


@pytest.mark.parametrize("automatic", [False, True])
def test_waterfall_preview_rejects_oversized_window_before_read(waterfall_api, automatic):
    client, _, _, csrf_token, ids, _ = waterfall_api
    service = client.app.state.waterfall_preview_service
    reader = service.reader
    calls = 0
    original_binding = service._binding

    def padded_binding(session, event_id):
        binding = original_binding(session, event_id)
        return replace(
            binding,
            profile=binding.profile.model_copy(update={"artifact_pad_seconds": 400.0}),
        )

    def count_read_block(path, start, stop, channels):
        nonlocal calls
        calls += 1
        raise AssertionError("oversized preview must not read source data")

    reader.sample_count = lambda path: 50_000
    reader.read_block = count_read_block
    service._binding = padded_binding
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={} if automatic else {"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422
    assert calls == 0


def test_waterfall_preview_allows_default_padding_around_short_1khz_event():
    """Regresses rejecting 5 s padding plus a short event at 1 kHz."""
    WaterfallPreviewService._validate_read_bounds(
        14_398,
        24_643,
        slice(30, 61),
    )


def test_waterfall_preview_parses_evidence_once_per_request(
    waterfall_api, monkeypatch: pytest.MonkeyPatch
):
    """Regresses duplicate evidence deserialization in binding and range lookup."""
    client, _, _, _, ids, _ = waterfall_api
    service = client.app.state.waterfall_preview_service
    original = service._candidates
    calls = 0

    def count_candidates(event):
        nonlocal calls
        calls += 1
        return original(event)

    monkeypatch.setattr(service, "_candidates", count_candidates)
    response = client.get(f"/api/v1/events/{ids['event']}/waterfall-options")

    assert response.status_code == 200
    assert calls == 1


@pytest.mark.parametrize(
    "corrupt",
    [
        "profile_version",
        "event_time",
        "event_channel",
        "evidence_peak",
        "occurred_at",
    ],
)
def test_waterfall_options_reject_persisted_binding_corruption(
    waterfall_api, corrupt: str
):
    """Regresses serving previews from mutually inconsistent persisted rows."""
    client, _, sessions, _, ids, source_path = waterfall_api
    with sessions.begin() as session:
        event = session.get(Event, ids["event"])
        assert event is not None
        run = session.get(ProcessingRun, event.processing_run_id)
        assert run is not None
        if corrupt == "profile_version":
            run.profile_version += 1
        elif corrupt == "event_time":
            event.start_sample += 1
        elif corrupt == "event_channel":
            event.channel_start = 0
        elif corrupt == "evidence_peak":
            payload = json.loads(event.evidence[0].payload_json)
            payload["peak_sample"] = payload["end_sample"]
            event.evidence[0].payload_json = json.dumps(payload)
        else:
            event.occurred_at += timedelta(seconds=1)

    response = client.get(f"/api/v1/events/{ids['event']}/waterfall-options")

    assert response.status_code == 409
    assert str(source_path) not in response.text


def test_waterfall_options_maps_missing_bound_profile_to_conflict(waterfall_api):
    """Only an absent event is a 404; missing event dependencies are conflicts."""
    client, _, sessions, _, ids, source_path = waterfall_api
    engine = sessions.kw["bind"]
    with sessions() as session:
        event = session.get(Event, ids["event"])
        run = session.get(ProcessingRun, event.processing_run_id)
        profile_id = run.processing_profile_id
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.exec_driver_sql(
            "DELETE FROM processing_profiles WHERE id = ?", (profile_id,)
        )
        connection.commit()

    response = client.get(f"/api/v1/events/{ids['event']}/waterfall-options")

    assert response.status_code == 409
    assert str(source_path) not in response.text


def test_waterfall_preview_uses_bound_model_version_in_plot_provenance(
    waterfall_api, monkeypatch: pytest.MonkeyPatch
):
    """Regresses labelling model-backed runs as rules-only."""
    import das_sentinel.services.waterfall_previews as previews

    client, _, sessions, csrf_token, ids, _ = waterfall_api
    with sessions.begin() as session:
        event = session.get(Event, ids["event"])
        run = session.get(ProcessingRun, event.processing_run_id)
        source = session.get(SourceFile, ids["source"])
        model = Repositories(session).models.add(
            site_id=source.site_id,
            version="model-7.2",
            artifact_path="unused.zip",
            bundle_sha256="a" * 64,
            status="active",
        )
        run.model_version_id = model.id
    observed: list[str] = []

    def capture_render(values, dates, distances, context, **kwargs):
        observed.append(context.model_version)
        return b"\x89PNG\r\n\x1a\npreview"

    monkeypatch.setattr(previews, "render_waterfall_png", capture_render)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert observed == ["model-7.2", "model-7.2"]


@pytest.mark.parametrize("automatic", [False, True])
def test_waterfall_preview_rejects_oversized_channel_read_before_read(
    waterfall_api, monkeypatch: pytest.MonkeyPatch, automatic
):
    client, _, sessions, csrf_token, ids, _ = waterfall_api
    reader = client.app.state.waterfall_preview_service.reader
    original_metadata = reader.read_metadata
    calls = 0
    with sessions.begin() as session:
        event = session.get(Event, ids["event"])
        assert event is not None
        event.channel_start = 0
        event.channel_stop = 257
        evidence = event.evidence[0]
        evidence.payload_json = json.dumps(
            {
                "start_sample": 20,
                "end_sample": 40,
                "peak_sample": 30,
                "channel_start": 0,
                "channel_stop": 257,
            }
        )

    def wide_metadata(path):
        metadata = original_metadata(path)
        distances = np.full(257, 100.0)
        distances[0], distances[-1] = 0.0, 1.0
        return replace(metadata, channel_distance_m=distances)

    def count_read_block(path, start, stop, channels):
        nonlocal calls
        calls += 1
        raise AssertionError("oversized preview must not read source data")

    monkeypatch.setattr(reader, "read_metadata", wide_metadata)
    monkeypatch.setattr(reader, "read_block", count_read_block)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={} if automatic else {"distance_start_m": 0.0, "distance_stop_m": 1.0},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 422
    assert calls == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"distance_start_m": 12.0}, {"distance_stop_m": 20.0},
        {"distance_start_m": None}, {"distance_stop_m": None},
        {"distance_start_m": None, "distance_stop_m": None},
        {"distance_start_m": None, "distance_stop_m": 20.0},
        {"distance_start_m": 12.0, "distance_stop_m": None},
        {"distance_start_m": "12", "distance_stop_m": 20.0},
        {"distance_start_m": True, "distance_stop_m": 20.0},
    ],
)
def test_waterfall_preview_rejects_partial_null_or_nonnumeric_range(waterfall_api, payload):
    client, _, _, csrf_token, ids, _ = waterfall_api
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview", json=payload,
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    ("source_count", "bounds", "pad", "expected", "short"),
    [
        (200, (20, 40), 0.05, (0, 200), True),
        (1000, (2, 10), 0.05, (0, 300), False),
        (1000, (990, 998), 0.05, (700, 1000), False),
        (1000, (450, 470), 0.05, (310, 610), False),
        (1000, (450, 470), 2.0, (250, 670), False),
    ],
)
def test_preview_three_plots_share_real_window_median_and_absolute_markers(
    waterfall_api, monkeypatch, source_count, bounds, pad, expected, short
):
    import das_sentinel.services.waterfall_previews as previews
    import matplotlib.dates as mdates
    from das_sentinel import artifacts

    client, _, sessions, csrf_token, ids, source_path = waterfall_api
    _write_source(source_path, source_count)
    with sessions.begin() as session:
        event = session.get(Event, ids["event"])
        event.start_sample, event.end_sample = bounds
        event.occurred_at = datetime(2026, 7, 26, tzinfo=UTC) + timedelta(seconds=bounds[0] / 100)
        run = session.get(ProcessingRun, event.processing_run_id)
        source = session.get(SourceFile, ids["source"])
        source.fingerprint = run.source_fingerprint = fingerprint(source_path)
        payload = json.loads(event.evidence[0].payload_json)
        payload.update(start_sample=bounds[0], end_sample=bounds[1], peak_sample=bounds[0] + 3)
        event.evidence[0].payload_json = json.dumps(payload)
        event.evidence[0].score = 0.5
        for peak in (bounds[0] + 5, bounds[0] + 4):
            Repositories(session).events.add_evidence(
                event.id, detector="stalta", score=0.95,
                payload={**payload, "peak_sample": peak},
            )
    service = client.app.state.waterfall_preview_service
    original_binding = service._binding
    original_read = service.reader.read_block
    original_preprocess = previews.preprocess_for_waterfall
    original_png = artifacts._png
    read_bounds, processed, figures = [], [], []

    def padded_binding(session, event_id):
        binding = original_binding(session, event_id)
        return replace(binding, profile=binding.profile.model_copy(update={"artifact_pad_seconds": pad}))

    def capture_read(path, start, stop, channels):
        read_bounds.append((start, stop, channels))
        return original_read(path, start, stop, channels)

    def capture_preprocess(*args):
        result = original_preprocess(*args)
        processed.append(result)
        return result

    def capture_png(figure):
        figures.append(figure)
        return original_png(figure)

    monkeypatch.setattr(service, "_binding", padded_binding)
    monkeypatch.setattr(service.reader, "read_block", capture_read)
    monkeypatch.setattr(previews, "preprocess_for_waterfall", capture_preprocess)
    monkeypatch.setattr(artifacts, "_png", capture_png)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    assert read_bounds == [(*expected, slice(1, 4))]
    assert len(processed) == 1
    assert len(figures) == 3
    duration = (expected[1] - expected[0]) / 100
    for figure in figures:
        title = " ".join(figure.axes[0].get_title().split())
        assert f"Display samples {expected[0]}:{expected[1]} | duration {duration:g} s" in title
        assert ("Source shorter than 3 s; showing actual duration" in title) is short
        assert "Artificial Distance 12–20 m" in title
    waveform = figures[2].axes[0]
    np.testing.assert_allclose(
        waveform.lines[0].get_ydata(), np.median(processed[0].data[:, [0, 2]], axis=1)
    )
    markers = {line.get_label(): line.get_xdata()[0] for line in waveform.lines[1:]}
    start_time = datetime(2026, 7, 26, tzinfo=UTC)
    for label, sample in (("event start", bounds[0]), ("event end", bounds[1]), ("event peak", bounds[0] + 4)):
        assert abs(markers[label] - mdates.date2num(start_time + timedelta(seconds=sample / 100))) < 1e-11
    expected_start = mdates.date2num(start_time + timedelta(seconds=expected[0] / 100 - 0.005))
    expected_stop = mdates.date2num(start_time + timedelta(seconds=expected[1] / 100 + 0.005))
    for figure in figures:
        assert abs(figure.axes[0].get_xlim()[0] - expected_start) < 1e-11
        assert abs(figure.axes[0].get_xlim()[1] - expected_stop) < 1e-11


@pytest.mark.parametrize("single_channel", [False, True])
def test_automatic_preview_uses_exact_event_indices_not_distance_envelope(
    waterfall_api, monkeypatch, single_channel
):
    from das_sentinel import artifacts

    client, _, sessions, csrf_token, ids, _ = waterfall_api
    reader = client.app.state.waterfall_preview_service.reader
    original_metadata, original_read = reader.read_metadata, reader.read_block
    original_png = artifacts._png
    # Channels outside the event also fall in the event's distance envelope.
    distances = np.array([16.0, 20.0, 12.0, 16.0, 18.0])
    if single_channel:
        with sessions.begin() as session:
            event = session.get(Event, ids["event"])
            event.channel_stop = 2
            payload = json.loads(event.evidence[0].payload_json)
            payload["channel_stop"] = 2
            event.evidence[0].payload_json = json.dumps(payload)
    reads, figures = [], []

    def metadata(path):
        return replace(original_metadata(path), channel_distance_m=distances)

    def read(path, start, stop, channels):
        reads.append(channels)
        block = original_read(path, start, stop, channels)
        return replace(block, metadata=replace(block.metadata, channel_distance_m=distances[channels]))

    def capture_png(figure):
        figures.append(figure)
        return original_png(figure)

    monkeypatch.setattr(reader, "read_metadata", metadata)
    monkeypatch.setattr(reader, "read_block", read)
    monkeypatch.setattr(artifacts, "_png", capture_png)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview", json={},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    assert reads == [slice(1, 2 if single_channel else 4)]
    assert response.json()["channel_count"] == (1 if single_channel else 2)
    assert response.json()["distance_start_m"] == (20.0 if single_channel else 16.0)
    assert response.json()["distance_stop_m"] == 20.0
    assert len(figures) == 3
    for figure in figures:
        assert "Automatic event range" in " ".join(figure.axes[0].get_title().split())


def test_preview_service_default_range_returns_waveform(waterfall_api, monkeypatch):
    import das_sentinel.services.waterfall_previews as previews

    client, _, sessions, _, ids, _ = waterfall_api
    original_render = previews.render_waveform_png
    records = []

    def capture_waveform(values, dates, context, event, **kwargs):
        records.append(event)
        return original_render(values, dates, context, event, **kwargs)

    monkeypatch.setattr(previews, "render_waveform_png", capture_waveform)
    with sessions() as session:
        result = client.app.state.waterfall_preview_service.render(session, ids["event"])
    assert result.waveform_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(records) == 1
    assert isinstance(records[0], EventRecord)
    assert records[0].event_id == ids["event"]
    assert (records[0].start_sample, records[0].end_sample, records[0].peak_sample) == (20, 40, 30)
    assert (records[0].channel_start, records[0].channel_stop) == (1, 4)
    assert len(records[0].evidence) == 1


@pytest.mark.parametrize("automatic", [False, True])
def test_preview_minimum_three_seconds_is_capped_before_source_read(
    waterfall_api, monkeypatch, automatic
):
    client, _, sessions, csrf_token, ids, _ = waterfall_api
    reader = client.app.state.waterfall_preview_service.reader
    original_metadata = reader.read_metadata
    with sessions.begin() as session:
        event = session.get(Event, ids["event"])
        event.occurred_at = datetime(2026, 7, 26, tzinfo=UTC) + timedelta(seconds=20 / 10001)

    def high_rate_metadata(path):
        return replace(original_metadata(path), sample_rate_hz=10001.0)

    def forbidden_read(*args):
        pytest.fail("the 30003-sample minimum window must be rejected before reading")

    monkeypatch.setattr(reader, "read_metadata", high_rate_metadata)
    monkeypatch.setattr(reader, "sample_count", lambda path: 50000)
    monkeypatch.setattr(reader, "read_block", forbidden_read)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview",
        json={} if automatic else {"distance_start_m": 12.0, "distance_stop_m": 20.0},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("failure", ["missing_source", "changed_source", "waveform_render"])
def test_automatic_preview_conflict_never_returns_partial_images(waterfall_api, monkeypatch, failure):
    import das_sentinel.services.waterfall_previews as previews

    client, _, _, csrf_token, ids, source_path = waterfall_api
    if failure == "missing_source":
        source_path.unlink()
    elif failure == "changed_source":
        _write_source(source_path, 300)
    else:
        def fail_waveform(*args, **kwargs):
            raise OSError("waveform render failed")

        monkeypatch.setattr(previews, "render_waveform_png", fail_waveform)
    response = client.post(
        f"/api/v1/events/{ids['event']}/waterfall-preview", json={},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 409
    assert set(response.json()) == {"detail"}
    assert str(source_path) not in response.text


@pytest.mark.parametrize(
    "waveform", [None, "1234567", "x" * (16 * 1024 * 1024 + 1)],
    ids=["missing", "too-short", "too-large"],
)
def test_preview_response_requires_bounded_waveform_payload(waveform):
    from das_sentinel.api.schemas import WaterfallPreviewResponse
    from pydantic import ValidationError

    payload = {
        "distance_start_m": 12.0, "distance_stop_m": 20.0, "channel_count": 2,
        "primary_detector": "stalta", "primary_detector_display": "STA/LTA",
        "filter_low_hz": 3.5, "filter_high_hz": 27.25,
        "raw_png_base64": "12345678", "filtered_png_base64": "12345678",
    }
    if waveform is not None:
        payload["waveform_png_base64"] = waveform
    with pytest.raises(ValidationError):
        WaterfallPreviewResponse.model_validate(payload)


def _tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }
