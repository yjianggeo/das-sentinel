from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import event as sqlalchemy_event

from tests.test_events_api import PASSWORD

pytest_plugins = ("tests.test_events_api",)


MEASUREMENT_NAMES = (
    "input_rms",
    "snr_db",
    "channel_snr_median_db",
    "channel_snr_p90_db",
    "valid_channel_count",
    "valid_channel_fraction",
    "bad_channel_fraction",
    "clipping_fraction",
    "trigger_duration_s",
    "energy_duration_5_95_s",
    "rise_time_s",
    "pulse_count",
    "pulse_interval_s",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
    "frequency_resolution_hz",
    "band_fraction_low",
    "band_fraction_mid",
    "band_fraction_high",
    "active_channel_fraction",
    "longest_active_span_m",
    "distance_extent_m",
    "lagged_correlation",
    "slowness_s_per_m",
    "apparent_velocity_mps",
    "direction",
    "moveout_residual_s",
)


def _measurements() -> dict[str, object]:
    values = {name: float(index + 1) for index, name in enumerate(MEASUREMENT_NAMES)}
    values["clipping_fraction"] = None
    statuses = {name: "measured" for name in MEASUREMENT_NAMES}
    statuses["clipping_fraction"] = "adc_limits_unavailable"
    return {
        "schema_version": "2",
        "values": values,
        "status": statuses,
        "context": {
            "sampling_rate_hz": 100.0,
            "effective_frequency_band_hz": [0.5, 50.0],
            "quality_flags": [],
        },
    }


def _sidecar(event_api, **overrides: object) -> dict[str, object]:
    _, _, sessions, _, ids = event_api
    with sessions() as session:
        from das_sentinel.persistence.models import Event, ProcessingRun

        event = session.get(Event, ids["event_1"])
        assert event is not None
        run = session.get(ProcessingRun, event.processing_run_id)
        assert run is not None
        payload: dict[str, object] = {
            "event_id": event.id,
            "source_fingerprint": run.source_fingerprint,
            "start_sample": event.start_sample,
            "end_sample": event.end_sample,
            "channel_start": event.channel_start,
            "channel_stop": event.channel_stop,
            "measurements": _measurements(),
        }
    payload.update(overrides)
    return payload


def _sidecar_path(event_api) -> Path:
    _, settings, _, _, ids = event_api
    root = settings.app_data_dir / "screening-measurements" / "v2"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{ids['event_1']}.json"


def _write_sidecar(event_api, payload: dict[str, object]) -> Path:
    path = _sidecar_path(event_api)
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    return path


def _evidence_payload(event_api, **overrides: object) -> dict[str, object]:
    _, _, sessions, _, ids = event_api
    with sessions() as session:
        from das_sentinel.persistence.models import Event, ProcessingRun

        event = session.get(Event, ids["event_1"])
        assert event is not None
        run = session.get(ProcessingRun, event.processing_run_id)
        assert run is not None
        payload: dict[str, object] = {
            "event_id": event.event_uid,
            "source_fingerprint": run.source_fingerprint,
            "event_range": {
                "start_sample": event.start_sample,
                "end_sample": event.end_sample,
                "channel_start": event.channel_start,
                "channel_stop": event.channel_stop,
            },
            "measurements": _measurements(),
            "features": {"legacy": 1.0},
        }
    payload.update(overrides)
    return payload


def _register_evidence(
    event_api,
    payload: dict[str, object] | None = None,
    *,
    stored_path: str | None = None,
    content: bytes | None = None,
    registered_sha256: str | None = None,
) -> Path:
    _, settings, sessions, _, ids = event_api
    encoded = (
        json.dumps(_evidence_payload(event_api) if payload is None else payload, allow_nan=True).encode()
        if content is None
        else content
    )
    relative = stored_path or f"{ids['event_1']}/evidence.json"
    target = settings.app_data_dir / "artifacts" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)
    with sessions.begin() as session:
        from das_sentinel.persistence.repositories import Repositories

        Repositories(session).artifacts.add(
            event_id=ids["event_1"],
            kind="evidence",
            path=relative,
            media_type="application/json",
            sha256=registered_sha256 or hashlib.sha256(encoded).hexdigest(),
        )
    return target


def test_measurements_require_auth_and_distinguish_unknown_from_missing(event_api):
    client, _, _, _, ids = event_api
    client.cookies.clear()

    assert client.get(f"/api/v1/events/{ids['event_1']}/measurements").status_code == 401

    assert client.post("/api/v1/session", json={"password": PASSWORD}).status_code == 204
    unknown = client.get("/api/v1/events/does-not-exist/measurements")
    missing = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert unknown.status_code == 404
    assert missing.status_code == 200
    assert missing.json() == {"available": False}


def test_matching_measurement_sidecar_is_returned_without_database_writes(event_api):
    client, _, _, _, ids = event_api
    measurements = _measurements()
    _write_sidecar(event_api, _sidecar(event_api, measurements=measurements))
    writes: list[str] = []

    def capture_writes(connection, cursor, statement, parameters, context, executemany):
        del connection, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            writes.append(statement)

    engine = client.app.state.engine
    sqlalchemy_event.listen(engine, "before_cursor_execute", capture_writes)
    try:
        response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")
    finally:
        sqlalchemy_event.remove(engine, "before_cursor_execute", capture_writes)

    assert response.status_code == 200
    assert response.json() == {"available": True, "measurements": measurements}
    assert writes == []


def test_registered_evidence_measurements_are_used_when_sidecar_is_absent(event_api):
    client, _, _, _, ids = event_api
    measurements = _measurements()
    _register_evidence(
        event_api,
        _evidence_payload(event_api, measurements=measurements),
    )

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.status_code == 200
    assert response.json() == {"available": True, "measurements": measurements}


def test_invalid_historical_sidecar_falls_back_to_registered_evidence(event_api):
    client, _, _, _, ids = event_api
    measurements = _measurements()
    _sidecar_path(event_api).write_text("{", encoding="utf-8")
    _register_evidence(
        event_api,
        _evidence_payload(event_api, measurements=measurements),
    )

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.json() == {"available": True, "measurements": measurements}


@pytest.mark.parametrize(
    ("field", "stale_value"),
    (
        ("event_id", "stale-event"),
        ("source_fingerprint", "0" * 64),
        ("start_sample", 9),
        ("end_sample", 21),
        ("channel_start", 0),
        ("channel_stop", 4),
    ),
)
def test_registered_evidence_identity_mismatch_is_unavailable(
    event_api, field, stale_value
):
    client, _, _, _, ids = event_api
    payload = _evidence_payload(event_api)
    if field in {"event_id", "source_fingerprint"}:
        payload[field] = stale_value
    else:
        payload["event_range"][field] = stale_value
    _register_evidence(event_api, payload)

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.json() == {"available": False}


@pytest.mark.parametrize(
    ("content", "registered_sha256"),
    (
        (b"{", None),
        (b" " * (1024 * 1024 + 1), None),
        (None, "0" * 64),
    ),
    ids=("malformed", "oversized", "digest-mismatch"),
)
def test_untrusted_registered_evidence_is_unavailable(
    event_api, content, registered_sha256
):
    client, _, _, _, ids = event_api
    _register_evidence(
        event_api,
        content=content,
        registered_sha256=registered_sha256,
    )

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.json() == {"available": False}


def test_registered_evidence_path_escape_is_unavailable(event_api):
    client, _, _, _, ids = event_api
    _register_evidence(event_api, stored_path="../outside.json")

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.json() == {"available": False}


def test_registered_evidence_symlink_is_unavailable(event_api, tmp_path):
    client, settings, sessions, _, ids = event_api
    encoded = json.dumps(_evidence_payload(event_api), allow_nan=False).encode()
    outside = tmp_path / "outside-evidence.json"
    outside.write_bytes(encoded)
    relative = f"{ids['event_1']}/evidence.json"
    linked = settings.app_data_dir / "artifacts" / relative
    linked.parent.mkdir(parents=True, exist_ok=True)
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    with sessions.begin() as session:
        from das_sentinel.persistence.repositories import Repositories

        Repositories(session).artifacts.add(
            event_id=ids["event_1"],
            kind="evidence",
            path=relative,
            media_type="application/json",
            sha256=hashlib.sha256(encoded).hexdigest(),
        )

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.json() == {"available": False}


def test_nonfinite_registered_measurements_are_unavailable(event_api):
    client, _, _, _, ids = event_api
    payload = _evidence_payload(event_api)
    payload["measurements"]["values"]["input_rms"] = float("nan")
    _register_evidence(event_api, payload)

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.json() == {"available": False}


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("event_id", "another-event"),
        ("source_fingerprint", "0" * 64),
        ("start_sample", 10.0),
        ("start_sample", 9),
        ("end_sample", 21),
        ("channel_start", 0),
        ("channel_stop", 4),
    ],
)
def test_stale_measurement_identity_is_unavailable(event_api, field: str, stale_value: object):
    client, _, _, _, ids = event_api
    _write_sidecar(event_api, _sidecar(event_api, **{field: stale_value}))

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.status_code == 200
    assert response.json() == {"available": False}


def test_measurement_symlink_is_unavailable(event_api, tmp_path: Path):
    client, _, _, _, ids = event_api
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_sidecar(event_api)), encoding="utf-8")
    linked = _sidecar_path(event_api)
    try:
        linked.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.status_code == 200
    assert response.json() == {"available": False}


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(b"{", id="malformed"),
        pytest.param(b" " * (2 * 1024 * 1024), id="oversized"),
    ],
)
def test_malformed_or_oversized_measurement_file_is_unavailable(event_api, contents: bytes):
    client, _, _, _, ids = event_api
    _sidecar_path(event_api).write_bytes(contents)

    response = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert response.status_code == 200
    assert response.json() == {"available": False}


def test_nonfinite_or_incomplete_measurement_envelope_is_unavailable(event_api):
    client, _, _, _, ids = event_api
    payload = _sidecar(event_api)
    measurements = payload["measurements"]
    assert isinstance(measurements, dict)
    measurements["values"]["input_rms"] = float("nan")
    path = _sidecar_path(event_api)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    nonfinite = client.get(f"/api/v1/events/{ids['event_1']}/measurements")

    assert nonfinite.status_code == 200
    assert nonfinite.json() == {"available": False}

    measurements["values"]["input_rms"] = 1.0
    del measurements["status"]["input_rms"]
    _write_sidecar(event_api, payload)

    incomplete = client.get(f"/api/v1/events/{ids['event_1']}/measurements")
    assert incomplete.status_code == 200
    assert incomplete.json() == {"available": False}
