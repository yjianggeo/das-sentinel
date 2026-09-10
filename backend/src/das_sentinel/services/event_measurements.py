from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from das_sentinel.services.discovery import DiscoveryPathError, _open_anchored_source

MAX_MEASUREMENT_FILE_BYTES = 1024 * 1024

REQUIRED_MEASUREMENT_NAMES = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class EventMeasurementIdentity:
    event_id: str
    source_fingerprint: str
    start_sample: int
    end_sample: int
    channel_start: int
    channel_stop: int


def read_event_measurements(
    app_data_dir: Path,
    identity: EventMeasurementIdentity,
) -> dict[str, Any] | None:
    """Return a verified v2 envelope, or None for absent/untrusted sidecars."""

    if not _safe_event_id(identity.event_id):
        return None
    root = (Path(app_data_dir).absolute() / "screening-measurements" / "v2")
    candidate = root / f"{identity.event_id}.json"
    try:
        with _open_anchored_source(
            root,
            candidate,
            required_suffix=".json",
        ) as source:
            if source.snapshot().size_bytes > MAX_MEASUREMENT_FILE_BYTES:
                return None
            encoded = source.stream.read(MAX_MEASUREMENT_FILE_BYTES + 1)
    except (DiscoveryPathError, FileNotFoundError, OSError, ValueError):
        return None
    if not encoded or len(encoded) > MAX_MEASUREMENT_FILE_BYTES:
        return None

    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        return _validated_measurements(payload, identity)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return None


def read_registered_event_measurements(
    artifact_root: Path,
    stored_path: str,
    expected_sha256: str,
    identity: EventMeasurementIdentity,
) -> dict[str, Any] | None:
    """Return measurements from a registered, digest-matched evidence artifact."""

    relative = Path(stored_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == Path(".")
        or ".." in relative.parts
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return None
    root = Path(artifact_root).absolute()
    candidate = root / relative
    try:
        with _open_anchored_source(root, candidate, required_suffix=".json") as source:
            if source.snapshot().size_bytes > MAX_MEASUREMENT_FILE_BYTES:
                return None
            encoded = source.stream.read(MAX_MEASUREMENT_FILE_BYTES + 1)
    except (DiscoveryPathError, FileNotFoundError, OSError, ValueError):
        return None
    if (
        not encoded
        or len(encoded) > MAX_MEASUREMENT_FILE_BYTES
        or hashlib.sha256(encoded).hexdigest() != expected_sha256
    ):
        return None
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
        return _validated_evidence_measurements(payload, identity)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return None


def _validated_measurements(
    payload: object,
    identity: EventMeasurementIdentity,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "event_id",
        "source_fingerprint",
        "start_sample",
        "end_sample",
        "channel_start",
        "channel_stop",
        "measurements",
    }:
        raise ValueError("measurement sidecar has an invalid shape")
    expected = {
        "event_id": identity.event_id,
        "source_fingerprint": identity.source_fingerprint,
        "start_sample": identity.start_sample,
        "end_sample": identity.end_sample,
        "channel_start": identity.channel_start,
        "channel_stop": identity.channel_stop,
    }
    if any(
        type(payload[name]) is not int
        for name in ("start_sample", "end_sample", "channel_start", "channel_stop")
    ):
        raise ValueError("measurement sidecar ranges must be integers")
    if any(payload[name] != value for name, value in expected.items()):
        raise ValueError("measurement sidecar identity does not match the event")

    return _validated_envelope(payload["measurements"])


def _validated_evidence_measurements(
    payload: object,
    identity: EventMeasurementIdentity,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("evidence artifact must be an object")
    event_range = payload.get("event_range")
    if not isinstance(event_range, dict) or set(event_range) != {
        "start_sample",
        "end_sample",
        "channel_start",
        "channel_stop",
    }:
        raise ValueError("evidence event range has an invalid shape")
    expected_range = {
        "start_sample": identity.start_sample,
        "end_sample": identity.end_sample,
        "channel_start": identity.channel_start,
        "channel_stop": identity.channel_stop,
    }
    if any(type(event_range[name]) is not int for name in expected_range):
        raise TypeError("evidence event ranges must be integers")
    if (
        payload.get("event_id") != identity.event_id
        or payload.get("source_fingerprint") != identity.source_fingerprint
        or event_range != expected_range
        or "measurements" not in payload
    ):
        raise ValueError("evidence measurement identity does not match the event")
    return _validated_envelope(payload["measurements"])


def _validated_envelope(envelope: object) -> dict[str, Any]:
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version",
        "values",
        "status",
        "context",
    }:
        raise ValueError("measurement envelope has an invalid shape")
    if envelope["schema_version"] != "2":
        raise ValueError("measurement schema is unsupported")
    values = envelope["values"]
    statuses = envelope["status"]
    context = envelope["context"]
    if not isinstance(values, dict) or not isinstance(statuses, dict):
        raise TypeError("measurement values and status must be objects")
    if not isinstance(context, dict):
        raise TypeError("measurement context must be an object")
    if set(values) != set(statuses) or not REQUIRED_MEASUREMENT_NAMES.issubset(values):
        raise ValueError("measurement values and status are incomplete")

    normalized: dict[str, float | None] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name or len(name) > 128:
            raise ValueError("measurement name is invalid")
        measurement_status = statuses[name]
        if (
            not isinstance(measurement_status, str)
            or not measurement_status.strip()
            or len(measurement_status) > 128
        ):
            raise ValueError("measurement status is invalid")
        if value is None:
            if measurement_status == "measured":
                raise ValueError("missing measurement cannot have measured status")
            normalized[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("measurement values must be numbers or null")
        number = float(value)
        if not math.isfinite(number) or measurement_status != "measured":
            raise ValueError("measurement value or status is invalid")
        normalized[name] = number

    _require_finite_json(context)
    return {
        "schema_version": "2",
        "values": normalized,
        "status": dict(statuses),
        "context": context,
    }


def _safe_event_id(event_id: str) -> bool:
    return (
        bool(event_id)
        and len(event_id) <= 128
        and Path(event_id).name == event_id
        and event_id not in {".", ".."}
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("measurement JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"measurement JSON contains invalid constant: {value}")


def _require_finite_json(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("measurement context contains a non-finite number")
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
