from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from scipy.signal import stft, welch

from das_sentinel.domain import EventRecord


@dataclass(frozen=True)
class EventArtifactData:
    raw_data: np.ndarray
    filtered_data: np.ndarray
    sample_offset: int
    sample_rate_hz: float
    channel_distance_m: np.ndarray
    bad_channel_mask: np.ndarray
    start_time_utc: datetime
    site_timezone: str
    profile: Mapping[str, Any]
    profile_version: str
    pipeline_version: str
    model_version: str
    backend_name: str
    rule_output: Mapping[str, Any]
    model_output: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        raw = _finite_float_copy(self.raw_data, "raw_data")
        filtered = _finite_float_copy(self.filtered_data, "filtered_data")
        if raw.ndim != 2 or raw.shape != filtered.shape:
            raise ValueError("raw_data and filtered_data must have the same [time, channel] shape")
        if raw.shape[0] == 0 or raw.shape[1] == 0:
            raise ValueError("artifact data must not be empty")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        distance = np.asarray(self.channel_distance_m, dtype=float).copy()
        mask = np.asarray(self.bad_channel_mask, dtype=bool).copy()
        if distance.shape != (raw.shape[1],) or not np.isfinite(distance).all():
            raise ValueError("channel_distance_m must be finite and match channel count")
        if mask.shape != (raw.shape[1],):
            raise ValueError("bad_channel_mask must match channel count")
        if self.start_time_utc.tzinfo is None or self.start_time_utc.utcoffset() is None:
            raise ValueError("start_time_utc must be timezone-aware")
        try:
            ZoneInfo(self.site_timezone)
        except ZoneInfoNotFoundError as error:
            raise ValueError("site_timezone must be a valid IANA timezone") from error
        if not self.profile_version or not self.pipeline_version or not self.model_version or not self.backend_name:
            raise ValueError("profile_version, pipeline_version, model_version, and backend_name must be non-empty")
        object.__setattr__(self, "raw_data", raw)
        object.__setattr__(self, "filtered_data", filtered)
        object.__setattr__(self, "channel_distance_m", distance)
        object.__setattr__(self, "bad_channel_mask", mask)
        object.__setattr__(self, "start_time_utc", self.start_time_utc.astimezone(UTC))


@dataclass(frozen=True)
class ArtifactFile:
    kind: str
    path: Path
    media_type: str
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    event_id: str
    files: dict[str, ArtifactFile]


class ArtifactConflictError(RuntimeError):
    """Raised when a stable artifact target already contains different bytes."""


class ArtifactStore:
    """Atomically store fixed artifact kinds beneath a validated event UUID."""

    _suffixes: ClassVar[dict[str, str]] = {
        "waterfall_raw": ".png", "waterfall_filtered": ".png", "waveform": ".png",
        "psd": ".png", "stft": ".png", "evidence": ".json",
    }

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, event_id: str, kind: str, content: bytes, media_type: str) -> ArtifactFile:
        try:
            validated_id = str(UUID(event_id))
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("event_id must be a UUID") from error
        if kind not in self._suffixes:
            raise ValueError("unknown artifact kind")
        event_dir = (self.root / validated_id).resolve()
        if event_dir.parent != self.root:
            raise ValueError("artifact path escapes root")
        event_dir.mkdir(parents=True, exist_ok=True)
        target = (event_dir / f"{kind}{self._suffixes[kind]}").resolve()
        if target.parent != event_dir:
            raise ValueError("artifact path escapes event directory")
        with _process_lock(target.with_suffix(target.suffix + ".lock")):
                digest = hashlib.sha256(content).hexdigest()
                if target.exists():
                    final_digest = hashlib.sha256(target.read_bytes()).hexdigest()
                    if final_digest == digest:
                        return ArtifactFile(kind, target, media_type, final_digest)
                    raise ArtifactConflictError(f"artifact already exists with different bytes: {kind}")
                descriptor, temporary_name = tempfile.mkstemp(prefix=f".{kind}-", suffix=".tmp", dir=event_dir)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        temporary.unlink()
                final_digest = hashlib.sha256(target.read_bytes()).hexdigest()
                if final_digest != digest:
                    raise RuntimeError("artifact bytes changed during atomic write")
                return ArtifactFile(kind, target, media_type, final_digest)

def render_event_artifacts(
    data: EventArtifactData, event: EventRecord, features: Mapping[str, Any], out_dir: Path
) -> ArtifactManifest:
    """Render event evidence from explicitly distinct raw and filtered arrays."""
    time_slice, channels = _event_selection(data, event)
    store = ArtifactStore(Path(out_dir))
    raw = data.raw_data[time_slice][:, channels]
    filtered = data.filtered_data[time_slice][:, channels]
    dates = _sample_dates(data, time_slice)
    distances = data.channel_distance_m[channels]
    waveform = _representative_beam(filtered)
    files = {
        "waterfall_raw": store.write(event.event_id, "waterfall_raw", _waterfall_png(raw, dates, distances, data, "Raw waterfall"), "image/png"),
        "waterfall_filtered": store.write(event.event_id, "waterfall_filtered", _waterfall_png(filtered, dates, distances, data, "Filtered waterfall"), "image/png"),
        "waveform": store.write(event.event_id, "waveform", _waveform_png(waveform, dates, data, event, time_slice), "image/png"),
        "psd": store.write(event.event_id, "psd", _psd_png(waveform, data, features), "image/png"),
        "stft": store.write(event.event_id, "stft", _stft_png(waveform, data), "image/png"),
    }
    evidence = _evidence(data, event, features, time_slice, channels)
    encoded_evidence = json.dumps(evidence, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    files["evidence"] = store.write(event.event_id, "evidence", encoded_evidence, "application/json")
    return ArtifactManifest(event.event_id, files)


def _finite_float_copy(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating-point array")
    result = array.copy()
    return result


@contextmanager
def _process_lock(path: Path):
    lock_file = open(path, "a+b")  # noqa: SIM115 - lock lifetime crosses yield.
    try:
        if os.name == "nt":
            import msvcrt
            lock_file.seek(0)
            lock_file.write(b"0")
            lock_file.flush()
            lock_file.seek(0)
            while True:
                try:
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EDEADLK}:
                        raise
                    time.sleep(0.01)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if os.name == "nt":
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _event_selection(data: EventArtifactData, event: EventRecord) -> tuple[slice, np.ndarray]:
    start = max(0, event.start_sample - data.sample_offset)
    stop = min(data.raw_data.shape[0], event.end_sample - data.sample_offset)
    if start >= stop:
        raise ValueError("event has no overlap with artifact input")
    channel_start = max(0, event.channel_start)
    channel_stop = min(data.raw_data.shape[1], event.channel_stop)
    channels = np.arange(channel_start, channel_stop)
    channels = channels[~data.bad_channel_mask[channels]]
    if channels.size == 0:
        raise ValueError("event has no usable channels")
    return slice(start, stop), channels


def _sample_dates(data: EventArtifactData, samples: slice) -> np.ndarray:
    return np.array([data.start_time_utc + timedelta(seconds=(samples.start + index) / data.sample_rate_hz) for index in range(samples.stop - samples.start)])


def _robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lower, upper = np.percentile(finite, (1, 99))
    if lower == upper:
        center = float(lower)
        return center - 1.0, center + 1.0
    return float(lower), float(upper)


def _time_axes(axis: Any, dates: np.ndarray, timezone: str, sample_rate_hz: float) -> None:
    utc_numbers = mdates.date2num(dates)
    step = 1 / sample_rate_hz / 86_400
    axis.set_xlim(utc_numbers[0] - step / 2, utc_numbers[-1] + 1.5 * step)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=UTC))
    axis.set_xlabel("UTC time")
    local = axis.twiny()
    local.set_xlim(axis.get_xlim())
    local.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=ZoneInfo(timezone)))
    local.set_xlabel(f"Station local time ({timezone})")


def _png(figure: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=120, metadata={"Software": "das-sentinel"})
        return buffer.getvalue()
    finally:
        plt.close(figure)


def _waterfall_png(values: np.ndarray, dates: np.ndarray, distances: np.ndarray, data: EventArtifactData, title: str) -> bytes:
    figure, axis = plt.subplots(figsize=(9, 4))
    low, high = _robust_limits(values)
    image = _waterfall_mesh(axis, values, mdates.date2num(dates), distances, low, high)
    figure.colorbar(image, ax=axis, label="Amplitude")
    axis.set_ylabel("Distance (m)")
    axis.set_title(f"{title} | {_versions(data)} | {data.sample_rate_hz:g} Hz | input unit")
    _time_axes(axis, dates, data.site_timezone, data.sample_rate_hz)
    return _png(figure)


def _coordinate_edges(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    middle = (values[:-1] + values[1:]) / 2
    return np.concatenate(([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])]))


def _waterfall_mesh(axis: Any, values: np.ndarray, times: np.ndarray, distances: np.ndarray, low: float, high: float) -> Any:
    return axis.pcolormesh(_coordinate_edges(times), _coordinate_edges(distances), values.T, shading="flat", vmin=low, vmax=high, cmap="viridis")


def _representative_beam(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        beam = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=1)
    return np.where(np.isfinite(beam), beam, 0.0)


def _waveform_png(values: np.ndarray, dates: np.ndarray, data: EventArtifactData, event: EventRecord, samples: slice) -> bytes:
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.plot(mdates.date2num(dates), values, linewidth=0.8)
    axis.set_ylabel("Filtered amplitude")
    for sample, label in ((event.start_sample, "event start"), (event.end_sample, "event end"), (event.peak_sample, "event peak")):
        axis.axvline(mdates.date2num(data.start_time_utc + timedelta(seconds=(sample - data.sample_offset) / data.sample_rate_hz)), label=label, linewidth=0.8)
    for candidate in event.evidence:
        axis.axvspan(mdates.date2num(data.start_time_utc + timedelta(seconds=(candidate.start_sample - data.sample_offset) / data.sample_rate_hz)), mdates.date2num(data.start_time_utc + timedelta(seconds=(candidate.end_sample - data.sample_offset) / data.sample_rate_hz)), alpha=0.1, label=f"{candidate.detector} span")
        axis.axvline(mdates.date2num(data.start_time_utc + timedelta(seconds=(candidate.peak_sample - data.sample_offset) / data.sample_rate_hz)), label=f"{candidate.detector} peak", linestyle="--", linewidth=0.7)
    axis.legend(fontsize="x-small")
    axis.set_title(f"Robust filtered good-channel beam | {_versions(data)} | samples {data.sample_offset + samples.start}:{data.sample_offset + samples.stop}")
    _time_axes(axis, dates, data.site_timezone, data.sample_rate_hz)
    return _png(figure)


def _psd_png(values: np.ndarray, data: EventArtifactData, features: Mapping[str, Any]) -> bytes:
    samples = len(values)
    nperseg = min(samples, max(1, round(data.sample_rate_hz * 2)))
    frequency, power = welch(values.astype(np.float64, copy=False), fs=data.sample_rate_hz, nperseg=nperseg)
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.semilogy(frequency, np.maximum(power, np.finfo(np.float64).tiny))
    dominant = features.get("dominant_frequency_hz")
    if isinstance(dominant, (int, float)) and np.isfinite(dominant):
        axis.axvline(dominant, color="tab:red", label="dominant frequency")
        axis.legend(fontsize="x-small")
    axis.set(xlabel="Frequency (Hz)", ylabel="PSD (amplitude^2/Hz)", title=f"Welch PSD | {data.profile.get('energy_bands_hz', ())} | {_versions(data)}")
    return _png(figure)


def _stft_png(values: np.ndarray, data: EventArtifactData) -> bytes:
    samples = len(values)
    nperseg = min(samples, max(1, samples // 4))
    noverlap = 0 if nperseg == 1 else nperseg // 2
    frequency, times, spectrum = stft(values, fs=data.sample_rate_hz, nperseg=nperseg, noverlap=noverlap)
    figure, axis = plt.subplots(figsize=(7, 3))
    image = axis.pcolormesh(times, frequency, np.abs(spectrum) ** 2, shading="auto")
    figure.colorbar(image, ax=axis, label="Energy")
    axis.set(xlabel="Time from event window start (s)", ylabel="Frequency (Hz)", title=f"STFT energy | {_versions(data)}")
    return _png(figure)


def _versions(data: EventArtifactData) -> str:
    return f"profile={data.profile_version} pipeline={data.pipeline_version} model={data.model_version}"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Enum):
        return _json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON evidence value: {type(value).__name__}")


def _evidence(data: EventArtifactData, event: EventRecord, features: Mapping[str, Any], samples: slice, channels: np.ndarray) -> dict[str, Any]:
    return _json_value({
        "event_id": event.event_id,
        "profile": data.profile,
        "profile_version": data.profile_version,
        "pipeline_version": data.pipeline_version,
        "model_version": data.model_version,
        "backend": data.backend_name,
        "source_range": {"sample_start": data.sample_offset + samples.start, "sample_stop": data.sample_offset + samples.stop, "sample_rate_hz": data.sample_rate_hz, "channel_indices": channels.tolist(), "distance_m": data.channel_distance_m[channels].tolist(), "start_time_utc": data.start_time_utc.isoformat(), "site_timezone": data.site_timezone},
        "detections": [asdict(candidate) for candidate in event.evidence],
        "features": features,
        "rule_output": data.rule_output,
        "model_output": data.model_output,
        "quality_flags": {"bad_channel_count": int(data.bad_channel_mask.sum()), "event_quality_flags": sorted({flag for candidate in event.evidence for flag in candidate.quality_flags})},
        "input_sha256": {"raw": hashlib.sha256(data.raw_data.tobytes()).hexdigest(), "filtered": hashlib.sha256(data.filtered_data.tobytes()).hexdigest()},
    })
