"""UTC minute analyses with native channel geometry and bounded artifacts."""

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt

from das_sentinel.config import ProcessingProfile
from das_sentinel.readers.continuous_low_frequency import read_low_frequency_window
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.screening import screen_block
from das_sentinel.services.discovery import fingerprint
from das_sentinel.signal.backend import select_backend
from das_sentinel.signal.scan_distance import scan_channel_mask
from das_sentinel.window_regions import merge_regions

WINDOW_SECONDS = 30
SAMPLE_RATE_HZ = 100
ALGORITHM_VERSION = "whole-window-v2"
CONTEXT_SECONDS = 30
READ_CONTEXT_SECONDS = 40  # Includes the reader's ten-second anti-alias halo.
MAX_EVIDENCE_BYTES = 32 * 1024 * 1024
MAX_ARTIFACTS = 8


def window_start(value: datetime, window_seconds: int = WINDOW_SECONDS) -> datetime:
    if window_seconds not in (30, 60):
        raise ValueError("window length must be 30 or 60 seconds")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("window time must include timezone")
    value = value.astimezone(UTC)
    return value.replace(second=(value.second // window_seconds) * window_seconds, microsecond=0)


def window_id(site_id: str, start: datetime, window_seconds: int = WINDOW_SECONDS) -> str:
    return str(
        uuid5(
            NAMESPACE_URL,
            f"whole-window|{site_id}|{window_start(start, window_seconds).isoformat()}|{window_seconds}",
        )
    )


def _json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def compute_input_sha256(metadata: Mapping) -> str:
    """Hash the explicit source/time/grid/profile/mask identity, not output features."""
    keys = (
        "algorithm_version",
        "start_utc",
        "end_utc",
        "sample_rate_hz",
        "shape",
        "unit",
        "grid_sha256",
        "sources",
        "profile",
        "recognition_range_m",
        "valid_channel_mask",
        "recognition_channel_mask",
        "read_start_utc",
        "read_end_utc",
        "read_sources",
        "conditioning_start_utc",
        "conditioning_end_utc",
    )
    return hashlib.sha256(_json({key: metadata[key] for key in keys})).hexdigest()


def inspect_sources(paths: Sequence[Path]) -> list[dict]:
    reader = ZdHdf5Reader()
    sources = []
    for path in dict.fromkeys(Path(p).resolve(strict=True) for p in paths):
        before = path.stat()
        signature = fingerprint(path)
        meta, count = reader.read_metadata_and_count(path)
        if (
            count <= 0
            or not np.isfinite(meta.sample_rate_hz)
            or meta.sample_rate_hz < 100
        ):
            raise ValueError("source must contain samples at >=100 Hz")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ) or signature != fingerprint(path):
            raise ValueError("source changed during inspection")
        sources.append(
            {
                "path": str(path),
                "fingerprint": signature,
                "size_bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "start_utc": meta.start_time_utc.astimezone(UTC).isoformat(),
                "end_utc": (
                    meta.start_time_utc + timedelta(seconds=count / meta.sample_rate_hz)
                )
                .astimezone(UTC)
                .isoformat(),
                "sample_rate_hz": meta.sample_rate_hz,
                "distance_m": meta.channel_distance_m.tolist(),
                "unit": meta.unit,
            }
        )
    return sorted(sources, key=lambda s: (s["start_utc"], s["path"]))


def plan_windows(sources: Sequence[dict], window_seconds: int = WINDOW_SECONDS) -> list[dict]:
    intervals = sorted(
        [
            (
                datetime.fromisoformat(s["start_utc"]).astimezone(UTC),
                datetime.fromisoformat(s["end_utc"]).astimezone(UTC),
                s,
            )
            for s in sources
        ],
        key=lambda p: (p[0], p[2]["path"]),
    )
    starts = set()
    for first, last, _ in intervals:
        current = window_start(first, window_seconds)
        while current < last:
            starts.add(current)
            current += timedelta(seconds=window_seconds)
    result = []
    for start in sorted(starts):
        end = start + timedelta(seconds=window_seconds)
        left, right = (
            start - timedelta(seconds=READ_CONTEXT_SECONDS),
            end + timedelta(seconds=READ_CONTEXT_SECONDS),
        )
        selected = [(a, b, s) for a, b, s in intervals if a < right and b > left]
        cursor, complete = left, True
        previous = None
        for a, b, s in selected:
            tolerance = min(1 / float(s["sample_rate_hz"]), 0.05)
            if a > cursor + timedelta(seconds=tolerance):
                complete = False
            if previous is not None:
                if abs((a - previous[1]).total_seconds()) > tolerance:
                    complete = False
                if any(
                    s[k] != previous[2][k]
                    for k in ("distance_m", "unit", "sample_rate_hz")
                ):
                    complete = False
            cursor = max(cursor, b)
            previous = (a, b, s)
        complete = (
            complete and cursor >= right and bool(selected) and selected[0][0] <= left
        )
        result.append(
            {
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
                "sources": [s for _, _, s in selected],
                "coverage": "complete" if complete else "incomplete",
            }
        )
    return result


def _measure(data, distance, valid, recognition):
    selected = data[:, recognition].astype(np.float64)
    f = {
        "valid_channel_fraction": float(valid.mean()),
        "active_channel_fraction": None,
        "active_contiguous_span_m": None,
        "lag_correlation_median": None,
    }
    if not selected.shape[1]:
        return f
    second_energy = np.mean(selected.reshape(-1, 100, selected.shape[1]) ** 2, axis=1)
    active = np.max(second_energy, axis=0) > 4 * np.maximum(
        np.median(second_energy, axis=0), np.finfo(float).tiny
    )
    f["active_channel_fraction"] = float(active.mean())
    full_active = np.zeros(len(distance), bool)
    full_active[recognition] = active
    runs = np.split(
        np.flatnonzero(full_active),
        np.where(np.diff(np.flatnonzero(full_active)) != 1)[0] + 1,
    )
    f["active_contiguous_span_m"] = max(
        (float(distance[r[-1]] - distance[r[0]]) for r in runs if len(r)), default=0.0
    )
    shifted = []
    indices = np.flatnonzero(recognition)
    for left, right in itertools.pairwise(indices):
        a, b = data[:, left].astype(float), data[:, right].astype(float)
        a -= a.mean()
        b -= b.mean()
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            continue
        corr = correlate(b, a, mode="full", method="fft") / norm
        axis = correlation_lags(len(b), len(a))
        keep = np.abs(axis) <= 200
        peak = int(np.argmax(np.abs(corr[keep])))
        shifted.append(float(corr[keep][peak]))
    if shifted:
        f["lag_correlation_median"] = float(np.median(shifted))
    return f


def _atomic(path: Path, payload: bytes):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def analyze_window(
    paths: Sequence[Path],
    start_utc: datetime,
    profile: ProcessingProfile,
    output_dir: Path,
    *,
    site_timezone: str = "Asia/Shanghai",
    window_seconds: int = WINDOW_SECONDS,
    skip_empty: bool = False,
) -> dict:
    from das_sentinel.whole_window_plots import create_visualization

    start = window_start(start_utc, window_seconds)
    end = start + timedelta(seconds=window_seconds)
    sample_count = window_seconds * SAMPLE_RATE_HZ
    if (
        max(
            profile.sta_seconds + profile.lta_seconds,
            profile.energy_window_seconds * 2,
            profile.coherence_window_seconds,
        )
        > CONTEXT_SECONDS
    ):
        raise ValueError("profile conditioning requires more than 30 seconds context")
    snapshots = inspect_sources(paths)
    plan = next(
        (p for p in plan_windows(snapshots, window_seconds) if p["start_utc"] == start.isoformat()),
        None,
    )
    if plan is None or plan["coverage"] != "complete":
        raise ValueError("incomplete source/context coverage or grid conflict")
    snapshots = plan["sources"]
    selected_paths = [Path(s["path"]) for s in snapshots]
    read = read_low_frequency_window(
        selected_paths,
        start,
        end,
        None,
        None,
        margin_seconds=CONTEXT_SECONDS,
        sample_rate_hz=100,
    )
    if inspect_sources(selected_paths) != snapshots:
        raise ValueError("source changed during analysis read")
    block = read.block
    a, b = read.analysis_start, read.analysis_stop
    raw = block.data[a:b].copy()
    distance = block.metadata.channel_distance_m.copy()
    if not np.array_equal(distance, np.asarray(snapshots[0]["distance_m"])):
        raise ValueError("reader did not preserve the complete native channel grid")
    if raw.shape != (sample_count, len(distance)) or not np.isfinite(block.data).all():
        raise ValueError("invalid whole-window waveform")
    valid = np.ptp(raw, axis=0) > 0
    recognition = valid & scan_channel_mask(distance, profile)
    # Clamp all detector bands to the conventional 0.5-40 Hz protocol.
    changes = {}
    for prefix in ("stalta", "coherence", "trajectory"):
        changes[prefix + "_freq_low_hz"] = max(
            0.5, min(39.0, getattr(profile, prefix + "_freq_low_hz"))
        )
        changes[prefix + "_freq_high_hz"] = min(
            40.0, getattr(profile, prefix + "_freq_high_hz")
        )
    changes["energy_bands_hz"] = ((0.5, 5.0), (5.0, 20.0), (20.0, 40.0))
    detector_profile = profile.model_copy(update=changes)
    evidence = []
    qualified_scores = []
    backend = select_backend(profile.device)
    for core in range(0, sample_count, 2000):
        core_stop = min(core + 2000, sample_count)
        first = a + core - CONTEXT_SECONDS * 100
        stop = a + core_stop + CONTEXT_SECONDS * 100
        detection_data = block.data[first:stop].copy()
        detection_data[:, ~recognition] = 0
        local = replace(block, data=detection_data, sample_offset=first)
        screened = screen_block(local, detector_profile, backend)
        qualified = {id(c) for event in screened.events for c in event.evidence}
        for candidate in screened.candidates:
            if not a + core <= candidate.peak_sample < a + core_stop:
                continue
            record = asdict(candidate)
            record["peak_utc"] = (
                block.metadata.start_time_utc
                + timedelta(seconds=candidate.peak_sample / 100)
            ).isoformat()
            record["qualified"] = id(candidate) in qualified
            for key in ("start_sample", "end_sample", "peak_sample"):
                record[key] -= a
            evidence.append(record)
            if record["qualified"]:
                qualified_scores.append(candidate.score)
    if skip_empty and window_seconds == 30 and not merge_regions(
        evidence, distance, recognition, sample_count, round(profile.merge_gap_seconds * 100)
    ):
        return {"skipped": True, "reason": "no_qualified_regions"}
    filtered = sosfiltfilt(
        butter(4, (0.5, 40), btype="bandpass", fs=100, output="sos"),
        block.data.astype(float),
        axis=0,
    )[a:b]
    features = _measure(filtered, distance, valid, recognition)
    metadata = {
        "schema_version": 1,
        "algorithm_version": ALGORITHM_VERSION if window_seconds == 30 else "whole-window-v1",
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "sample_rate_hz": 100,
        "shape": [len(distance), sample_count],
        "unit": block.metadata.unit,
        "grid_sha256": hashlib.sha256(
            np.asarray(distance, dtype="<f8").tobytes()
        ).hexdigest(),
        "recognition_range_m": {
            "start": profile.scan_distance_start_m,
            "stop": profile.scan_distance_stop_m,
        },
        "sources": snapshots,
        "profile": profile.model_dump(mode="json"),
        "valid_channel_mask": valid.tolist(),
        "recognition_channel_mask": recognition.tolist(),
        "read_sources": list(read.sources),
        "read_start_utc": min(source["read_start_utc"] for source in read.sources),
        "read_end_utc": max(source["read_stop_utc"] for source in read.sources),
        "conditioning_start_utc": block.metadata.start_time_utc.isoformat(),
        "conditioning_end_utc": (
            block.metadata.start_time_utc + timedelta(seconds=len(block.data) / 100)
        ).isoformat(),
        "features": features,
        "warnings": list(read.warnings),
        "site_timezone": site_timezone,
        "candidate": bool(qualified_scores),
        "score": max(qualified_scores, default=0.0),
        "evidence_count": len(evidence),
    }
    metadata["input_sha256"] = compute_input_sha256(metadata)
    visual_metadata, visual_files = create_visualization(
        raw, filtered, distance, start, recognition, block.metadata.unit, evidence,
        merge_gap_seconds=profile.merge_gap_seconds, site_timezone=site_timezone,
    )
    metadata.update(visual_metadata)
    encoded = _json(evidence)
    if len(encoded) > MAX_EVIDENCE_BYTES:
        raise ValueError("evidence exceeds 32 MiB limit")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic(output_dir / "evidence.json", encoded)
    _atomic(output_dir / "metadata_visual_v2.json", _json(metadata))
    import io

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        waveform=raw.T,
        distance_m=distance,
        relative_seconds=np.arange(sample_count) / 100,
        valid_channel_mask=valid,
        recognition_channel_mask=recognition,
        region_id=np.where(distance < 299, 0, np.where(distance < 499, 1, 2)).astype(
            "i1"
        ),
    )
    _atomic(output_dir / "waveform.npz", buffer.getvalue())
    for name, payload in visual_files.items():
        _atomic(output_dir / name, payload)
    kinds = [
        ("waveform", "waveform.npz", "application/octet-stream"),
        ("metadata", "metadata_visual_v2.json", "application/json"),
        ("evidence", "evidence.json", "application/json"),
        ("waterfall", "waterfall_raw_v2.png", "image/png"),
        ("waterfall_filtered", "waterfall_filtered_v2.png", "image/png"),
        ("filtered_waveform", "filtered_waveform_v2.npz", "application/octet-stream"),
    ]
    if len(kinds) > MAX_ARTIFACTS:
        raise ValueError("artifact manifest exceeds 8 items")
    result = dict(
        metadata,
        artifacts=[
            {
                "kind": k,
                "path": n,
                "media_type": m,
                "sha256": _file_sha256(output_dir / n),
            }
            for k, n, m in kinds
        ],
    )
    _atomic(output_dir / "result.json", _json(result))
    return result
