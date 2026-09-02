from __future__ import annotations

import errno
import hashlib
import json
import multiprocessing
import os
import threading
import warnings
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from das_sentinel.domain import Candidate, EventRecord
from PIL import Image

EVENT_ID = "123e4567-e89b-12d3-a456-426614174000"

# Spawned children re-import this module (and matplotlib) before signalling
# readiness; on slow storage such as a NAS that import alone can exceed 10 s.
_SPAWN_READY_S = 60


def _process_store_worker(root: str, content: bytes, ready, start, results) -> None:
    from das_sentinel.artifacts import ArtifactConflictError, ArtifactStore

    ready.set()
    start.wait(_SPAWN_READY_S)
    try:
        artifact = ArtifactStore(Path(root)).write(EVENT_ID, "evidence", content, "application/json")
        results.put(("success", artifact.sha256))
    except ArtifactConflictError:
        results.put(("conflict", None))


def _lock_worker(path: str, attempting, acquired, release, crash: bool) -> None:
    from das_sentinel.artifacts import _process_lock

    attempting.set()
    with _process_lock(Path(path)):
        acquired.set()
        if crash:
            os._exit(0)
        release.wait(_SPAWN_READY_S)


def _event() -> EventRecord:
    return EventRecord(
        event_id=EVENT_ID,
        start_sample=1_020,
        end_sample=1_080,
        peak_sample=1_050,
        channel_start=1,
        channel_stop=5,
        evidence=(Candidate("energy", 1_020, 1_080, 1_050, 1, 5, 0.9),),
    )


def _artifact_data():
    from das_sentinel.artifacts import EventArtifactData
    from das_sentinel.waterfalls import WaterfallFilter

    time = np.arange(160, dtype=np.float32) / 100.0
    raw = np.column_stack([np.sin(2 * np.pi * 8 * time + channel) for channel in range(6)]).astype(np.float32)
    filtered = raw * np.float32(0.25)
    return EventArtifactData(
        raw_data=raw,
        filtered_data=filtered,
        sample_offset=1_000,
        sample_rate_hz=100.0,
        channel_distance_m=np.arange(6, dtype=np.float32) * 2.0,
        bad_channel_mask=np.array([False, False, True, False, False, False]),
        start_time_utc=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        site_timezone="Asia/Shanghai",
        profile={"name": "default"},
        profile_version="profile-1",
        pipeline_version="v1",
        model_version="rules-only",
        backend_name="cpu",
        rule_output={"label": "traffic", "score": 0.9},
        model_output={"label": "traffic", "score": float("nan")},
        waterfall_filter=WaterfallFilter("energy", 0.5, 80.0),
    )


def test_filtered_waterfall_title_names_detector_and_exact_band():
    """Dropping filter provenance or rounding its literal frequencies must fail."""
    from das_sentinel.artifacts import (
        WaterfallPlotContext,
        filtered_waterfall_title,
        render_waterfall_png,
    )
    from das_sentinel.waterfalls import WaterfallFilter

    context = WaterfallPlotContext(
        sample_rate_hz=1000.0,
        site_timezone="Asia/Shanghai",
        profile_version="7",
        pipeline_version="v1",
        model_version="rules-only",
        filter=WaterfallFilter("stalta", 2.5, 40.25),
    )

    assert filtered_waterfall_title(context) == (
        "Filtered waterfall | Band-pass 2.5–40.25 Hz | "
        "primary detector=STA/LTA | profile=7 pipeline=v1 model=rules-only | "
        "1000 Hz | input unit"
    )
    assert render_waterfall_png(
        np.ones((2, 2), dtype=np.float32),
        np.array(
            [
                datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
                datetime(2026, 7, 26, 12, 0, 1, tzinfo=UTC),
            ]
        ),
        np.array([8.0, 12.0]),
        context,
        filtered=True,
    ).startswith(b"\x89PNG\r\n\x1a\n")


def test_filtered_waterfall_title_preserves_shortest_round_trip_precision():
    """Regresses six-significant-digit :g rounding while removing only useless .0."""
    from das_sentinel.artifacts import WaterfallPlotContext, filtered_waterfall_title
    from das_sentinel.waterfalls import WaterfallFilter

    context = WaterfallPlotContext(
        sample_rate_hz=1000.0,
        site_timezone="Asia/Shanghai",
        profile_version="7",
        pipeline_version="v1",
        model_version="rules-only",
        filter=WaterfallFilter("stalta", 0.30000000000000004, 40.0),
    )

    assert "Band-pass 0.30000000000000004–40 Hz" in filtered_waterfall_title(context)
    assert filtered_waterfall_title(context).endswith("1000 Hz | input unit")


def test_renders_complete_event_artifact_set_with_distinct_raw_and_filtered_inputs(tmp_path):
    """Using one input for both waterfall products must make this fail."""
    from das_sentinel.artifacts import render_event_artifacts

    data = _artifact_data()
    manifest = render_event_artifacts(data, _event(), {"rms": 1.0}, tmp_path)

    assert set(manifest.files) == {
        "waterfall_raw", "waterfall_filtered", "waveform", "psd", "stft", "evidence"
    }
    for artifact in manifest.files.values():
        assert artifact.path.exists()
        assert len(artifact.sha256) == 64
        assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    evidence = json.loads(manifest.files["evidence"].path.read_text("utf-8"))
    assert evidence["event_id"] == EVENT_ID
    assert evidence["profile_version"] == "profile-1"
    assert evidence["pipeline_version"] == "v1"
    assert evidence["model_version"] == "rules-only"
    assert evidence["source_range"]["sample_rate_hz"] == 100.0
    assert evidence["quality_flags"]["bad_channel_count"] == 1
    assert evidence["input_sha256"]["raw"] == hashlib.sha256(data.raw_data.tobytes()).hexdigest()
    assert evidence["input_sha256"]["filtered"] == hashlib.sha256(data.filtered_data.tobytes()).hexdigest()
    assert evidence["input_sha256"]["raw"] != evidence["input_sha256"]["filtered"]
    assert "NaN" not in manifest.files["evidence"].path.read_text("utf-8")


def test_png_artifacts_leave_a_clean_margin_around_plot_content(tmp_path):
    from das_sentinel.artifacts import render_event_artifacts

    manifest = render_event_artifacts(_artifact_data(), _event(), {}, tmp_path)
    for kind in ("waterfall_raw", "waterfall_filtered", "waveform", "psd", "stft"):
        pixels = np.asarray(Image.open(manifest.files[kind].path).convert("RGB"))
        border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
        assert np.all(border >= 250), kind


def test_artifacts_are_confined_to_uuid_event_directory_and_repeatable(tmp_path):
    """Path traversal or a non-idempotent artifact store must make this fail."""
    from das_sentinel.artifacts import render_event_artifacts

    first = render_event_artifacts(_artifact_data(), _event(), {}, tmp_path)
    second = render_event_artifacts(_artifact_data(), _event(), {}, tmp_path)

    root = tmp_path.resolve()
    assert all(artifact.path.parent == root / EVENT_ID for artifact in first.files.values())
    assert {kind: file.sha256 for kind, file in first.files.items()} == {
        kind: file.sha256 for kind, file in second.files.items()
    }
    with pytest.raises(ValueError, match="UUID"):
        render_event_artifacts(_artifact_data(), replace(_event(), event_id="../escape"), {}, tmp_path)


def test_rejects_event_outside_input_and_invalid_artifact_metadata(tmp_path):
    """Accepting a non-overlapping event or invalid acquisition metadata must make this fail."""
    from das_sentinel.artifacts import EventArtifactData, render_event_artifacts

    with pytest.raises(ValueError, match="overlap"):
        render_event_artifacts(_artifact_data(), replace(_event(), start_sample=2_000, end_sample=2_100), {}, tmp_path)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        EventArtifactData(**{**_artifact_data().__dict__, "sample_rate_hz": 0.0})


def test_array_zero_time_is_not_shifted_by_global_sample_offset():
    """Adding sample_offset twice must make this fail."""
    from das_sentinel.artifacts import _event_selection, _sample_dates

    data, event = _artifact_data(), _event()
    selection, _ = _event_selection(data, event)

    assert _sample_dates(data, selection)[0] == data.start_time_utc.replace(microsecond=200_000)


def test_robust_beam_rejects_a_high_amplitude_good_channel():
    """Changing the beam to select a single good channel must make this fail."""
    from das_sentinel.artifacts import _representative_beam

    values = np.array([[1.0, 1.1, 1000.0], [2.0, 2.1, -1000.0]], dtype=np.float32)

    np.testing.assert_allclose(_representative_beam(values), [1.1, 2.0], atol=0.2)


def test_waterfall_mesh_preserves_nonuniform_channel_distance_coordinates():
    """Replacing physical-coordinate mesh rendering with an equal-spaced image must make this fail."""
    from das_sentinel.artifacts import _waterfall_mesh
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots()
    try:
        mesh = _waterfall_mesh(
            axis,
            np.ones((2, 3)),
            np.array([0.0, 1.0]),
            np.array([2.0, 6.0, 8.0]),
            0.0,
            1.0,
        )
        assert mesh.get_coordinates()[:, 0, 1].tolist() == [0.0, 4.0, 7.0, 9.0]
    finally:
        plt.close(figure)


def test_nan_arrays_are_preserved_and_evidence_hashes_the_stored_bytes(tmp_path):
    """Sanitizing stored input arrays or hashing different bytes must make this fail."""
    from das_sentinel.artifacts import EventArtifactData, render_event_artifacts

    data = _artifact_data()
    raw = data.raw_data.copy()
    raw[0, 0] = np.nan
    preserved = EventArtifactData(**{**data.__dict__, "raw_data": raw})
    assert np.isnan(raw[0, 0]) and np.isnan(preserved.raw_data[0, 0])

    manifest = render_event_artifacts(preserved, _event(), {}, tmp_path)
    evidence = json.loads(manifest.files["evidence"].path.read_text("utf-8"))
    assert evidence["input_sha256"]["raw"] == hashlib.sha256(preserved.raw_data.tobytes()).hexdigest()


def test_json_value_is_deterministic_for_supported_structured_values():
    """Stringifying arbitrary values or leaving arrays/enums unsafely encoded must make this fail."""
    from das_sentinel.artifacts import _json_value

    class Choice(Enum):
        ONE = "one"

    value = {"array": np.array([1, 2]), "choice": Choice.ONE, "when": datetime(2026, 1, 1, tzinfo=UTC), "path": Path("a"), "set": {"b", "a"}}
    assert _json_value(value) == {"array": [1, 2], "choice": "one", "when": "2026-01-01T00:00:00+00:00", "path": "a", "set": ["a", "b"]}
    with pytest.raises(TypeError):
        _json_value(object())


def test_short_zero_energy_event_renders_without_warnings_and_nonempty_pngs(tmp_path):
    """A log-scale warning or empty short-event product must make this fail."""
    from das_sentinel.artifacts import EventArtifactData, render_event_artifacts

    data = _artifact_data()
    tiny = EventArtifactData(**{**data.__dict__, "raw_data": np.zeros((1, 6), dtype=np.float32), "filtered_data": np.zeros((1, 6), dtype=np.float32)})
    event = replace(_event(), start_sample=1_000, end_sample=1_001)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        manifest = render_event_artifacts(tiny, event, {"dominant_frequency_hz": 0.0}, tmp_path)
    assert all(artifact.path.stat().st_size > 0 for kind, artifact in manifest.files.items() if kind != "evidence")


def test_store_rejects_concurrent_different_content_without_wrong_digest(tmp_path):
    """Returning a digest for bytes another writer replaced must make this fail."""
    from das_sentinel.artifacts import ArtifactConflictError, ArtifactStore

    store = ArtifactStore(tmp_path)
    first = store.write(EVENT_ID, "evidence", b"first", "application/json")
    outcomes: list[object] = []

    def write_conflict() -> None:
        try:
            outcomes.append(store.write(EVENT_ID, "evidence", b"second", "application/json"))
        except ArtifactConflictError as error:
            outcomes.append(error)

    thread = threading.Thread(target=write_conflict)
    thread.start()
    thread.join()
    assert first.sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()
    assert len(outcomes) == 1 and isinstance(outcomes[0], ArtifactConflictError)


def test_store_rejects_lock_symlink_without_modifying_external_file(tmp_path):
    """Following a pre-positioned lock symlink outside the artifact root must make this fail."""
    from das_sentinel.artifacts import ArtifactStore

    root = tmp_path / "artifacts"
    event_dir = root / EVENT_ID
    event_dir.mkdir(parents=True)
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"sentinel-bytes")
    lock_path = event_dir / "evidence.json.lock"
    try:
        lock_path.symlink_to(sentinel)
    except OSError as error:
        if error.errno in {errno.EACCES, errno.EPERM} or getattr(error, "winerror", None) == 1314:
            pytest.skip(f"symbolic links are not permitted: {error}")
        raise

    with pytest.raises(OSError):
        ArtifactStore(root).write(EVENT_ID, "evidence", b"artifact", "application/json")

    assert sentinel.read_bytes() == b"sentinel-bytes"
    assert not (event_dir / "evidence.json").exists()


def test_store_serializes_real_processes_for_conflicting_bytes(tmp_path):
    """Removing the cross-process lock must allow both writers to incorrectly succeed."""
    context = multiprocessing.get_context("spawn")
    ready = [context.Event(), context.Event()]
    start, results = context.Event(), context.Queue()
    processes = [
        context.Process(target=_process_store_worker, args=(str(tmp_path), content, signal, start, results))
        for content, signal in ((b"first", ready[0]), (b"second", ready[1]))
    ]
    for process in processes:
        process.start()
    assert all(signal.wait(_SPAWN_READY_S) for signal in ready)
    start.set()
    for process in processes:
        process.join(_SPAWN_READY_S)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=5), results.get(timeout=5)]
    successes = [digest for status, digest in outcomes if status == "success"]
    assert len(successes) == 1
    assert sum(status == "conflict" for status, _ in outcomes) == 1
    target = tmp_path / EVENT_ID / "evidence.json"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == successes[0]


def test_process_lock_excludes_spawned_contender_until_release(tmp_path):
    context = multiprocessing.get_context("spawn")
    lock = str(tmp_path / "artifact.lock")
    first_attempt, first_acquired, release = context.Event(), context.Event(), context.Event()
    second_attempt, second_acquired = context.Event(), context.Event()
    first = context.Process(target=_lock_worker, args=(lock, first_attempt, first_acquired, release, False))
    second = context.Process(target=_lock_worker, args=(lock, second_attempt, second_acquired, release, False))
    first.start(); assert first_acquired.wait(_SPAWN_READY_S)
    second.start(); assert second_attempt.wait(_SPAWN_READY_S)
    assert not second_acquired.wait(0.3)
    release.set(); first.join(_SPAWN_READY_S); second.join(_SPAWN_READY_S)
    assert first.exitcode == second.exitcode == 0 and second_acquired.is_set()


def test_process_lock_recovers_after_crashed_owner_leaves_lock_file(tmp_path):
    context = multiprocessing.get_context("spawn")
    lock = str(tmp_path / "artifact.lock")
    attempted, acquired, release = context.Event(), context.Event(), context.Event()
    crashed = context.Process(target=_lock_worker, args=(lock, attempted, acquired, release, True))
    crashed.start(); assert acquired.wait(_SPAWN_READY_S); crashed.join(_SPAWN_READY_S)
    assert crashed.exitcode == 0 and Path(lock).exists()
    next_attempt, next_acquired = context.Event(), context.Event()
    recovered = context.Process(target=_lock_worker, args=(lock, next_attempt, next_acquired, release, False))
    recovered.start(); assert next_acquired.wait(_SPAWN_READY_S)
    release.set(); recovered.join(_SPAWN_READY_S)
    assert recovered.exitcode == 0


def test_waveform_axis_contains_event_and_candidate_exclusive_boundaries(monkeypatch):
    from das_sentinel import artifacts
    from matplotlib import dates as mdates
    from matplotlib import pyplot as plt

    data = _artifact_data()
    candidate = Candidate("energy", 1_030, 1_070, 1_050, 1, 5, 0.9)
    event = replace(_event(), evidence=(candidate,))
    captured = {}

    def capture(figure):
        captured["axis"] = figure.axes[0]
        plt.close(figure)
        return b"png"

    monkeypatch.setattr(artifacts, "_png", capture)
    dates = artifacts._sample_dates(data, slice(20, 80))
    artifacts._waveform_png(np.zeros(60), dates, data, event, slice(20, 80))
    axis = captured["axis"]
    start = mdates.date2num(data.start_time_utc + timedelta(seconds=.2))
    end = mdates.date2num(data.start_time_utc + timedelta(seconds=.8))
    candidate_start = mdates.date2num(data.start_time_utc + timedelta(seconds=.3))
    candidate_end = mdates.date2num(data.start_time_utc + timedelta(seconds=.7))
    assert {round(line.get_xdata()[0], 10) for line in axis.lines} >= {round(start, 10), round(end, 10)}
    assert any(round(patch.get_x(), 10) == round(candidate_start, 10) and round(patch.get_x() + patch.get_width(), 10) == round(candidate_end, 10) for patch in axis.patches)
    assert axis.get_xlim()[0] < start and axis.get_xlim()[1] > end
