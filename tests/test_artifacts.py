from __future__ import annotations

import hashlib
import json
import threading
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
from das_sentinel.domain import Candidate, EventRecord

EVENT_ID = "123e4567-e89b-12d3-a456-426614174000"


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
        pipeline_version="v1",
        backend_name="cpu",
        rule_output={"label": "traffic", "score": 0.9},
        model_output={"label": "traffic", "score": float("nan")},
    )


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
    assert evidence["source_range"]["sample_rate_hz"] == 100.0
    assert evidence["quality_flags"]["bad_channel_count"] == 1
    assert evidence["input_sha256"]["raw"] == hashlib.sha256(data.raw_data.tobytes()).hexdigest()
    assert evidence["input_sha256"]["filtered"] == hashlib.sha256(data.filtered_data.tobytes()).hexdigest()
    assert evidence["input_sha256"]["raw"] != evidence["input_sha256"]["filtered"]
    assert "NaN" not in manifest.files["evidence"].path.read_text("utf-8")


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
