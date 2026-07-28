from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

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
