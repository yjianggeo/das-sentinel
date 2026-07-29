from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from das_sentinel.artifacts import ArtifactManifest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.signal.backend import select_backend
from das_sentinel.signal.preprocess import PreprocessedBlock


def _write_boundary_file(path: Path) -> np.ndarray:
    sample_rate_hz = 10.0
    sample_count = 420
    channel_count = 6
    start = datetime(2026, 7, 26, tzinfo=UTC)
    rng = np.random.default_rng(20260726)
    raw_data = rng.normal(0.0, 0.05, (sample_count, channel_count)).astype(np.float32)
    raw_data[185:196, 1:5] += np.hanning(11).astype(np.float32)[:, None] * 8.0
    times = start.timestamp() * 1_000_000 + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=raw_data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"
    return raw_data


def _profile() -> ProcessingProfile:
    return ProcessingProfile(
        block_seconds=20.0,
        overlap_seconds=5.0,
        stalta_freq_low_hz=0.5,
        stalta_freq_high_hz=4.0,
        coherence_freq_low_hz=0.5,
        coherence_freq_high_hz=4.0,
        trajectory_freq_low_hz=0.5,
        trajectory_freq_high_hz=4.0,
        energy_bands_hz=((0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
        artifact_pad_seconds=5.0,
    )


def _fixed_stalta(block, profile):
    del profile
    candidates = []
    block_stop = block.sample_offset + block.data.shape[0]
    if block.sample_offset <= 190 < block_stop:
        candidates.append(
            Candidate(
                detector="stalta",
                start_sample=185,
                end_sample=196,
                peak_sample=190,
                channel_start=1,
                channel_stop=5,
                score=0.95,
            )
        )
    if block.sample_offset == 300:
        candidates.append(
            Candidate(
                detector="stalta",
                start_sample=320,
                end_sample=325,
                peak_sample=322,
                channel_start=1,
                channel_stop=5,
                score=0.70,
            )
        )
    return candidates


def _no_candidates(*args):
    del args
    return []


def test_pipeline_merges_cross_block_event_and_preserves_artifact_inputs(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "boundary.h5"
    expected_raw = _write_boundary_file(source_path)
    captured = []

    def capture_artifacts(data, event, features, out_dir):
        del features, out_dir
        captured.append((data, event))
        return ArtifactManifest(event.event_id, {})

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(pipeline, "render_event_artifacts", capture_artifacts)

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
        pipeline.AnalysisContext(
            site_id="well-a",
            site_timezone="Asia/Shanghai",
            profile_version="profile-7",
            pipeline_version="pipeline-3",
            cable_type="downhole",
            device_config="interrogator-a",
            rule_profile={"traffic_velocity_mps": (3.0, 45.0)},
        ),
    )

    assert len(result.events) == 1
    event = result.events[0]
    assert (event.start_sample, event.end_sample, event.peak_sample) == (185, 196, 190)
    assert len(event.evidence) == 2
    assert result.audit_candidate_count == 3
    assert result.actual_backend == "cpu"
    assert result.backend_fallback_reason is None
    assert result.metadata_warnings == ()
    source_stat = source_path.resolve().stat()
    fingerprint_payload = json.dumps(
        (str(source_path.resolve()), source_stat.st_size, source_stat.st_mtime_ns),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert result.source_fingerprint == hashlib.sha256(fingerprint_payload).hexdigest()
    assert set(result.timings_ms) == {
        "read",
        "preprocess",
        "detect",
        "fuse",
        "classify",
        "render",
        "total",
    }
    assert all(value >= 0 for value in result.timings_ms.values())
    assert len(result.artifacts) == 1
    assert len(captured) == 1

    artifact_data, rendered_event = captured[0]
    assert rendered_event == event
    assert artifact_data.sample_offset == 135
    np.testing.assert_array_equal(artifact_data.raw_data, expected_raw[135:246])
    assert artifact_data.filtered_data.shape == artifact_data.raw_data.shape
    assert not np.array_equal(artifact_data.filtered_data, artifact_data.raw_data)
    assert artifact_data.profile_version == "profile-7"
    assert artifact_data.pipeline_version == "pipeline-3"
    assert artifact_data.model_version == "rules-only"
    assert artifact_data.backend_name == "cpu"
    assert artifact_data.site_timezone == "Asia/Shanghai"
    assert event.features["site_id"] == "well-a"
    assert event.features["cable_type"] == "downhole"
    assert event.features["device_config"] == "interrogator-a"


def test_pipeline_result_is_stable_except_for_recorded_timings(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    source_path = tmp_path / "stable.h5"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        lambda data, event, features, out_dir: ArtifactManifest(event.event_id, {}),
    )

    first = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "first",
    )
    second = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "second",
    )

    assert first == replace(second, timings_ms={name: -1.0 for name in second.timings_ms})
    assert first.events[0].event_id == second.events[0].event_id


def test_pipeline_event_identity_distinguishes_source_files(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    _write_boundary_file(first_path)
    _write_boundary_file(second_path)
    with h5py.File(second_path, "r+") as handle:
        handle["Acquisition/Raw[0]/RawData"][0, 0] += 1.0
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        lambda data, event, features, out_dir: ArtifactManifest(event.event_id, {}),
    )

    first = pipeline.analyze_file(
        first_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "first-artifacts",
    )
    second = pipeline.analyze_file(
        second_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "second-artifacts",
    )

    assert first.source_fingerprint != second.source_fingerprint
    assert first.events[0].event_id != second.events[0].event_id


class _NoImplicitNumpyConversion:
    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        del dtype, copy
        raise TypeError("implicit device-to-host conversion is forbidden")


class _NumpyCudaBackend:
    name = "cuda"
    xp = np
    unavailable_reason = None

    @staticmethod
    def asarray(value, **kwargs):
        return np.asarray(value, **kwargs)

    @staticmethod
    def asnumpy(value):
        if isinstance(value, _NoImplicitNumpyConversion):
            return value.values
        return np.asarray(value)


def test_pipeline_explicitly_transfers_event_window_before_cpu_features(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "cuda.h5"
    _write_boundary_file(source_path)
    captured = []
    original_preprocess = pipeline._preprocess_for_detector

    def device_preprocess(block, profile, backend, detector):
        prepared = original_preprocess(block, profile, backend, detector)
        return PreprocessedBlock(
            data=_NoImplicitNumpyConversion(prepared.data),
            bad_channel_mask=prepared.bad_channel_mask,
            backend_name="cuda",
            sample_offset=prepared.sample_offset,
            sample_rate_hz=prepared.sample_rate_hz,
            channel_distance_m=prepared.channel_distance_m,
        )

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(pipeline, "_preprocess_for_detector", device_preprocess)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        lambda data, event, features, out_dir: (
            captured.append(data) or ArtifactManifest(event.event_id, {})
        ),
    )

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        _NumpyCudaBackend(),
        tmp_path / "artifacts",
    )

    assert result.actual_backend == "cuda"
    assert len(result.events) == 1
    assert isinstance(captured[0].filtered_data, np.ndarray)


def test_analysis_context_has_stable_rules_only_defaults():
    from das_sentinel.pipeline import AnalysisContext

    context = AnalysisContext()

    assert context.site_id == "unknown-site"
    assert context.site_timezone == "UTC"
    assert context.profile_version == "profile-v1"
    assert context.pipeline_version == "pipeline-v1"
    assert context.cable_type == "unknown"
    assert context.device_config == "unknown"
    assert context.rule_profile == {}


def test_analysis_context_rejects_non_mapping_rule_profile():
    from das_sentinel.pipeline import AnalysisContext

    with pytest.raises(TypeError, match="rule_profile"):
        AnalysisContext(rule_profile=[])  # type: ignore[arg-type]


def test_pipeline_propagates_reader_warning_and_selected_backend_reason(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "20260727T000000.h5"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    fallback_backend = replace(
        select_backend("cpu"), unavailable_reason="configured CUDA backend unavailable"
    )

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        fallback_backend,
        tmp_path / "artifacts",
    )

    assert result.events == ()
    assert result.artifacts == ()
    assert result.audit_candidate_count == 0
    assert result.metadata_warnings == (
        "filename timestamp disagrees with RawDataTime",
    )
    assert result.backend_fallback_reason == "configured CUDA backend unavailable"
