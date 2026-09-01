from __future__ import annotations

import gc
import hashlib
import json
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest
from das_sentinel.artifacts import ArtifactManifest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate, Label
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.signal.backend import select_backend
from das_sentinel.signal.preprocess import PreparedSignal, PreprocessedBlock


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


def _write_default_stalta_file(path: Path) -> None:
    sample_rate_hz = 200.0
    sample_count = round(20.0 * sample_rate_hz)
    start = datetime(2026, 7, 26, tzinfo=UTC)
    rng = np.random.default_rng(20260730)
    raw_data = rng.normal(0.0, 0.02, (sample_count, 8)).astype(np.float32)
    event_start = round(15.0 * sample_rate_hz)
    event_samples = round(0.3 * sample_rate_hz)
    burst = (
        np.hanning(event_samples)
        * np.sin(2 * np.pi * 10.0 * np.arange(event_samples) / sample_rate_hz)
        * 20.0
    ).astype(np.float32)
    raw_data[event_start : event_start + event_samples, :4] += burst[:, None]
    times = (
        start.timestamp() * 1_000_000
        + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    )
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=raw_data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


def _write_history_stalta_file(path: Path) -> None:
    sample_rate_hz = 200.0
    sample_count = round(50.0 * sample_rate_hz)
    start = datetime(2026, 7, 26, tzinfo=UTC)
    rng = np.random.default_rng(20260731)
    raw_data = rng.normal(0.0, 0.02, (sample_count, 8)).astype(np.float32)
    for event_seconds in (22.0, 37.0):
        event_start = round(event_seconds * sample_rate_hz)
        event_samples = round(0.3 * sample_rate_hz)
        burst = (
            np.hanning(event_samples)
            * np.sin(
                2 * np.pi * 10.0 * np.arange(event_samples) / sample_rate_hz
            )
            * 20.0
        ).astype(np.float32)
        raw_data[event_start : event_start + event_samples, :4] += burst[
            :, None
        ]
    times = (
        start.timestamp() * 1_000_000
        + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    )
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=raw_data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


def _write_history_dominance_file(path: Path) -> None:
    sample_rate_hz = 10.0
    sample_count = 420
    channel_count = 6
    start = datetime(2026, 7, 26, tzinfo=UTC)
    raw_data = np.zeros((sample_count, channel_count), dtype=np.float32)
    raw_data[100, :4] = 20.0
    raw_data[200, :4] = 5.0
    times = (
        start.timestamp() * 1_000_000
        + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    )
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=raw_data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


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
                score=0.99 if block.sample_offset == 0 else 0.90,
                quality_flags=("edge_only",) if block.sample_offset == 0 else (),
            )
        )
    if block.sample_offset <= 322 < block_stop:
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


def _fixed_energy(block, profile):
    del profile
    if not (
        block.sample_offset <= 191
        < block.sample_offset + block.data.shape[0]
    ):
        return []
    return [
        Candidate(
            detector="energy",
            start_sample=184,
            end_sample=197,
            peak_sample=191,
            channel_start=1,
            channel_stop=5,
            score=0.80,
        )
    ]


def _no_candidates(*args):
    del args
    return []


_DETECTOR_PREPROCESSING_GOLDEN = [
    (
        "stalta",
        "float32",
        (350, 6),
        "27f77b2e47c0ae57254aea6956dc2a11e4222266ad0fd7d6131dc050a2831091",
        0,
        350,
        "cpu",
        "bool",
        (6,),
        "11635df4532364e5fa01ac9540563d9a833f881994f4d2f3ae2a2e9c87680163",
        (True, False, False, False, False, True),
    ),
    (
        "energy",
        "float32",
        (200, 6),
        "680c60f6ec80428a2e269b1096df604447137b342ee20e414b8f1e5388466bbd",
        150,
        200,
        "cpu",
        "bool",
        (6,),
        "11635df4532364e5fa01ac9540563d9a833f881994f4d2f3ae2a2e9c87680163",
        (True, False, False, False, False, True),
    ),
    (
        "coherence",
        "float32",
        (200, 6),
        "680c60f6ec80428a2e269b1096df604447137b342ee20e414b8f1e5388466bbd",
        150,
        200,
        "cpu",
        "bool",
        (6,),
        "11635df4532364e5fa01ac9540563d9a833f881994f4d2f3ae2a2e9c87680163",
        (True, False, False, False, False, True),
    ),
    (
        "trajectory",
        "float32",
        (200, 6),
        "680c60f6ec80428a2e269b1096df604447137b342ee20e414b8f1e5388466bbd",
        150,
        200,
        "cpu",
        "bool",
        (6,),
        "11635df4532364e5fa01ac9540563d9a833f881994f4d2f3ae2a2e9c87680163",
        (True, False, False, False, False, True),
    ),
]


def _detector_block_identity(detector, block):
    data = np.asarray(block.data)
    bad_channel_mask = np.asarray(block.bad_channel_mask)
    assert data.flags.c_contiguous
    assert bad_channel_mask.flags.c_contiguous
    return (
        detector,
        str(data.dtype),
        tuple(data.shape),
        hashlib.sha256(data.tobytes()).hexdigest(),
        block.sample_offset,
        data.shape[0],
        block.backend_name,
        str(bad_channel_mask.dtype),
        tuple(bad_channel_mask.shape),
        hashlib.sha256(bad_channel_mask.tobytes()).hexdigest(),
        tuple(bad_channel_mask.tolist()),
    )


def _assert_detector_preprocessing_golden(
    source_path, monkeypatch, *, tamper_detector_input=False
):
    from das_sentinel import pipeline

    captured = []

    def capture(detector, implementation):
        def recorded(block, *args):
            if tamper_detector_input:
                block = replace(block, data=np.zeros_like(block.data))
            captured.append(_detector_block_identity(detector, block))
            return implementation(block, *args)

        return recorded

    monkeypatch.setattr(pipeline, "detect_stalta", capture("stalta", _fixed_stalta))
    monkeypatch.setattr(pipeline, "detect_energy", capture("energy", _fixed_energy))
    monkeypatch.setattr(
        pipeline, "detect_coherence", capture("coherence", _no_candidates)
    )
    monkeypatch.setattr(
        pipeline, "detect_trajectories", capture("trajectory", _no_candidates)
    )
    block = ZdHdf5Reader().read_block(source_path, 0, 350)
    pipeline._detect_block(
        block,
        _profile(),
        select_backend("cpu"),
        {name: 0.0 for name in pipeline._TIMING_NAMES},
        logical_start=150,
    )

    assert captured == _DETECTOR_PREPROCESSING_GOLDEN


def _empty_artifact_manifest(data, event, features, out_dir):
    del data, features
    (Path(out_dir) / event.event_id).mkdir(parents=True)
    return ArtifactManifest(event.event_id, {})


class _PredictingBundle:
    def predict_probabilities(self, features):
        assert features["site_id"] == "site-a"
        return {
            Label.EARTHQUAKE: 0.01,
            Label.BLAST: 0.01,
            Label.TRAFFIC: 0.97,
            Label.NOISE: 0.01,
        }


def test_pipeline_uses_bound_model_and_records_model_evidence(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    captured = []
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        lambda data, event, features, out_dir: captured.append(data)
        or _empty_artifact_manifest(data, event, features, out_dir),
    )
    context = pipeline.AnalysisContext(
        site_id="site-a",
        processing_profile_id="profile-a",
        profile_version="1",
        pipeline_version="v1",
        model_version_id="model-a",
        model_version="7",
    )
    source_path = tmp_path / "source.h5"
    _write_boundary_file(source_path)

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
        context,
        _PredictingBundle(),
    )

    assert result.events[0].label == Label.TRAFFIC
    assert captured[0].model_version == "7"
    assert captured[0].model_output["probabilities"]["traffic"] == 0.97


def test_pipeline_keeps_profile_artifacts_separate_and_deterministic(
    tmp_path, monkeypatch
):
    """Omitting profile identity from fusion would make two profile runs collide."""
    from das_sentinel import pipeline

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        _empty_artifact_manifest,
    )
    source_path = tmp_path / "source.h5"
    artifact_root = tmp_path / "artifacts"
    _write_boundary_file(source_path)

    def analyze(profile_id: str):
        return pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            artifact_root,
            pipeline.AnalysisContext(
                site_id="site-a",
                processing_profile_id=profile_id,
                profile_version="1",
                pipeline_version="v1",
            ),
        )

    first = analyze("profile-a")
    changed = analyze("profile-b")
    repeat = analyze("profile-a")

    first_event_id = first.events[0].event_id
    changed_event_id = changed.events[0].event_id
    assert first_event_id != changed_event_id
    assert repeat.events[0].event_id == first_event_id
    assert (artifact_root / first_event_id).is_dir()
    assert (artifact_root / changed_event_id).is_dir()


def test_pipeline_wraps_bound_model_prediction_errors(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    class InvalidPredictingBundle:
        def predict_probabilities(self, features):
            del features
            raise ValueError("invalid probabilities")

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    source_path = tmp_path / "source.h5"
    _write_boundary_file(source_path)
    context = pipeline.AnalysisContext(
        model_version_id="model-a",
        model_version="7",
    )

    with pytest.raises(pipeline.InvalidModelError, match="prediction"):
        pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            tmp_path / "artifacts",
            context,
            InvalidPredictingBundle(),
        )


def test_pipeline_merges_cross_block_event_and_preserves_artifact_inputs(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "boundary.h5"
    expected_raw = _write_boundary_file(source_path)
    captured = []

    def capture_artifacts(data, event, features, out_dir):
        del features
        captured.append((data, event))
        (Path(out_dir) / event.event_id).mkdir(parents=True)
        return ArtifactManifest(event.event_id, {})

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _fixed_energy)
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
    assert (event.start_sample, event.end_sample, event.peak_sample) == (184, 197, 190)
    assert len(event.evidence) == 2
    assert {candidate.detector for candidate in event.evidence} == {"stalta", "energy"}
    assert next(
        candidate for candidate in event.evidence if candidate.detector == "stalta"
    ).quality_flags == ()
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
    assert artifact_data.sample_offset == 134
    np.testing.assert_array_equal(artifact_data.raw_data, expected_raw[134:247])
    assert artifact_data.filtered_data.shape == artifact_data.raw_data.shape
    assert not np.array_equal(artifact_data.filtered_data, artifact_data.raw_data)
    assert artifact_data.profile_version == "profile-7"
    assert artifact_data.pipeline_version == "pipeline-3"
    assert artifact_data.model_version == "rules-only"
    assert artifact_data.backend_name == "cpu"
    assert artifact_data.site_timezone == "Asia/Shanghai"
    assert artifact_data.waterfall_filter.detector == "stalta"
    assert artifact_data.waterfall_filter.low_hz == 0.5
    assert artifact_data.waterfall_filter.high_hz == 4.0
    assert artifact_data.profile["rule_profile"] == {
        "traffic_velocity_mps": (3.0, 45.0)
    }
    assert artifact_data.profile["profile_version"] == "profile-7"
    assert artifact_data.profile["rule_profile_sha256"]
    assert artifact_data.rule_output["rule_profile"] == {
        "traffic_velocity_mps": (3.0, 45.0)
    }
    assert (
        artifact_data.rule_output["rule_profile_sha256"]
        == artifact_data.profile["rule_profile_sha256"]
    )
    assert event.features["site_id"] == "well-a"
    assert event.features["cable_type"] == "downhole"
    assert event.features["device_config"] == "interrogator-a"


def test_pipeline_preprocessing_golden_preserves_review_identity(tmp_path, monkeypatch):
    """Catches preprocessing changes that drift review-visible pipeline output."""
    from das_sentinel import pipeline

    source_path = tmp_path / "golden-boundary.h5"
    _write_boundary_file(source_path)
    _assert_detector_preprocessing_golden(source_path, monkeypatch)
    captured: dict[str, str | None] = {}

    def capture_artifacts(data, event, features, out_dir):
        del features
        captured["raw_sha256"] = hashlib.sha256(
            data.raw_data.tobytes()
        ).hexdigest()
        captured["filtered_sha256"] = hashlib.sha256(
            data.filtered_data.tobytes()
        ).hexdigest()
        captured["model_output"] = data.model_output
        (Path(out_dir) / event.event_id).mkdir(parents=True)
        return ArtifactManifest(event.event_id, {})

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _fixed_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(pipeline, "render_event_artifacts", capture_artifacts)
    monkeypatch.setattr(
        pipeline, "_source_fingerprint", lambda path: "baseline-source-fingerprint"
    )
    monkeypatch.setattr(
        pipeline, "_ensure_source_unchanged", lambda path, expected: None
    )

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
        pipeline.AnalysisContext(
            site_id="well-a",
            site_timezone="Asia/Shanghai",
            processing_profile_id="profile-a",
            profile_version="profile-7",
            pipeline_version="pipeline-3",
            cable_type="downhole",
            device_config="interrogator-a",
            rule_profile={"traffic_velocity_mps": (3.0, 45.0)},
        ),
    )

    assert len(result.events) == 1
    event = result.events[0]
    assert event.event_id == "fb2d78ee-58b5-5ce8-a3ae-73394c82762a"
    assert (
        event.start_sample,
        event.end_sample,
        event.peak_sample,
        event.channel_start,
        event.channel_stop,
    ) == (184, 197, 190, 1, 5)
    assert event.label == Label.UNKNOWN
    assert round(event.decision_score, 12) == 0.5
    assert [
        (
            candidate.detector,
            candidate.start_sample,
            candidate.end_sample,
            candidate.peak_sample,
            candidate.channel_start,
            candidate.channel_stop,
            round(candidate.score, 12),
        )
        for candidate in event.evidence
    ] == [
        ("energy", 184, 197, 191, 1, 5, 0.8),
        ("stalta", 185, 196, 190, 1, 5, 0.9),
    ]
    assert captured == {
        "raw_sha256": "15da6dc903a195b76cf94de794e4818e1646d4fcab6ede1693380c95db6bd0f3",
        "filtered_sha256": "eef5c31d1e0a124511189229ab80d3cffc1e69d38195e7d8badcff4e247211d2",
        "model_output": None,
    }


def test_pipeline_preprocessing_golden_rejects_tampered_detector_blocks(
    tmp_path, monkeypatch
):
    source_path = tmp_path / "golden-boundary.h5"
    _write_boundary_file(source_path)

    with pytest.raises(AssertionError):
        _assert_detector_preprocessing_golden(
            source_path, monkeypatch, tamper_detector_input=True
        )


def test_pipeline_default_block_and_lta_detect_real_stalta_event(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    source_path = tmp_path / "default-stalta.h5"
    _write_default_stalta_file(source_path)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        _empty_artifact_manifest,
    )

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        ProcessingProfile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
    )

    assert len(result.events) == 1
    assert any(candidate.detector == "stalta" for candidate in result.events[0].evidence)


class _RecordingReader:
    def __init__(self):
        self.delegate = ZdHdf5Reader()
        self.block_ranges = []

    def probe(self, path):
        return self.delegate.probe(path)

    def sample_count(self, path):
        return self.delegate.sample_count(path)

    def read_metadata(self, path):
        return self.delegate.read_metadata(path)

    def read_block(self, path, sample_start, sample_stop, channel_slice=slice(None)):
        self.block_ranges.append((sample_start, sample_stop))
        return self.delegate.read_block(
            path, sample_start, sample_stop, channel_slice
        )


def test_pipeline_prepends_lta_history_and_canonically_owns_gap_events(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "history-stalta.h5"
    _write_history_stalta_file(source_path)
    reader = _RecordingReader()
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        _empty_artifact_manifest,
    )

    result = pipeline.analyze_file(
        source_path,
        reader,
        ProcessingProfile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
    )

    assert reader.block_ranges[:3] == [(0, 4_000), (1_000, 7_000), (4_000, 10_000)]
    assert len(result.events) == 2
    assert result.audit_candidate_count == 2
    assert [event.peak_sample // 200 for event in result.events] == [22, 37]


@pytest.mark.parametrize(
    ("detector_name", "detector_attribute"),
    [
        ("coherence", "detect_coherence"),
        ("trajectory", "detect_trajectories"),
    ],
)
def test_non_stalta_detectors_search_only_the_logical_block(
    tmp_path,
    monkeypatch,
    detector_name,
    detector_attribute,
):
    from das_sentinel import pipeline

    source_path = tmp_path / f"history-{detector_name}.h5"
    _write_history_dominance_file(source_path)

    def identity_prepare(block, profile, backend):
        del profile
        return PreparedSignal(
            data=backend.asarray(block.data),
            spectrum=None,
            bad_channel_mask=np.zeros(block.data.shape[1], dtype=bool),
            backend_name=backend.name,
            sample_offset=block.sample_offset,
            sample_rate_hz=block.metadata.sample_rate_hz,
            channel_distance_m=block.metadata.channel_distance_m,
        )

    def identity_filter(prepared, low_hz, high_hz, backend):
        del low_hz, high_hz
        return PreprocessedBlock(
            data=prepared.data,
            bad_channel_mask=prepared.bad_channel_mask,
            backend_name=backend.name,
            sample_offset=prepared.sample_offset,
            sample_rate_hz=prepared.sample_rate_hz,
            channel_distance_m=prepared.channel_distance_m,
        )

    def strongest_signal(block, *args):
        del args
        strength = np.max(np.abs(block.data), axis=1)
        peak = int(np.argmax(strength))
        if strength[peak] == 0:
            return []
        peak += block.sample_offset
        return [
            Candidate(
                detector=detector_name,
                start_sample=peak,
                end_sample=peak + 1,
                peak_sample=peak,
                channel_start=0,
                channel_stop=4,
                score=0.95,
            )
        ]

    monkeypatch.setattr(pipeline, "prepare_signal", identity_prepare)
    monkeypatch.setattr(pipeline, "filter_prepared_signal", identity_filter)
    monkeypatch.setattr(pipeline, "detect_stalta", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    monkeypatch.setattr(pipeline, detector_attribute, strongest_signal)
    monkeypatch.setattr(
        pipeline,
        "render_event_artifacts",
        _empty_artifact_manifest,
    )

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
    )

    assert result.audit_candidate_count == 2
    assert [event.peak_sample for event in result.events] == [100, 200]


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
        _empty_artifact_manifest,
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
        _empty_artifact_manifest,
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

    @property
    def shape(self):
        return self.values.shape


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
    original_preprocess = pipeline.preprocess_block

    def device_preprocess(block, profile, backend):
        prepared = original_preprocess(block, profile, backend)
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
    monkeypatch.setattr(pipeline, "preprocess_block", device_preprocess)
    def capture_device_artifacts(data, event, features, out_dir):
        captured.append(data)
        return _empty_artifact_manifest(data, event, features, out_dir)

    monkeypatch.setattr(
        pipeline, "render_event_artifacts", capture_device_artifacts
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


@pytest.mark.parametrize(
    ("model_version_id", "model_version"),
    [("model-a", None), (None, "7")],
)
def test_analysis_context_rejects_partial_model_identity(
    model_version_id, model_version
):
    from das_sentinel.pipeline import AnalysisContext

    with pytest.raises(ValueError, match="provided together"):
        AnalysisContext(
            model_version_id=model_version_id,
            model_version=model_version,
        )


def test_analysis_context_snapshots_canonical_rule_profile():
    from das_sentinel.pipeline import AnalysisContext

    source_profile = {"traffic_velocity_mps": (3.0, 45.0)}
    context = AnalysisContext(rule_profile=source_profile)
    source_profile["traffic_velocity_mps"] = (1.0, 2.0)
    canonical = b'{"traffic_velocity_mps":[3.0,45.0]}'

    assert context.rule_profile == {"traffic_velocity_mps": (3.0, 45.0)}
    assert context.rule_profile_sha256 == hashlib.sha256(canonical).hexdigest()


def test_analysis_context_recursively_freezes_nested_rule_profile():
    from das_sentinel.pipeline import AnalysisContext

    source_profile = {
        "nested": {"thresholds": [1.0, {"minimum": 2.0}]}
    }
    context = AnalysisContext(rule_profile=source_profile)
    original_hash = context.rule_profile_sha256

    with pytest.raises(TypeError):
        context.rule_profile["nested"]["new"] = 3.0
    with pytest.raises(TypeError):
        context.rule_profile["nested"]["thresholds"][1]["minimum"] = 9.0

    assert context.rule_profile_sha256 == original_hash
    assert context.rule_profile["nested"]["thresholds"] == (
        1.0,
        {"minimum": 2.0},
    )


@pytest.mark.parametrize(
    "rule_profile",
    [{"threshold": float("nan")}, {"unsupported": object()}],
)
def test_analysis_context_rejects_noncanonical_rule_profile(rule_profile):
    from das_sentinel.pipeline import AnalysisContext

    with pytest.raises(ValueError, match="canonical JSON"):
        AnalysisContext(rule_profile=rule_profile)


def test_pipeline_evidence_records_reproducible_rule_identity(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    source_path = tmp_path / "rule-evidence.h5"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _fixed_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    context = pipeline.AnalysisContext(
        profile_version="profile-reviewed-2",
        rule_profile={"traffic_velocity_mps": (3.0, 45.0)},
    )

    result = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        tmp_path / "artifacts",
        context,
    )

    evidence_path = result.artifacts[0].files["evidence"].path
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["profile"]["rule_profile"] == {
        "traffic_velocity_mps": [3.0, 45.0]
    }
    assert evidence["profile"]["profile_version"] == "profile-reviewed-2"
    assert evidence["rule_output"]["rule_profile_sha256"] == (
        context.rule_profile_sha256
    )


class _MutatingReader:
    def __init__(self, path):
        self.delegate = ZdHdf5Reader()
        self.path = path
        self.mutated = False

    def probe(self, path):
        return self.delegate.probe(path)

    def sample_count(self, path):
        return self.delegate.sample_count(path)

    def read_metadata(self, path):
        return self.delegate.read_metadata(path)

    def read_block(self, path, sample_start, sample_stop, channel_slice=slice(None)):
        block = self.delegate.read_block(
            path, sample_start, sample_stop, channel_slice
        )
        if not self.mutated:
            stat = self.path.stat()
            os.utime(
                self.path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            self.mutated = True
        return block


def test_pipeline_rejects_source_changed_during_block_reads(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    source_path = tmp_path / "changing-read.h5"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)

    with pytest.raises(pipeline.SourceChangedError, match="changed"):
        pipeline.analyze_file(
            source_path,
            _MutatingReader(source_path),
            _profile(),
            select_backend("cpu"),
            tmp_path / "artifacts",
        )


def test_pipeline_rejects_source_changed_while_rendering(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    source_path = tmp_path / "changing-render.h5"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _fixed_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)

    artifact_root = tmp_path / "artifacts"
    original_render = pipeline.render_event_artifacts

    def changing_render(data, event, features, out_dir):
        manifest = original_render(data, event, features, out_dir)
        stat = source_path.stat()
        os.utime(
            source_path,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )
        return manifest

    monkeypatch.setattr(pipeline, "render_event_artifacts", changing_render)

    with pytest.raises(pipeline.SourceChangedError, match="changed"):
        pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            artifact_root,
        )
    assert not [path for path in artifact_root.iterdir() if path.is_dir()]
    assert not list(artifact_root.glob(".das-staging-*"))


def _two_review_events(block, profile):
    del profile
    candidates = []
    for peak in (190, 320):
        if block.sample_offset <= peak < block.sample_offset + block.data.shape[0]:
            candidates.append(
                Candidate(
                    detector="stalta",
                    start_sample=peak - 5,
                    end_sample=peak + 6,
                    peak_sample=peak,
                    channel_start=1,
                    channel_stop=5,
                    score=0.95,
                )
            )
    return candidates


def test_pipeline_rolls_back_only_newly_promoted_event_directories(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "promotion-change.h5"
    artifact_root = tmp_path / "artifacts"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _two_review_events)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)
    original_promote = pipeline._promote_event_directory
    promoted = 0

    def changing_promote(*args, **kwargs):
        nonlocal promoted
        result = original_promote(*args, **kwargs)
        promoted += 1
        if promoted == 1:
            stat = source_path.stat()
            os.utime(
                source_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
        return result

    monkeypatch.setattr(pipeline, "_promote_event_directory", changing_promote)

    with pytest.raises(pipeline.SourceChangedError, match="changed"):
        pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            artifact_root,
        )

    assert not [path for path in artifact_root.iterdir() if path.is_dir()]
    assert not list(artifact_root.glob(".das-staging-*"))


def test_pipeline_reuses_and_never_rolls_back_preexisting_artifacts(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "preexisting.h5"
    artifact_root = tmp_path / "artifacts"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _fixed_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)

    first = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        artifact_root,
    )
    second = pipeline.analyze_file(
        source_path,
        ZdHdf5Reader(),
        _profile(),
        select_backend("cpu"),
        artifact_root,
    )
    original_files = {
        kind: (artifact.path, artifact.sha256)
        for kind, artifact in first.artifacts[0].files.items()
    }
    assert {
        kind: (artifact.path, artifact.sha256)
        for kind, artifact in second.artifacts[0].files.items()
    } == original_files

    original_render = pipeline.render_event_artifacts

    def changing_render(data, event, features, out_dir):
        manifest = original_render(data, event, features, out_dir)
        stat = source_path.stat()
        os.utime(
            source_path,
            ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
        )
        return manifest

    monkeypatch.setattr(pipeline, "render_event_artifacts", changing_render)
    with pytest.raises(pipeline.SourceChangedError, match="changed"):
        pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            artifact_root,
        )

    assert all(
        path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in original_files.values()
    )
    assert not list(artifact_root.glob(".das-staging-*"))


def test_post_commit_failure_cannot_delete_artifacts_reused_by_another_run(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "post-commit-race.h5"
    artifact_root = tmp_path / "artifacts"
    _write_boundary_file(source_path)
    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_stalta)
    monkeypatch.setattr(pipeline, "detect_energy", _fixed_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)

    original_commit = pipeline._ArtifactTransaction.commit
    first_committed = Event()
    second_finished = Event()

    def fail_first_after_commit(transaction):
        manifests = original_commit(transaction)
        if not first_committed.is_set():
            first_committed.set()
            if not second_finished.wait(timeout=20):
                raise TimeoutError("second run did not reuse committed artifacts")
            stat = source_path.stat()
            os.utime(
                source_path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            raise pipeline.SourceChangedError(
                "source changed after artifact commit"
            )
        return manifests

    monkeypatch.setattr(
        pipeline._ArtifactTransaction,
        "commit",
        fail_first_after_commit,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            pipeline.analyze_file,
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            artifact_root,
        )
        assert first_committed.wait(timeout=20)
        try:
            second = pipeline.analyze_file(
                source_path,
                ZdHdf5Reader(),
                _profile(),
                select_backend("cpu"),
                artifact_root,
            )
        finally:
            second_finished.set()
        with pytest.raises(pipeline.SourceChangedError, match="after artifact"):
            first.result(timeout=20)

    assert len(second.artifacts) == 1
    assert len(second.artifacts[0].files) == 6
    assert all(
        artifact.path.exists()
        and hashlib.sha256(artifact.path.read_bytes()).hexdigest()
        == artifact.sha256
        for artifact in second.artifacts[0].files.values()
    )
    assert not list(artifact_root.glob(".das-staging-*"))


def test_detect_block_releases_each_distinct_band_before_next(
    tmp_path, monkeypatch
):
    from das_sentinel import pipeline

    source_path = tmp_path / "bands.h5"
    _write_boundary_file(source_path)
    block = ZdHdf5Reader().read_block(source_path, 0, 200)
    profile = _profile().model_copy(
        update={
            "coherence_freq_low_hz": 0.5,
            "coherence_freq_high_hz": 4.0,
            "energy_bands_hz": ((1.0, 3.0),),
            "trajectory_freq_low_hz": 2.0,
            "trajectory_freq_high_hz": 4.0,
        }
    )
    references = []
    bands = []
    original_filter = pipeline.filter_prepared_signal

    def tracked_filter(prepared, low_hz, high_hz, backend):
        gc.collect()
        assert all(reference() is None for reference in references)
        filtered = original_filter(prepared, low_hz, high_hz, backend)
        references.append(weakref.ref(filtered))
        bands.append((low_hz, high_hz))
        return filtered

    monkeypatch.setattr(pipeline, "filter_prepared_signal", tracked_filter)
    monkeypatch.setattr(pipeline, "detect_stalta", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)

    pipeline._detect_block(
        block,
        profile,
        select_backend("cpu"),
        {name: 0.0 for name in pipeline._TIMING_NAMES},
    )

    assert bands == [(0.5, 4.0), (1.0, 3.0), (2.0, 4.0)]


def test_detect_block_uses_one_forward_fft_per_temporal_source(tmp_path, monkeypatch):
    """Catches preprocessing once per detector band instead of source range."""
    from das_sentinel import pipeline

    source_path = tmp_path / "fft-count.h5"
    _write_boundary_file(source_path)
    block = ZdHdf5Reader().read_block(source_path, 0, 200)
    profile = _profile().model_copy(
        update={
            "stalta_freq_low_hz": 0.5,
            "stalta_freq_high_hz": 1.0,
            "energy_bands_hz": ((1.0, 2.0),),
            "coherence_freq_low_hz": 2.0,
            "coherence_freq_high_hz": 3.0,
            "trajectory_freq_low_hz": 3.0,
            "trajectory_freq_high_hz": 4.0,
        }
    )
    backend = select_backend("cpu")
    calls = 0
    original_rfft = backend.xp.fft.rfft

    def counted_rfft(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_rfft(*args, **kwargs)

    monkeypatch.setattr(backend.xp.fft, "rfft", counted_rfft)
    monkeypatch.setattr(pipeline, "detect_stalta", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_energy", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_coherence", _no_candidates)
    monkeypatch.setattr(pipeline, "detect_trajectories", _no_candidates)

    pipeline._detect_block(
        block,
        profile,
        backend,
        {name: 0.0 for name in pipeline._TIMING_NAMES},
        logical_start=50,
    )

    assert calls == 2


@pytest.mark.parametrize("sample_rate_hz", [float("nan"), 0.0, -1.0])
def test_pipeline_rejects_invalid_metadata_sample_rate(
    tmp_path, monkeypatch, sample_rate_hz
):
    from das_sentinel import pipeline

    source_path = tmp_path / "invalid-rate.h5"
    _write_boundary_file(source_path)
    with h5py.File(source_path, "r+") as handle:
        handle["Acquisition/Raw[0]"].attrs["OutputDataRate"] = sample_rate_hz

    with pytest.raises(ValueError, match="sample rate"):
        pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            _profile(),
            select_backend("cpu"),
            tmp_path / "artifacts",
        )


@pytest.mark.parametrize(
    "profile",
    [
        _profile().model_copy(
            update={"stalta_freq_low_hz": 5.1, "stalta_freq_high_hz": 6.0}
        ),
        _profile().model_copy(update={"coherence_freq_high_hz": 5.1}),
        _profile().model_copy(
            update={"trajectory_freq_low_hz": 5.1, "trajectory_freq_high_hz": 6.0}
        ),
        _profile().model_copy(
            update={"energy_bands_hz": ((0.5, 1.0), (1.0, 5.1))}
        ),
    ],
)
def test_pipeline_rejects_detector_band_outside_nyquist(tmp_path, profile):
    from das_sentinel import pipeline

    source_path = tmp_path / "invalid-band.h5"
    _write_boundary_file(source_path)

    with pytest.raises(ValueError, match="Nyquist"):
        pipeline.analyze_file(
            source_path,
            ZdHdf5Reader(),
            profile,
            select_backend("cpu"),
            tmp_path / "artifacts",
        )


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
