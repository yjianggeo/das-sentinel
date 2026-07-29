from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.pipeline import FileAnalysis, analyze_file
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.signal.backend import select_backend


@dataclass(frozen=True)
class PerformanceResult:
    peak_rss_bytes: int
    wall_time_seconds: float
    analysis: FileAnalysis


def measure_analysis(path: Path, profile: ProcessingProfile, artifact_root: Path):
    started = time.perf_counter()
    analysis = analyze_file(
        path, ZdHdf5Reader(), profile, select_backend("cpu"), artifact_root
    )
    return PerformanceResult(
        _peak_rss_bytes(), time.perf_counter() - started, analysis
    )


def _peak_rss_bytes() -> int:
    if os.name != "nt":
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)

    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("page_fault_count", wintypes.DWORD),
            ("peak_working_set_size", ctypes.c_size_t),
            ("working_set_size", ctypes.c_size_t),
            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
            ("quota_paged_pool_usage", ctypes.c_size_t),
            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
            ("quota_non_paged_pool_usage", ctypes.c_size_t),
            ("pagefile_usage", ctypes.c_size_t),
            ("peak_pagefile_usage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    if not get_process_memory_info(
        get_current_process(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return int(counters.peak_working_set_size)


def _write_baseline_h5(path: Path) -> None:
    sample_rate_hz = 1_000.0
    sample_count = 60_000
    channel_count = 1_000
    start = datetime(2026, 7, 26, tzinfo=UTC)
    times = (
        start.timestamp() * 1_000_000
        + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    )
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        data = raw.create_dataset(
            "RawData",
            shape=(sample_count, channel_count),
            dtype=np.float32,
            chunks=(1_000, 100),
            fillvalue=0.0,
        )
        data[30_000:30_200, :8] = np.hanning(200).astype(np.float32)[
            :, None
        ]
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


def _fixed_capacity_event(block, profile):
    del profile
    if not (
        block.sample_offset <= 30_100
        < block.sample_offset + block.data.shape[0]
    ):
        return []
    return [
        Candidate(
            detector="stalta",
            start_sample=30_000,
            end_sample=30_200,
            peak_sample=30_100,
            channel_start=0,
            channel_stop=8,
            score=0.95,
        )
    ]


def _no_candidates(*args):
    del args
    return []


@pytest.mark.performance
def test_baseline_file_stays_under_memory_budget(tmp_path, monkeypatch):
    from das_sentinel import pipeline

    baseline_h5 = tmp_path / "baseline.h5"
    _write_baseline_h5(baseline_h5)
    detector_calls = {"energy": 0, "coherence": 0, "trajectory": 0}
    preprocess_bands = []
    original_energy = pipeline.detect_energy
    original_coherence = pipeline.detect_coherence
    original_trajectory = pipeline.detect_trajectories
    original_preprocess = pipeline._preprocess_for_band

    def tracked_energy(*args):
        detector_calls["energy"] += 1
        return original_energy(*args)

    def tracked_coherence(*args):
        detector_calls["coherence"] += 1
        return original_coherence(*args)

    def tracked_trajectory(*args):
        detector_calls["trajectory"] += 1
        return original_trajectory(*args)

    def tracked_preprocess(block, profile, backend, band):
        preprocess_bands.append(band)
        return original_preprocess(block, profile, backend, band)

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_capacity_event)
    monkeypatch.setattr(pipeline, "detect_energy", tracked_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", tracked_coherence)
    monkeypatch.setattr(pipeline, "detect_trajectories", tracked_trajectory)
    monkeypatch.setattr(pipeline, "_preprocess_for_band", tracked_preprocess)
    profile = ProcessingProfile(
        block_seconds=20.0,
        overlap_seconds=5.0,
        artifact_pad_seconds=0.1,
    )

    result = measure_analysis(baseline_h5, profile, tmp_path / "artifacts")

    assert ZdHdf5Reader().sample_count(baseline_h5) == 60_000
    assert len(ZdHdf5Reader().read_metadata(baseline_h5).channel_distance_m) == 1_000
    assert len(result.analysis.events) == 1
    assert len(result.analysis.artifacts) == 1
    assert detector_calls == {"energy": 4, "coherence": 4, "trajectory": 4}
    assert preprocess_bands.count((2.0, 40.0)) == 5
    assert preprocess_bands.count((0.5, 80.0)) == 4
    assert preprocess_bands.count((1.0, 40.0)) == 4
    assert preprocess_bands.count((1.0, 30.0)) == 4
    assert set(result.analysis.artifacts[0].files) == {
        "waterfall_raw",
        "waterfall_filtered",
        "waveform",
        "psd",
        "stft",
        "evidence",
    }
    assert all(
        artifact.path.exists()
        for artifact in result.analysis.artifacts[0].files.values()
    )
    assert result.peak_rss_bytes < 4 * 1024**3
    assert 0 < result.wall_time_seconds < 60
    if os.environ.get("DAS_PERFORMANCE_REPORT"):
        print(
            f"baseline peak RSS: {result.peak_rss_bytes / 1024**3:.3f} GiB; "
            f"wall time: {result.wall_time_seconds:.3f}s"
        )
