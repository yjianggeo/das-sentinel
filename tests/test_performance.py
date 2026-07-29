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
from das_sentinel.pipeline import analyze_file
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.signal.backend import select_backend


@dataclass(frozen=True)
class PerformanceResult:
    peak_rss_bytes: int
    wall_time_seconds: float


def measure_analysis(path: Path, profile: ProcessingProfile, artifact_root: Path):
    started = time.perf_counter()
    analyze_file(path, ZdHdf5Reader(), profile, select_backend("cpu"), artifact_root)
    return PerformanceResult(_peak_rss_bytes(), time.perf_counter() - started)


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
    sample_rate_hz = 20.0
    sample_count = 400
    start = datetime(2026, 7, 26, tzinfo=UTC)
    rng = np.random.default_rng(110)
    values = rng.normal(0.0, 0.1, (sample_count, 8)).astype(np.float32)
    times = start.timestamp() * 1_000_000 + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=values)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


@pytest.mark.performance
def test_baseline_file_stays_under_memory_budget(tmp_path):
    baseline_h5 = tmp_path / "baseline.h5"
    _write_baseline_h5(baseline_h5)
    profile = ProcessingProfile(
        block_seconds=20.0,
        overlap_seconds=5.0,
        stalta_freq_low_hz=0.5,
        stalta_freq_high_hz=8.0,
        coherence_freq_low_hz=0.5,
        coherence_freq_high_hz=8.0,
        trajectory_freq_low_hz=0.5,
        trajectory_freq_high_hz=8.0,
        energy_bands_hz=((0.5, 2.0), (2.0, 5.0), (5.0, 8.0)),
        sta_seconds=0.25,
        lta_seconds=2.0,
        min_coincidence=4,
        stalta_threshold=100.0,
        energy_ratio_threshold=100.0,
        coherence_threshold=1.0,
        trajectory_threshold=1.0,
    )

    result = measure_analysis(baseline_h5, profile, tmp_path / "artifacts")

    assert result.peak_rss_bytes < 4 * 1024**3
    assert result.wall_time_seconds > 0
    if os.environ.get("DAS_PERFORMANCE_REPORT"):
        print(f"baseline wall time: {result.wall_time_seconds:.3f}s")
