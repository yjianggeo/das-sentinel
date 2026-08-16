from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.pipeline import FileAnalysis, analyze_file
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.signal.backend import select_backend

_BENCHMARK_SPEC = spec_from_file_location(
    "benchmark_baseline",
    Path(__file__).resolve().parents[1] / "scripts" / "benchmark_baseline.py",
)
assert _BENCHMARK_SPEC is not None
assert _BENCHMARK_SPEC.loader is not None
benchmark_baseline = module_from_spec(_BENCHMARK_SPEC)
_BENCHMARK_SPEC.loader.exec_module(benchmark_baseline)


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
    detector_filter_bands = []
    artifact_preprocess_bands = []
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

    original_filter = pipeline.filter_prepared_signal

    def tracked_filter(prepared, low_hz, high_hz, backend):
        detector_filter_bands.append((low_hz, high_hz))
        return original_filter(prepared, low_hz, high_hz, backend)

    def tracked_preprocess(block, profile, backend, band):
        artifact_preprocess_bands.append(band)
        return original_preprocess(block, profile, backend, band)

    monkeypatch.setattr(pipeline, "detect_stalta", _fixed_capacity_event)
    monkeypatch.setattr(pipeline, "detect_energy", tracked_energy)
    monkeypatch.setattr(pipeline, "detect_coherence", tracked_coherence)
    monkeypatch.setattr(pipeline, "detect_trajectories", tracked_trajectory)
    monkeypatch.setattr(pipeline, "filter_prepared_signal", tracked_filter)
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
    assert detector_filter_bands.count((2.0, 40.0)) == 4
    assert detector_filter_bands.count((0.5, 80.0)) == 4
    assert detector_filter_bands.count((1.0, 40.0)) == 4
    assert detector_filter_bands.count((1.0, 30.0)) == 4
    assert artifact_preprocess_bands == [(2.0, 40.0)]
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


def test_benchmark_summary_enforces_mean_memory_and_repeat_identity_parity():
    runs = [
        {
            "wall_seconds": 55.0,
            "peak_rss_bytes": 900_000_000,
            "event_ids": ["event-a"],
            "event_labels": ["traffic"],
        },
        {
            "wall_seconds": 57.0,
            "peak_rss_bytes": 950_000_000,
            "event_ids": ["event-a"],
            "event_labels": ["traffic"],
        },
    ]

    assert benchmark_baseline.summarize_runs(runs) == {
        "mean_wall_seconds": 56.0,
        "max_peak_rss_bytes": 950_000_000,
        "parity": True,
    }


@pytest.mark.parametrize(
    "second_run",
    [
        {"event_ids": ["event-b"], "event_labels": ["traffic"]},
        {"event_ids": ["event-a"], "event_labels": ["earthquake"]},
    ],
)
def test_benchmark_summary_rejects_changed_event_identity_or_label(second_run):
    runs = [
        {
            "wall_seconds": 1.0,
            "peak_rss_bytes": 1,
            "event_ids": ["event-a"],
            "event_labels": ["traffic"],
        },
        {"wall_seconds": 1.0, "peak_rss_bytes": 1, **second_run},
    ]

    assert benchmark_baseline.summarize_runs(runs)["parity"] is False


@pytest.mark.parametrize(
    ("summary", "message"),
    [
        (
            {
                "mean_wall_seconds": 1.0,
                "max_peak_rss_bytes": 1,
                "parity": False,
            },
            "parity",
        ),
        (
            {
                "mean_wall_seconds": 1.0,
                "max_peak_rss_bytes": 4 * 1024**3,
                "parity": True,
            },
            "below 4 GB",
        ),
        (
            {
                "mean_wall_seconds": 60.0,
                "max_peak_rss_bytes": 1,
                "parity": True,
            },
            "below 60 seconds",
        ),
    ],
)
def test_benchmark_gate_rejects_each_inclusive_limit(summary, message):
    with pytest.raises(benchmark_baseline.PerformanceGateError, match=message):
        benchmark_baseline.enforce_gate(summary)


def test_benchmark_report_keeps_acceptance_evidence_and_appends_auditable_gate(
    tmp_path,
):
    report_path = tmp_path / "acceptance-report.md"
    report_path.write_text(
        "# Existing acceptance evidence\n\n## 性能基准\n\n"
        "Legacy benchmark is pending.\n\n"
        "## Accuracy boundary\n\nNo labelled samples.\n",
        encoding="utf-8",
    )
    runs = [
        {
            "wall_seconds": 55.0,
            "peak_rss_bytes": 900_000_000,
            "event_ids": ["event-a"],
            "event_labels": ["traffic"],
            "requested_backend": "cpu",
            "actual_backend": "cpu",
            "fallback_reason": None,
            "event_count": 1,
            "timings_ms": {"detect": 123.4},
        },
        {
            "wall_seconds": 57.0,
            "peak_rss_bytes": 950_000_000,
            "event_ids": ["event-a"],
            "event_labels": ["traffic"],
            "requested_backend": "cpu",
            "actual_backend": "cpu",
            "fallback_reason": None,
            "event_count": 1,
            "timings_ms": {"detect": 120.0},
        },
    ]

    performance_section = benchmark_baseline.render_report(
        runs,
        source=tmp_path / "baseline.h5",
        device="cpu",
    )
    benchmark_baseline.write_report(report_path, performance_section)

    written = report_path.read_text(encoding="utf-8")
    assert "# Existing acceptance evidence" in written
    assert "## Accuracy boundary" in written
    assert "Legacy benchmark is pending." not in written
    assert "## Performance benchmark" in written
    assert written.index("## Performance benchmark") < written.index(
        "## Accuracy boundary"
    )
    assert "1000 ch x 1000 Hz x 60 s" in written
    assert '"mean_wall_seconds": 56.0' in written
    assert '"max_peak_rss_bytes": 950000000' in written
    assert '"parity": true' in written
    assert "- Event IDs: ['event-a']" in written

    replacement_section = performance_section.replace(
        '"mean_wall_seconds": 56.0',
        '"mean_wall_seconds": 58.0',
    )
    benchmark_baseline.write_report(report_path, replacement_section)

    updated = report_path.read_text(encoding="utf-8")
    assert updated.count("<!-- DAS_SENTINEL_PERFORMANCE_BENCHMARK:START -->") == 1
    assert updated.index("## Performance benchmark") < updated.index(
        "## Accuracy boundary"
    )
    assert '"mean_wall_seconds": 58.0' in updated
    assert '"mean_wall_seconds": 56.0' not in updated


def _benchmark_run(event_id="event-a", label="traffic"):
    return {
        "wall_seconds": 1.0,
        "peak_rss_bytes": 1,
        "event_ids": [event_id],
        "event_labels": [label],
        "requested_backend": "cpu",
        "actual_backend": "cpu",
        "fallback_reason": None,
        "event_count": 1,
        "timings_ms": {"detect": 1.0},
    }


def test_benchmark_main_enforces_exactly_two_runs(tmp_path, monkeypatch):
    source = tmp_path / "baseline.h5"
    source.touch()
    calls = []

    def run_once(path, device, artifact_root):
        calls.append((path, device, artifact_root))
        return _benchmark_run()

    monkeypatch.setattr(benchmark_baseline, "run_once", run_once)
    monkeypatch.setattr(
        benchmark_baseline.sys,
        "argv",
        [
            "benchmark_baseline.py",
            "--device",
            "cpu",
            "--file",
            str(source),
            "--work-dir",
            str(tmp_path / "work"),
            "--out",
            str(tmp_path / "acceptance-report.md"),
            "--enforce",
        ],
    )

    benchmark_baseline.main()

    assert len(calls) == benchmark_baseline.REPEAT_RUNS == 2


def test_benchmark_main_exits_nonzero_when_enforced_gate_fails(tmp_path, monkeypatch):
    source = tmp_path / "baseline.h5"
    source.touch()
    run_count = 0

    def mismatched_run(*args):
        nonlocal run_count
        del args
        run_count += 1
        return _benchmark_run(event_id=f"event-{run_count}")

    monkeypatch.setattr(
        benchmark_baseline,
        "run_once",
        mismatched_run,
    )
    monkeypatch.setattr(
        benchmark_baseline.sys,
        "argv",
        [
            "benchmark_baseline.py",
            "--file",
            str(source),
            "--work-dir",
            str(tmp_path / "work"),
            "--enforce",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark_baseline.main()

    assert exc_info.value.code == 1
