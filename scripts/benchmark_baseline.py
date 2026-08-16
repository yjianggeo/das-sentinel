"""Baseline performance benchmark for the DAS Sentinel pipeline.

Generates (or reuses) a synthetic 1000 channel x 1000 Hz x 60 s ZD HDF5
file, runs the full analysis pipeline on it, and records wall time, peak
RSS, backend selection, library versions, and repeat-run parity into a
Markdown report.

Usage:
    python scripts/benchmark_baseline.py --device cpu --out docs/acceptance-report.md
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import platform
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "src"))

from das_sentinel.config import ProcessingProfile
from das_sentinel.pipeline import analyze_file
from das_sentinel.readers.base import ReaderRegistry
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.signal.backend import select_backend

N_CHANNELS = 1000
SAMPLE_RATE_HZ = 1000.0
DURATION_S = 60.0
REPEAT_RUNS = 2
_PERFORMANCE_SECTION_START = "<!-- DAS_SENTINEL_PERFORMANCE_BENCHMARK:START -->"
_PERFORMANCE_SECTION_END = "<!-- DAS_SENTINEL_PERFORMANCE_BENCHMARK:END -->"


class PerformanceGateError(RuntimeError):
    """Raised when the fixed benchmark does not meet an acceptance budget."""


def summarize_runs(runs: list[dict]) -> dict:
    """Summarize repeat-run timing, memory, and output identity evidence."""
    identities = {
        (tuple(run["event_ids"]), tuple(run["event_labels"])) for run in runs
    }
    return {
        "mean_wall_seconds": round(
            sum(run["wall_seconds"] for run in runs) / len(runs), 2
        ),
        "max_peak_rss_bytes": max(run["peak_rss_bytes"] for run in runs),
        "parity": len(identities) == 1,
    }


def enforce_gate(summary: dict) -> None:
    """Reject inclusive performance-limit breaches and repeat-output changes."""
    if not summary["parity"]:
        raise PerformanceGateError("repeat-run output parity failed")
    if summary["max_peak_rss_bytes"] >= 4 * 1024**3:
        raise PerformanceGateError("peak RSS must remain below 4 GB")
    if summary["mean_wall_seconds"] >= 60.0:
        raise PerformanceGateError("mean wall time must remain below 60 seconds")


def generate_baseline_file(path: Path) -> None:
    """Write the deterministic synthetic baseline acquisition file."""
    n_time = int(SAMPLE_RATE_HZ * DURATION_S)
    rng = np.random.default_rng(20260726)
    start = datetime(2026, 7, 26, tzinfo=UTC)
    times = start.timestamp() * 1_000_000 + np.arange(n_time) * 1_000_000 / SAMPLE_RATE_HZ

    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        dataset = raw.create_dataset(
            "RawData",
            shape=(n_time, N_CHANNELS),
            dtype=np.float32,
            chunks=(6000, N_CHANNELS),
        )
        # Write in 6 s slabs to keep generation memory bounded.
        slab = 6_000
        for offset in range(0, n_time, slab):
            stop = min(offset + slab, n_time)
            block = rng.normal(0.0, 0.02, size=(stop - offset, N_CHANNELS)).astype(np.float32)
            dataset[offset:stop] = block
        # One coherent multi-channel pulse, one single-channel spike.
        pulse = np.hanning(200).astype(np.float32) * 8.0
        dataset[30_000 : 30_000 + 200, 400:430] += pulse[:, None]
        dataset[45_000 : 45_000 + 200, 700] += pulse
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = SAMPLE_RATE_HZ
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 0
        raw.attrs["RawDataUnit"] = "strain_rate"


class _RssSampler:
    """Background peak-RSS sampler (getrusage only reports at call time)."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self.peak_bytes = 0
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    @staticmethod
    def _current_peak_bytes() -> int:
        try:
            import resource

            # Linux reports KiB; the acceptance host is Linux.
            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except ImportError:
            pass
        if sys.platform == "win32":
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
        return 0

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._current_peak_bytes())
            time.sleep(self._interval)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def run_once(path: Path, device: str, artifact_root: Path) -> dict:
    backend = select_backend(device)
    registry = ReaderRegistry([ZdHdf5Reader()])
    reader = registry.resolve(path)
    with _RssSampler() as rss:
        started = time.perf_counter()
        result = analyze_file(
            path,
            reader,
            ProcessingProfile(device=device),
            backend,
            artifact_root,
        )
        wall_s = time.perf_counter() - started
    return {
        "wall_seconds": round(wall_s, 2),
        "peak_rss_bytes": rss.peak_bytes,
        "requested_backend": device,
        "actual_backend": result.actual_backend,
        "fallback_reason": result.backend_fallback_reason,
        "event_count": len(result.events),
        "event_ids": [event.event_id for event in result.events],
        "event_labels": [event.label.value for event in result.events],
        "timings_ms": {k: round(v, 1) for k, v in sorted(result.timings_ms.items())},
    }


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("das-sentinel", "numpy", "scipy", "h5py", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def render_report(runs: list[dict], source: Path, device: str) -> str:
    summary = summarize_runs(runs)
    lines = [
        _PERFORMANCE_SECTION_START,
        "## Performance benchmark",
        "",
        f"- Date (UTC): {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- Source file: `{source}` ({N_CHANNELS} ch x {SAMPLE_RATE_HZ:.0f} Hz x {DURATION_S:.0f} s)",
        f"- Requested device: `{device}`",
        "",
        "## Versions",
        "",
    ]
    lines += [f"- {name}: {value}" for name, value in _versions().items()]
    lines += ["", "## Runs", ""]
    for index, run in enumerate(runs, start=1):
        lines += [
            f"### Run {index}",
            "",
            f"- Wall time: {run['wall_seconds']} s",
            f"- Peak RSS: {run['peak_rss_bytes'] / 1e6:.1f} MB",
            f"- Backend: {run['actual_backend']} (fallback: {run['fallback_reason'] or 'none'})",
            f"- Event IDs: {run['event_ids']}",
            f"- Events: {run['event_count']} {run['event_labels']}",
            "",
            "| Stage | ms |",
            "| --- | ---: |",
        ]
        lines += [f"| {stage} | {elapsed} |" for stage, elapsed in run["timings_ms"].items()]
        lines.append("")
    lines += [
        "## Gate summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        _PERFORMANCE_SECTION_END,
    ]
    return "\n".join(lines)


def write_report(out: Path, performance_section: str) -> None:
    """Append or replace only the benchmark section of an acceptance report."""
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    if (
        _PERFORMANCE_SECTION_START in existing
        and _PERFORMANCE_SECTION_END in existing
    ):
        before, _, remainder = existing.partition(_PERFORMANCE_SECTION_START)
        _, _, after = remainder.partition(_PERFORMANCE_SECTION_END)
        out.write_text(
            before.rstrip() + "\n\n" + performance_section + after,
            encoding="utf-8",
        )
        return
    legacy_heading = "\n## 性能基准\n"
    legacy_start = existing.find(legacy_heading)
    if legacy_start != -1:
        next_heading = existing.find("\n## ", legacy_start + len(legacy_heading))
        after = "" if next_heading == -1 else existing[next_heading:]
        updated = existing[:legacy_start].rstrip() + "\n\n" + performance_section + after
    elif existing:
        updated = existing.rstrip() + "\n\n" + performance_section + "\n"
    else:
        updated = performance_section + "\n"
    out.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--file", type=Path, default=None, help="reuse an existing baseline file")
    parser.add_argument("--out", type=Path, default=None, help="write a Markdown report here")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit nonzero unless parity, RSS, and mean wall-time gates pass",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("benchmark-data"),
        help="where the generated baseline file and artifacts live",
    )
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    source = args.file or (args.work_dir / "baseline-1000ch-1000hz-60s.h5")
    if not source.exists():
        print(f"generating {source} ...", flush=True)
        generate_baseline_file(source)

    artifact_root = args.work_dir / "artifacts"
    runs = [run_once(source, args.device, artifact_root) for _ in range(REPEAT_RUNS)]
    summary = summarize_runs(runs)
    report = render_report(runs, source, args.device)
    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_report(args.out, report)
        print(f"report written to {args.out}")
    (args.work_dir / "benchmark-result.json").write_text(
        json.dumps({"device": args.device, "runs": runs, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    if args.enforce:
        try:
            enforce_gate(summary)
        except PerformanceGateError as error:
            print(f"performance gate failed: {error}", file=sys.stderr)
            raise SystemExit(1) from error


if __name__ == "__main__":
    main()
