from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from das_sentinel.artifacts import ArtifactFile, ArtifactManifest
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.domain import Candidate, DasMetadata, EventRecord, Label
from das_sentinel.pipeline import FileAnalysis
from das_sentinel.readers.base import ProbeResult, ReaderRegistry
from das_sentinel.services.discovery import fingerprint
from das_sentinel.signal.backend import ArrayBackend, BackendUnavailable
from das_sentinel.worker.main import (
    CudaOutOfMemoryError,
    HttpxWorkerClient,
    Worker,
    WorkerClaim,
)


class FakeReader:
    def probe(self, path) -> ProbeResult:
        del path
        return ProbeResult(True, "fake")

    def read_metadata(self, path: Path) -> DasMetadata:
        return DasMetadata(
            source_path=str(path),
            start_time_utc=datetime(2026, 7, 30, tzinfo=UTC),
            sample_rate_hz=1000.0,
            channel_distance_m=np.arange(8, dtype=float),
            unit="strain_rate",
        )


class FakeProtocol:
    def __init__(self, claims: list[WorkerClaim | None]) -> None:
        self.claims = claims
        self.heartbeats: list[tuple[str, str, int, int]] = []
        self.results: list[tuple[str, dict[str, Any]]] = []
        self.failures: list[tuple[str, str, str, str, bool]] = []

    def claim(self, worker_id: str) -> WorkerClaim | None:
        del worker_id
        return self.claims.pop(0) if self.claims else None

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        numerator: int,
        denominator: int,
    ) -> None:
        self.heartbeats.append((job_id, worker_id, numerator, denominator))

    def submit_result(
        self,
        job_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.results.append((job_id, payload))

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry_eligible: bool,
    ) -> None:
        self.failures.append(
            (job_id, worker_id, error_code, error_message, retry_eligible)
        )


def _settings(tmp_path: Path, *, worker_device: str = "auto") -> Settings:
    app_data = tmp_path / "app-data"
    app_data.mkdir()
    return Settings(
        data_root=tmp_path,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'db.sqlite3'}",
        api_url="http://api.invalid",
        session_secret="s" * 32,
        worker_token="secret-" + "x" * 32,
        worker_device=worker_device,
        secure_cookies=False,
    )


def _claim(
    source: Path,
    *,
    profile: ProcessingProfile | None = None,
    source_fingerprint: str | None = None,
    source_size_bytes: int | None = None,
    source_mtime_ns: int | None = None,
) -> WorkerClaim:
    stat_result = source.stat()
    return WorkerClaim(
        id="job-1",
        source_file_id="source-1",
        source_path=str(source.resolve()),
        source_fingerprint=source_fingerprint or fingerprint(source),
        source_size_bytes=(
            stat_result.st_size if source_size_bytes is None else source_size_bytes
        ),
        source_mtime_ns=(
            stat_result.st_mtime_ns if source_mtime_ns is None else source_mtime_ns
        ),
        site_id="site-1",
        site_data_root=str(source.parent.resolve()),
        site_timezone="UTC",
        cable_type="urban",
        processing_profile_id="profile-1",
        profile_version=3,
        profile_payload=(profile or ProcessingProfile()).model_dump(mode="json"),
        pipeline_version="pipeline-v1",
        attempt_count=1,
        worker_id="worker-a",
    )


def _analysis(
    source: Path,
    backend: ArrayBackend,
    artifact_root: Path,
) -> FileAnalysis:
    event_id = "00000000-0000-0000-0000-000000000001"
    event = EventRecord(
        event_id=event_id,
        start_sample=10,
        end_sample=20,
        peak_sample=15,
        channel_start=1,
        channel_stop=3,
        evidence=(
            Candidate(
                detector="energy",
                start_sample=10,
                end_sample=20,
                peak_sample=15,
                channel_start=1,
                channel_stop=3,
                score=0.8,
                metrics={"ratio": 4.0},
            ),
        ),
        features={"duration_seconds": 0.01},
        label=Label.UNKNOWN,
        decision_score=0.55,
        reasons=("low margin",),
    )
    event_dir = artifact_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, ArtifactFile] = {}
    suffixes = {
        "waterfall_raw": ".png",
        "waterfall_filtered": ".png",
        "waveform": ".png",
        "psd": ".png",
        "stft": ".png",
        "evidence": ".json",
    }
    for kind, suffix in suffixes.items():
        path = event_dir / f"{kind}{suffix}"
        content = kind.encode()
        path.write_bytes(content)
        files[kind] = ArtifactFile(
            kind=kind,
            path=path,
            media_type="application/json" if suffix == ".json" else "image/png",
            sha256=hashlib.sha256(content).hexdigest(),
        )
    return FileAnalysis(
        source_path=str(source),
        source_fingerprint=fingerprint(source),
        actual_backend=backend.name,
        backend_fallback_reason=backend.unavailable_reason,
        metadata_warnings=(),
        events=(event,),
        audit_candidate_count=1,
        timings_ms={"total": 12.5},
        artifacts=(ArtifactManifest(event_id=event_id, files=files),),
    )


def _worker(
    tmp_path: Path,
    protocol: FakeProtocol,
    analyzer,
    *,
    settings: Settings | None = None,
    selector=None,
    heartbeat_interval_seconds: float = 0.01,
) -> Worker:
    return Worker(
        settings=settings or _settings(tmp_path),
        client=protocol,
        worker_id="worker-a",
        readers=ReaderRegistry([FakeReader()]),
        analyzer=analyzer,
        backend_selector=selector,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


def test_worker_returns_idle_when_no_job(tmp_path: Path):
    protocol = FakeProtocol([None])
    worker = _worker(
        tmp_path,
        protocol,
        lambda *args: pytest.fail("analysis must not run"),
    )

    outcome = worker.run_once()

    assert outcome.status == "idle"
    assert protocol.heartbeats == []
    assert protocol.results == []
    assert protocol.failures == []


def test_worker_submits_success_result_and_holds_no_heartbeat_thread(
    tmp_path: Path,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([_claim(source)])
    analysis_entered = threading.Event()
    release_analysis = threading.Event()

    def analyze(path, reader, profile, backend, artifact_root, context):
        del reader, profile, context
        analysis_entered.set()
        assert release_analysis.wait(2)
        return _analysis(Path(path), backend, Path(artifact_root))

    worker = _worker(tmp_path, protocol, analyze)
    completed: list[Any] = []
    runner = threading.Thread(target=lambda: completed.append(worker.run_once()))
    runner.start()
    assert analysis_entered.wait(1)
    release_analysis.set()
    runner.join(2)

    assert completed[0].status == "succeeded"
    assert protocol.heartbeats[0] == ("job-1", "worker-a", 0, 1)
    assert protocol.results[0][0] == "job-1"
    payload = protocol.results[0][1]
    assert payload["source_file_id"] == "source-1"
    assert payload["processing_profile_id"] == "profile-1"
    assert payload["profile_version"] == 3
    assert payload["pipeline_version"] == "pipeline-v1"
    assert payload["events"][0]["classification"]["label"] == "unknown"
    assert len(payload["artifacts"]) == 6
    assert worker.heartbeat_thread is None
    assert not any(
        thread.name.startswith("das-heartbeat-job-1")
        for thread in threading.enumerate()
    )


def test_settings_device_overrides_profile_device(tmp_path: Path):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    profile = ProcessingProfile(device="cuda")
    protocol = FakeProtocol([_claim(source, profile=profile)])
    selected: list[str] = []

    def select(device: str) -> ArrayBackend:
        selected.append(device)
        return ArrayBackend("cpu", np)

    worker = _worker(
        tmp_path,
        protocol,
        lambda path, reader, profile, backend, artifact_root, context: _analysis(
            Path(path), backend, Path(artifact_root)
        ),
        settings=_settings(tmp_path, worker_device="cpu"),
        selector=select,
    )

    assert worker.run_once().status == "succeeded"
    assert selected == ["cpu"]
    assert protocol.results[0][1]["requested_backend"] == "cpu"


def test_auto_settings_uses_profile_device(tmp_path: Path):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol(
        [_claim(source, profile=ProcessingProfile(device="cuda"))]
    )
    selected: list[str] = []

    def select(device: str) -> ArrayBackend:
        selected.append(device)
        return ArrayBackend("cuda", object())

    worker = _worker(
        tmp_path,
        protocol,
        lambda path, reader, profile, backend, artifact_root, context: _analysis(
            Path(path), backend, Path(artifact_root)
        ),
        selector=select,
    )

    assert worker.run_once().status == "succeeded"
    assert selected == ["cuda"]


def test_worker_falls_back_once_when_cuda_is_unavailable(tmp_path: Path):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol(
        [_claim(source, profile=ProcessingProfile(device="cuda"))]
    )
    selected: list[str] = []
    analyses: list[str] = []

    def select(device: str) -> ArrayBackend:
        selected.append(device)
        if device == "cuda":
            raise BackendUnavailable("driver absent")
        return ArrayBackend("cpu", np)

    def analyze(path, reader, profile, backend, artifact_root, context):
        del reader, profile, context
        analyses.append(backend.name)
        return _analysis(Path(path), backend, Path(artifact_root))

    outcome = _worker(
        tmp_path,
        protocol,
        analyze,
        selector=select,
    ).run_once()

    assert outcome.status == "succeeded"
    assert outcome.actual_backend == "cpu"
    assert outcome.backend_fallback_reason == "cuda_unavailable"
    assert selected == ["cuda", "cpu"]
    assert analyses == ["cpu"]
    assert protocol.results[0][1]["backend_fallback_reason"] == "cuda_unavailable"


def test_worker_retries_exact_cuda_oom_once_on_cpu_and_clears_cache(
    tmp_path: Path,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol(
        [_claim(source, profile=ProcessingProfile(device="cuda"))]
    )
    cleared: list[bool] = []

    class CudaBackend(ArrayBackend):
        def clear_cache(self) -> None:
            cleared.append(True)

    cuda = CudaBackend("cuda", object())
    cpu = ArrayBackend("cpu", np)
    analyses: list[str] = []

    def analyze(path, reader, profile, backend, artifact_root, context):
        del reader, profile, context
        analyses.append(backend.name)
        if backend.name == "cuda":
            raise CudaOutOfMemoryError("allocation failed")
        return _analysis(Path(path), backend, Path(artifact_root))

    outcome = _worker(
        tmp_path,
        protocol,
        analyze,
        selector=lambda device: cuda if device == "cuda" else cpu,
    ).run_once()

    assert outcome.status == "succeeded"
    assert outcome.actual_backend == "cpu"
    assert outcome.backend_fallback_reason == "cuda_out_of_memory"
    assert analyses == ["cuda", "cpu"]
    assert cleared == [True]


def test_non_cuda_exception_is_failed_without_cpu_retry_and_next_job_runs(
    tmp_path: Path,
):
    corrupt = tmp_path / "corrupt.h5"
    good = tmp_path / "good.h5"
    corrupt.write_bytes(b"corrupt")
    good.write_bytes(b"good")
    protocol = FakeProtocol(
        [
            replace(_claim(corrupt), id="job-bad"),
            replace(
                _claim(good),
                id="job-good",
                source_file_id="source-good",
            ),
        ]
    )
    calls: list[str] = []

    def analyze(path, reader, profile, backend, artifact_root, context):
        del reader, profile, context
        calls.append(Path(path).name + ":" + backend.name)
        if Path(path) == corrupt:
            raise OSError("corrupt hdf5 /secret/raw")
        return _analysis(Path(path), backend, Path(artifact_root))

    worker = _worker(
        tmp_path,
        protocol,
        analyze,
        selector=lambda device: ArrayBackend("cpu", np),
    )

    first = worker.run_once()
    second = worker.run_once()

    assert first.status == "failed"
    assert second.status == "succeeded"
    assert calls == ["corrupt.h5:cpu", "good.h5:cpu"]
    assert protocol.failures[0][0] == "job-bad"
    assert protocol.failures[0][2] == "analysis_failed"
    assert "Traceback" not in protocol.failures[0][3]
    assert len(protocol.failures[0][3]) <= 1024


def test_worker_releases_claimed_source_handle_after_analysis_failure(
    tmp_path: Path,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([_claim(source)])
    claimed_streams = []

    class TrackingRegistry(ReaderRegistry):
        def resolve(self, value):
            claimed_streams.append(value)
            return super().resolve(value)

    def fail_analysis(*args):
        del args
        raise OSError("analysis failed")

    worker = Worker(
        settings=_settings(tmp_path),
        client=protocol,
        worker_id="worker-a",
        readers=TrackingRegistry([FakeReader()]),
        analyzer=fail_analysis,
        backend_selector=lambda device: ArrayBackend("cpu", np),
        heartbeat_interval_seconds=0.01,
    )

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert claimed_streams
    assert all(stream.closed for stream in claimed_streams)


def test_result_submission_failure_is_structured_and_stops_heartbeat(
    tmp_path: Path,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")

    class RejectingProtocol(FakeProtocol):
        def submit_result(self, job_id, payload):
            del job_id, payload
            raise RuntimeError("database unavailable")

    protocol = RejectingProtocol([_claim(source)])
    worker = _worker(
        tmp_path,
        protocol,
        lambda path, reader, profile, backend, artifact_root, context: _analysis(
            Path(path), backend, Path(artifact_root)
        ),
        selector=lambda device: ArrayBackend("cpu", np),
    )

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "persistence_failed"
    assert protocol.failures[0][2] == "persistence_failed"
    assert worker.heartbeat_thread is None


def test_worker_rejects_analyzer_backend_identity_mismatch(tmp_path: Path):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([_claim(source)])

    def mismatched(path, reader, profile, backend, artifact_root, context):
        del reader, profile, context
        analysis = _analysis(Path(path), backend, Path(artifact_root))
        return replace(analysis, actual_backend="cuda")

    outcome = _worker(
        tmp_path,
        protocol,
        mismatched,
        selector=lambda device: ArrayBackend("cpu", np),
    ).run_once()

    assert outcome.status == "failed"
    assert outcome.error_code == "analysis_failed"
    assert protocol.results == []


@pytest.mark.parametrize("replacement_kind", ["file", "ancestor"])
def test_worker_binds_reader_and_analysis_to_claimed_source_handle(
    tmp_path: Path,
    replacement_kind: str,
):
    site_root = tmp_path / "site"
    site_root.mkdir()
    source = site_root / "input.h5"
    source.write_bytes(b"internal")
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_source = outside_root / "input.h5"
    outside_source.write_bytes(b"external")
    timestamp_ns = 1_800_000_000_000_000_000
    os.utime(source, ns=(timestamp_ns, timestamp_ns))
    os.utime(outside_source, ns=(timestamp_ns, timestamp_ns))
    claim = _claim(source)
    protocol = FakeProtocol([claim])
    observed: list[bytes] = []
    claimed_streams = []
    replaced = False

    def replace_claim_path() -> None:
        nonlocal replaced
        if replaced:
            return
        replaced = True
        if replacement_kind == "file":
            try:
                source.rename(site_root / "original.h5")
            except OSError:
                # A held Windows file handle deliberately blocks replacement.
                return
            outside_source.rename(source)
            return
        try:
            site_root.rename(tmp_path / "site-original")
        except OSError:
            # A held Windows directory handle deliberately blocks replacement.
            return
        try:
            site_root.symlink_to(outside_root, target_is_directory=True)
        except OSError:
            # A held Windows directory handle deliberately blocks replacement.
            (tmp_path / "site-original").rename(site_root)

    def read_marker(value) -> bytes:
        stream = getattr(value, "stream", value)
        if hasattr(stream, "read"):
            position = stream.tell()
            stream.seek(0)
            content = stream.read()
            stream.seek(position)
            return content
        return Path(stream).read_bytes()

    class ReplacingRegistry:
        def resolve(self, value):
            claimed_streams.append(value)
            replace_claim_path()
            observed.append(read_marker(value))
            return reader

    class TrackingReader(FakeReader):
        def read_metadata(self, value):
            observed.append(read_marker(value))
            return super().read_metadata(source)

    reader = TrackingReader()
    stable_helper = tmp_path / "stable.h5"
    stable_helper.write_bytes(b"stable")

    def analyze(value, reader, profile, backend, artifact_root, context):
        del reader, profile, context
        observed.append(read_marker(value))
        result = _analysis(stable_helper, backend, Path(artifact_root))
        return replace(
            result,
            source_path=claim.source_path,
            source_fingerprint=claim.source_fingerprint,
        )

    worker = Worker(
        settings=_settings(tmp_path),
        client=protocol,
        worker_id="worker-a",
        readers=ReplacingRegistry(),
        analyzer=analyze,
        backend_selector=lambda device: ArrayBackend("cpu", np),
        heartbeat_interval_seconds=0.01,
    )

    outcome = worker.run_once()

    assert outcome.status == "succeeded"
    assert observed == [b"internal", b"internal", b"internal"]
    assert all(stream.closed for stream in claimed_streams)
    assert b"external" not in observed


@pytest.mark.parametrize(
    "claim_change",
    [
        {"source_size_bytes": 999},
        {"source_mtime_ns": 999},
        {"source_fingerprint": "0" * 64},
        {"profile_payload": {"device": "cuda", "unknown": True}},
    ],
)
def test_claim_source_and_profile_mismatch_fail_before_analysis(
    tmp_path: Path,
    claim_change: dict[str, Any],
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([replace(_claim(source), **claim_change)])
    worker = _worker(
        tmp_path,
        protocol,
        lambda *args: pytest.fail("invalid claim must not be analyzed"),
    )

    outcome = worker.run_once()

    assert outcome.status == "failed"
    assert protocol.failures[0][2] == "invalid_claim"
    assert protocol.results == []
    assert worker.heartbeat_thread is None


def test_http_client_uses_only_worker_token_header_and_strict_claim_validation(
    tmp_path: Path,
):
    httpx = pytest.importorskip("httpx")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                **_claim(tmp_path / "input.h5").to_dict(),
            },
        )

    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    transport = httpx.MockTransport(handler)
    client = HttpxWorkerClient(
        "http://test",
        "top-secret",
        transport=transport,
    )

    claim = client.claim("worker-a")

    assert claim is not None
    assert seen[0].headers["x-worker-token"] == "top-secret"
    assert "cookie" not in seen[0].headers
    assert "top-secret" not in repr(client)


def test_result_payload_is_json_serializable_with_numpy_scalars(tmp_path):
    import json

    from das_sentinel.worker.main import _result_payload

    source = tmp_path / "source.h5"
    source.write_bytes(b"payload-test")
    claim = _claim(source)
    candidate = Candidate(
        detector="stalta",
        start_sample=np.int64(10),
        end_sample=np.int64(20),
        peak_sample=np.int64(15),
        channel_start=np.int64(1),
        channel_stop=np.int64(4),
        score=np.float64(0.9),
        metrics={"peak_ratio": np.float32(3.5)},
        quality_flags=("synthetic",),
    )
    event = EventRecord(
        event_id="event-1",
        start_sample=np.int64(10),
        end_sample=np.int64(20),
        peak_sample=np.int64(15),
        channel_start=np.int64(1),
        channel_stop=np.int64(4),
        evidence=(candidate,),
        features={"duration_s": np.float64(0.5)},
        label=Label.EARTHQUAKE,
        decision_score=0.9,
        reasons=("test",),
    )
    analysis = FileAnalysis(
        source_path=str(source),
        source_fingerprint="fp",
        actual_backend="cpu",
        backend_fallback_reason=None,
        metadata_warnings=(),
        events=(event,),
        audit_candidate_count=1,
        timings_ms={"total": np.float64(12.5)},
        artifacts=(),
    )

    payload = _result_payload(
        claim,
        datetime(2026, 7, 30, tzinfo=UTC),
        1000.0,
        analysis,
        requested_backend="cpu",
        actual_backend="cpu",
        fallback_reason=None,
    )

    encoded = json.dumps(payload)
    assert "event-1" in encoded


def test_internal_result_schema_accepts_string_feature_values():
    from das_sentinel.api.schemas import WorkerEventResult

    result = WorkerEventResult(
        event_uid="event-1",
        start_sample=10,
        end_sample=20,
        channel_start=1,
        channel_stop=4,
        features={
            "duration_s": 0.5,
            "site_id": "8069e40a-839a-4f8d-9b1e-aeadb6151bd1",
            "cable_type": "downhole",
        },
        classification={
            "label": "earthquake",
            "decision_score": 0.9,
            "method": "rules",
            "reasons": ["test"],
        },
    )

    assert result.features["site_id"] == "8069e40a-839a-4f8d-9b1e-aeadb6151bd1"
    assert result.features["duration_s"] == 0.5
