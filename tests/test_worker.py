from __future__ import annotations

import hashlib
import os
import threading
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from das_sentinel.artifacts import ArtifactFile, ArtifactManifest
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.domain import Candidate, DasMetadata, EventRecord, Label
from das_sentinel.pipeline import FileAnalysis, SourceChangedError
from das_sentinel.readers.base import ProbeResult, ReaderRegistry
from das_sentinel.services.discovery import fingerprint
from das_sentinel.signal.backend import ArrayBackend, BackendUnavailable
from das_sentinel.worker.__main__ import run_forever
from das_sentinel.worker.main import (
    CudaOutOfMemoryError,
    HttpxWorkerClient,
    Worker,
    WorkerClaim,
    WorkerFatalProtocolError,
    WorkerTransientError,
    _failure_retry_eligible,
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


class PeriodicHeartbeatFailureProtocol(FakeProtocol):
    def __init__(self, claims, error: BaseException) -> None:
        super().__init__(claims)
        self.error = error
        self.heartbeat_calls = 0
        self.periodic_attempted = threading.Event()

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        numerator: int,
        denominator: int,
    ) -> None:
        self.heartbeat_calls += 1
        if self.heartbeat_calls == 1:
            return super().heartbeat(
                job_id,
                worker_id,
                numerator,
                denominator,
            )
        self.periodic_attempted.set()
        raise self.error


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
        pipeline_version="v2",
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
    model_version_id: str | None = None,
    model_version: str | None = None,
    model_artifact_path: str | None = None,
    model_bundle_sha256: str | None = None,
    model_feature_names: tuple[str, ...] | None = None,
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
        pipeline_version="v2",
        model_version_id=model_version_id,
        model_version=model_version,
        model_artifact_path=model_artifact_path,
        model_bundle_sha256=model_bundle_sha256,
        model_feature_names=model_feature_names,
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


@pytest.mark.parametrize("version", ["v1", "pipeline-v1", "v3"])
def test_worker_rejects_noncurrent_pipeline_before_source_io(tmp_path, monkeypatch, version):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([replace(_claim(source), pipeline_version=version)])
    source_opens = []

    def unexpected_source_open(*args, **kwargs):
        source_opens.append(args)
        raise RuntimeError("source must not be opened for another pipeline version")

    monkeypatch.setattr("das_sentinel.worker.main._open_anchored_source", unexpected_source_open)
    worker = _worker(tmp_path, protocol, lambda *args: pytest.fail("analysis must not run"))
    outcome = worker.run_once()

    assert outcome.error_code == "invalid_claim"
    assert source_opens == []
    assert protocol.heartbeats == []
    assert protocol.results == []
    assert len(protocol.failures) == 1
    assert protocol.failures[0][4] is False
    assert "pipeline version" in protocol.failures[0][3]


@pytest.mark.parametrize("status", [500, 502, 503])
def test_http_client_maps_server_errors_to_transient(status: int) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, request=request)
    )
    client = HttpxWorkerClient("http://api", "w" * 40, transport=transport)
    with pytest.raises(WorkerTransientError):
        client.claim("worker-1")


def _unsupported_compression_bundle() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", b"{}")
        archive.writestr("model.skops", b"invalid")
    raw = bytearray(buffer.getvalue())
    for signature, compression_offset in (
        (b"PK\x03\x04", 8),
        (b"PK\x01\x02", 10),
    ):
        offset = 0
        while (member_offset := raw.find(signature, offset)) >= 0:
            method_offset = member_offset + compression_offset
            raw[method_offset : method_offset + 2] = (99).to_bytes(
                2,
                "little",
            )
            offset = member_offset + len(signature)
    return bytes(raw)


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_http_client_maps_transport_failures_to_transient(error_type) -> None:
    def fail(request):
        raise error_type("offline", request=request)

    client = HttpxWorkerClient(
        "http://api",
        "w" * 40,
        transport=httpx.MockTransport(fail),
    )
    with pytest.raises(WorkerTransientError):
        client.claim("worker-1")


def test_http_client_maps_unauthorized_to_fatal() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, request=request)
    )
    client = HttpxWorkerClient("http://api", "w" * 40, transport=transport)
    with pytest.raises(WorkerFatalProtocolError):
        client.claim("worker-1")


def test_http_client_fatal_error_does_not_expose_response_body() -> None:
    secret = "sensitive-server-payload"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(422, text=secret, request=request)
    )
    client = HttpxWorkerClient("http://api", "w" * 40, transport=transport)

    with pytest.raises(WorkerFatalProtocolError) as raised:
        client.claim("worker-1")

    assert "422" in str(raised.value)
    assert secret not in str(raised.value)


@pytest.mark.parametrize("stage", ["initial_heartbeat", "periodic_heartbeat", "result", "fail"])
@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (409, "job is not running"),
        (409, "worker is not the job owner"),
        (404, "job not found: job-1"),
    ],
)
def test_lost_claim_does_not_stop_next_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, status: int, detail: str
) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    claims = [_claim(source), replace(_claim(source), id="job-2")]
    requests = []
    heartbeat_count = 0
    periodic_attempted = threading.Event()

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal heartbeat_count
        path = request.url.path
        requests.append(path)
        if path.endswith("/claim"):
            return httpx.Response(200, json=claims.pop(0).to_dict())
        if "/job-1/" in path:
            if path.endswith("/heartbeat"):
                heartbeat_count += 1
                if stage == "initial_heartbeat" or (
                    stage == "periodic_heartbeat" and heartbeat_count > 1
                ):
                    periodic_attempted.set()
                    return httpx.Response(status, json={"detail": detail})
            elif path.endswith(f"/{stage}"):
                return httpx.Response(status, json={"detail": detail})
        return httpx.Response(200, json={})

    analyzed = 0

    def analyze(path, reader, profile, backend, artifact_root, context):
        nonlocal analyzed
        del reader, profile, context
        analyzed += 1
        if analyzed == 1:
            if stage == "periodic_heartbeat":
                assert periodic_attempted.wait(1)
            elif stage == "fail":
                raise OSError("analysis failed after cancellation")
        return _analysis(Path(path), backend, Path(artifact_root))

    client = HttpxWorkerClient("http://api", "w" * 40, transport=httpx.MockTransport(respond))
    worker = _worker(
        tmp_path, client, analyze, selector=lambda _: ArrayBackend("cpu", np)
    )
    run_once = worker.run_once
    outcomes = []

    def record_outcome():
        outcome = run_once()
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(worker, "run_once", record_outcome)
    try:
        assert run_forever(worker, max_iterations=2, sleep=lambda _: None) == 2
    finally:
        client.close()

    assert [(outcome.status, outcome.job_id) for outcome in outcomes] == [
        ("claim_lost", "job-1"),
        ("succeeded", "job-2"),
    ]
    assert requests[-1] == "/api/v1/internal/jobs/job-2/result"
    assert requests.count("/api/v1/internal/jobs/claim") == 2
    assert requests.count("/api/v1/internal/jobs/job-1/fail") == (stage == "fail")
    assert worker.heartbeat_thread is None


@pytest.mark.parametrize("operation", ["claim", "heartbeat", "result", "fail"])
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (401, {"detail": "job is not running"}),
        (403, {"detail": "job is not running"}),
        (422, {"detail": "job is not running"}),
        (409, {"detail": "source fingerprint does not match claimed job"}),
        (404, {"detail": "job not found: another-job"}),
        (404, {"detail": "Not Found"}),
        (409, ["job is not running"]),
    ],
)
def test_unrelated_job_protocol_errors_still_stop_worker(operation, status, body) -> None:
    client = HttpxWorkerClient(
        "http://api", "w" * 40,
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body)),
    )
    calls = {
        "claim": lambda: client.claim("worker-a"),
        "heartbeat": lambda: client.heartbeat("job-1", "worker-a", 0, 1),
        "result": lambda: client.submit_result("job-1", {}),
        "fail": lambda: client.fail("job-1", "worker-a", "analysis_failed", "failed", True),
    }
    try:
        with pytest.raises(WorkerFatalProtocolError):
            calls[operation]()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("status", "detail"),
    [(409, "job is not running"), (409, "worker is not the job owner"), (404, "job not found: job-1")],
)
def test_claim_endpoint_errors_are_not_treated_as_lost_ownership(status, detail) -> None:
    client = HttpxWorkerClient(
        "http://api", "w" * 40,
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json={"detail": detail})),
    )
    try:
        with pytest.raises(WorkerFatalProtocolError):
            client.claim("worker-a")
    finally:
        client.close()


def test_non_json_job_conflict_remains_a_fatal_protocol_error() -> None:
    client = HttpxWorkerClient(
        "http://api", "w" * 40,
        transport=httpx.MockTransport(lambda _: httpx.Response(409, text="invalid response")),
    )
    try:
        with pytest.raises(WorkerFatalProtocolError):
            client.heartbeat("job-1", "worker-a", 0, 1)
    finally:
        client.close()


def test_invalid_claim_is_reported_nonretryable(tmp_path: Path) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol([replace(_claim(source), source_size_bytes=999)])
    outcome = _worker(
        tmp_path,
        protocol,
        lambda *args: pytest.fail("invalid claim must not be analyzed"),
    ).run_once()
    assert outcome.error_code == "invalid_claim"
    assert protocol.failures[0][4] is False


@pytest.mark.parametrize(
    "error",
    [SourceChangedError("changed"), ValueError("invalid"), TypeError("invalid")],
)
def test_irrecoverable_worker_failures_are_not_retry_eligible(error) -> None:
    assert _failure_retry_eligible(error) is False


def test_ordinary_analysis_failure_remains_retry_eligible() -> None:
    assert _failure_retry_eligible(OSError("temporary read failure")) is True


def test_transport_loss_after_claim_leaves_job_for_stale_recovery(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")

    class OfflineResultProtocol(FakeProtocol):
        def submit_result(self, job_id: str, payload: dict[str, Any]) -> None:
            del job_id, payload
            raise WorkerTransientError("offline")

    protocol = OfflineResultProtocol([_claim(source)])
    worker = _worker(
        tmp_path,
        protocol,
        lambda path, reader, profile, backend, artifact_root, context: _analysis(
            Path(path), backend, Path(artifact_root)
        ),
    )

    with pytest.raises(WorkerTransientError, match="offline"):
        worker.run_once()

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
    assert payload["pipeline_version"] == "v2"
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


@pytest.mark.parametrize("profile_device", ["cuda", "auto"])
def test_auto_settings_uses_cuda_when_profile_allows_it(
    tmp_path: Path,
    profile_device: str,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = FakeProtocol(
        [_claim(source, profile=ProcessingProfile(device=profile_device))]
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
    assert selected == [profile_device]
    assert protocol.results[0][1]["actual_backend"] == "cuda"


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


def test_transient_failure_report_error_drives_polling_backoff(tmp_path: Path) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")

    class OfflineFailureProtocol(FakeProtocol):
        def fail(self, *args, **kwargs) -> None:
            del args, kwargs
            raise WorkerTransientError("offline while reporting failure")

    protocol = OfflineFailureProtocol([_claim(source)])

    def fail_analysis(*args):
        del args
        raise OSError("analysis failed")

    sleeps = []
    run_forever(
        _worker(tmp_path, protocol, fail_analysis),
        max_iterations=1,
        sleep=sleeps.append,
    )

    assert sleeps == [1.0]
    assert protocol.results == []


def test_fatal_failure_report_error_terminates_polling(tmp_path: Path) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")

    class UnauthorizedFailureProtocol(FakeProtocol):
        def fail(self, *args, **kwargs) -> None:
            del args, kwargs
            raise WorkerFatalProtocolError("unauthorized while reporting failure")

    protocol = UnauthorizedFailureProtocol([_claim(source)])

    def fail_analysis(*args):
        del args
        raise OSError("analysis failed")

    with pytest.raises(WorkerFatalProtocolError, match="unauthorized"):
        run_forever(
            _worker(tmp_path, protocol, fail_analysis),
            max_iterations=1,
            sleep=lambda _: pytest.fail("fatal errors must not sleep"),
        )

    assert protocol.results == []


def test_periodic_transient_heartbeat_loss_drives_polling_backoff(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = PeriodicHeartbeatFailureProtocol(
        [_claim(source)],
        WorkerTransientError("heartbeat offline"),
    )

    def finish_after_lease_loss(
        path, reader, profile, backend, artifact_root, context
    ):
        del reader, profile, context
        assert protocol.periodic_attempted.wait(1)
        return _analysis(Path(path), backend, Path(artifact_root))

    sleeps = []
    run_forever(
        _worker(tmp_path, protocol, finish_after_lease_loss),
        max_iterations=1,
        sleep=sleeps.append,
    )

    assert sleeps == [1.0]
    assert protocol.results == []
    assert protocol.failures == []


def test_periodic_fatal_heartbeat_loss_terminates_polling(tmp_path: Path) -> None:
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    protocol = PeriodicHeartbeatFailureProtocol(
        [_claim(source)],
        WorkerFatalProtocolError("heartbeat unauthorized"),
    )

    def finish_after_lease_loss(
        path, reader, profile, backend, artifact_root, context
    ):
        del reader, profile, context
        assert protocol.periodic_attempted.wait(1)
        return _analysis(Path(path), backend, Path(artifact_root))

    with pytest.raises(WorkerFatalProtocolError, match="unauthorized"):
        run_forever(
            _worker(tmp_path, protocol, finish_after_lease_loss),
            max_iterations=1,
            sleep=lambda _: pytest.fail("fatal errors must not sleep"),
        )

    assert protocol.results == []
    assert protocol.failures == []


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


def test_worker_claim_model_binding_round_trips(tmp_path: Path):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    bound = replace(
        _claim(source),
        model_version_id="model-7",
        model_version="7",
        model_artifact_path="C:/models/7/model.bundle",
        model_bundle_sha256="7" * 64,
        model_feature_names=("duration_s", "rms"),
    )

    assert WorkerClaim.from_dict(bound.to_dict()) == bound


def test_worker_claim_rejects_partial_model_binding(tmp_path: Path):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    payload = _claim(source).to_dict()
    payload["model_version_id"] = "model-7"

    with pytest.raises(ValueError, match="all model binding fields"):
        WorkerClaim.from_dict(payload)


@pytest.mark.parametrize(
    "raw_model",
    [b"corrupt", _unsupported_compression_bundle()],
    ids=("invalid-zip", "unsupported-compression"),
)
def test_bound_corrupt_model_fails_without_rules_fallback(
    tmp_path: Path,
    raw_model: bytes,
):
    source = tmp_path / "input.h5"
    source.write_bytes(b"stable")
    settings = _settings(tmp_path)
    model_path = (
        settings.app_data_dir
        / "artifacts"
        / "models"
        / "model-1.bundle"
    )
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(raw_model)
    claim = _claim(
        source,
        model_version_id="model-1",
        model_version="1",
        model_artifact_path=str(model_path),
        model_bundle_sha256=hashlib.sha256(raw_model).hexdigest(),
        model_feature_names=("duration_s", "rms"),
    )
    protocol = FakeProtocol([claim])

    def reject_rules_fallback(*args):
        del args
        raise RuntimeError("corrupt model used rules fallback")

    worker = _worker(
        tmp_path,
        protocol,
        reject_rules_fallback,
        settings=settings,
    )

    outcome = worker.run_once()

    assert outcome.error_code == "invalid_model"
    assert protocol.heartbeats
    assert protocol.failures[0][4] is False
    assert protocol.results == []


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


def test_bound_model_result_payload_uses_one_model_identity_and_fusion_method(
    tmp_path: Path,
):
    from das_sentinel.api.schemas import WorkerResultRequest
    from das_sentinel.worker.main import _result_payload

    source = tmp_path / "source.h5"
    source.write_bytes(b"payload-test")
    model_path = (
        tmp_path
        / "app-data"
        / "artifacts"
        / "models"
        / "site-1"
        / "7"
        / "model.bundle"
    )
    model_path.parent.mkdir(parents=True)
    claim = _claim(
        source,
        model_version_id="model-7",
        model_version="7",
        model_artifact_path=str(model_path),
        model_bundle_sha256="7" * 64,
        model_feature_names=("duration_s", "rms"),
    )
    analysis = _analysis(
        source,
        ArrayBackend("cpu", np),
        tmp_path / "artifacts",
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
    validated = WorkerResultRequest.model_validate(payload)

    assert validated.model_version_id == "model-7"
    assert validated.events[0].classification.method == "fusion"


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
