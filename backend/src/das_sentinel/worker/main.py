from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.pipeline import (
    AnalysisContext,
    BoundSource,
    FileAnalysis,
    SourceChangedError,
    analyze_file,
)
from das_sentinel.readers.base import ReaderRegistry
from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import _open_anchored_source
from das_sentinel.signal.backend import (
    ArrayBackend,
    BackendUnavailable,
    select_backend,
)

MAX_FAILURE_MESSAGE = 1024


class WorkerTransientError(RuntimeError):
    """A worker API failure that is safe to retry after backoff."""


class WorkerFatalProtocolError(RuntimeError):
    """A worker API response that must stop the polling process."""


class CudaOutOfMemoryError(RuntimeError):
    """Explicit adapter exception for an exact CUDA allocation failure."""


class ResultSubmissionError(RuntimeError):
    """The API rejected or could not persist an otherwise complete result."""


@dataclass(frozen=True)
class WorkerClaim:
    id: str
    source_file_id: str
    source_path: str
    source_fingerprint: str
    source_size_bytes: int
    source_mtime_ns: int
    site_id: str
    site_data_root: str
    site_timezone: str
    cable_type: str
    processing_profile_id: str
    profile_version: int
    profile_payload: dict[str, Any]
    pipeline_version: str
    attempt_count: int
    worker_id: str
    heartbeat_at: datetime = datetime(2026, 1, 1, tzinfo=UTC)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerClaim:
        from das_sentinel.api.schemas import JobClaimResponse

        parsed = JobClaimResponse.model_validate(value)
        return cls(
            id=parsed.id,
            source_file_id=parsed.source_file_id,
            source_path=parsed.source_path,
            source_fingerprint=parsed.source_fingerprint,
            source_size_bytes=parsed.source_size_bytes,
            source_mtime_ns=parsed.source_mtime_ns,
            site_id=parsed.site_id,
            site_data_root=parsed.site_data_root,
            site_timezone=parsed.site_timezone,
            cable_type=parsed.cable_type,
            processing_profile_id=parsed.processing_profile_id,
            profile_version=parsed.profile_version,
            profile_payload=parsed.profile_payload,
            pipeline_version=parsed.pipeline_version,
            attempt_count=parsed.attempt_count,
            worker_id=parsed.worker_id,
            heartbeat_at=parsed.heartbeat_at,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["heartbeat_at"] = self.heartbeat_at.isoformat()
        return result


@dataclass(frozen=True)
class WorkerOutcome:
    status: str
    job_id: str | None = None
    requested_backend: str | None = None
    actual_backend: str | None = None
    backend_fallback_reason: str | None = None
    error_code: str | None = None


class WorkerProtocol(Protocol):
    def claim(self, worker_id: str) -> WorkerClaim | None: ...

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        numerator: int,
        denominator: int,
    ) -> None: ...

    def submit_result(self, job_id: str, payload: dict[str, Any]) -> None: ...

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry_eligible: bool,
    ) -> None: ...


class HttpxWorkerClient:
    """Synchronous internal API client with one secret header and no cookies."""

    def __init__(
        self,
        api_url: str,
        worker_token: str,
        *,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=api_url.rstrip("/"),
            headers={"X-Worker-Token": worker_token},
            timeout=timeout,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=json)
        except httpx.TransportError as error:
            raise WorkerTransientError("worker API transport failure") from error
        if response.status_code >= 500:
            raise WorkerTransientError(
                f"worker API returned HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            raise WorkerFatalProtocolError(
                f"worker API returned HTTP {response.status_code}"
            )
        response.raise_for_status()
        return response

    def claim(self, worker_id: str) -> WorkerClaim | None:
        response = self._request(
            "POST",
            "/api/v1/internal/jobs/claim",
            json={"worker_id": worker_id},
        )
        if response.status_code == 204:
            return None
        return WorkerClaim.from_dict(response.json())

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        numerator: int,
        denominator: int,
    ) -> None:
        self._request(
            "POST",
            f"/api/v1/internal/jobs/{job_id}/heartbeat",
            json={
                "worker_id": worker_id,
                "numerator": numerator,
                "denominator": denominator,
            },
        )

    def submit_result(self, job_id: str, payload: dict[str, Any]) -> None:
        self._request(
            "POST",
            f"/api/v1/internal/jobs/{job_id}/result",
            json=payload,
        )

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry_eligible: bool,
    ) -> None:
        self._request(
            "POST",
            f"/api/v1/internal/jobs/{job_id}/fail",
            json={
                "worker_id": worker_id,
                "error_code": error_code,
                "error_message": error_message,
                "retry_eligible": retry_eligible,
            },
        )


class Worker:
    def __init__(
        self,
        *,
        settings: Settings,
        client: WorkerProtocol,
        worker_id: str,
        readers: ReaderRegistry | None = None,
        analyzer: Callable[..., FileAnalysis] = analyze_file,
        backend_selector: Callable[[str], ArrayBackend] | None = None,
        heartbeat_interval_seconds: float = 15.0,
    ) -> None:
        if not worker_id.strip() or len(worker_id) > 200:
            raise ValueError("worker_id must be nonblank and at most 200 characters")
        if (
            isinstance(heartbeat_interval_seconds, bool)
            or not math.isfinite(heartbeat_interval_seconds)
            or heartbeat_interval_seconds <= 0
        ):
            raise ValueError("heartbeat interval must be positive and finite")
        self.settings = settings
        self.client = client
        self.worker_id = worker_id
        self.readers = readers or ReaderRegistry([ZdHdf5Reader()])
        self.analyzer = analyzer
        self.backend_selector = backend_selector or select_backend
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop: threading.Event | None = None

    def run_once(self) -> WorkerOutcome:
        claim = self.client.claim(self.worker_id)
        if claim is None:
            return WorkerOutcome(status="idle")

        requested_backend: str | None = None
        try:
            with self._validate_claim(claim) as (source, profile, reader):
                requested_backend = (
                    self.settings.worker_device
                    if self.settings.worker_device != "auto"
                    else profile.device
                )
                metadata = reader.read_metadata(source.stream)
                self._start_heartbeat(claim)
                try:
                    analysis, actual_backend, fallback_reason = self._analyze(
                        claim,
                        source,
                        profile,
                        reader,
                        requested_backend,
                    )
                    if analysis.source_fingerprint != claim.source_fingerprint:
                        raise SourceChangedError(
                            "analysis source fingerprint does not match claim"
                        )
                    if analysis.actual_backend != actual_backend:
                        raise RuntimeError(
                            "analysis backend identity does not match selected backend"
                        )
                    payload = _result_payload(
                        claim,
                        metadata.start_time_utc,
                        metadata.sample_rate_hz,
                        analysis,
                        requested_backend=requested_backend,
                        actual_backend=actual_backend,
                        fallback_reason=fallback_reason,
                    )
                    try:
                        self.client.submit_result(claim.id, payload)
                    except (WorkerTransientError, WorkerFatalProtocolError):
                        raise
                    except Exception as error:
                        raise ResultSubmissionError(
                            f"result submission failed: {error}"
                        ) from error
                    return WorkerOutcome(
                        status="succeeded",
                        job_id=claim.id,
                        requested_backend=requested_backend,
                        actual_backend=actual_backend,
                        backend_fallback_reason=fallback_reason,
                    )
                finally:
                    self._stop_heartbeat()
        except (WorkerTransientError, WorkerFatalProtocolError):
            self._stop_heartbeat()
            raise
        except Exception as error:  # noqa: BLE001 - isolate each claimed file
            self._stop_heartbeat()
            code = _failure_code(error)
            message = self._safe_error_message(error, claim)
            self._report_failure(
                claim.id,
                code,
                message,
                _failure_retry_eligible(error),
            )
            return WorkerOutcome(
                status="failed",
                job_id=claim.id,
                requested_backend=requested_backend,
                error_code=code,
            )

    @contextmanager
    def _validate_claim(self, claim: WorkerClaim):
        if claim.worker_id != self.worker_id:
            raise ValueError("claim worker does not match this worker")
        if claim.profile_version < 1 or claim.attempt_count < 1:
            raise ValueError("claim versions and attempts must be positive")
        if len(claim.source_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in claim.source_fingerprint
        ):
            raise ValueError("claim source fingerprint is invalid")
        unknown_profile_fields = (
            set(claim.profile_payload) - set(ProcessingProfile.model_fields)
        )
        if unknown_profile_fields:
            raise ValueError("claim profile contains unsupported fields")
        profile = ProcessingProfile.model_validate_json(
            json.dumps(claim.profile_payload, allow_nan=False),
            strict=True,
        )
        source = Path(claim.source_path)
        if not source.is_absolute():
            raise ValueError("claim source path must be absolute")
        root = Path(claim.site_data_root)
        with _open_anchored_source(root, source) as anchored:
            discovered = anchored.snapshot().discovered()
            if (
                discovered.size_bytes != claim.source_size_bytes
                or discovered.mtime_ns != claim.source_mtime_ns
                or discovered.fingerprint != claim.source_fingerprint
            ):
                raise ValueError("claim source identity changed")
            bound_source = BoundSource(
                path=source,
                stream=anchored.stream,
                fingerprint=discovered.fingerprint,
                size_bytes=discovered.size_bytes,
                mtime_ns=discovered.mtime_ns,
            )
            yield (
                bound_source,
                profile,
                self.readers.resolve(bound_source.stream),
            )

    def _analyze(
        self,
        claim: WorkerClaim,
        source: BoundSource,
        profile: ProcessingProfile,
        reader,
        requested_backend: str,
    ) -> tuple[FileAnalysis, str, str | None]:
        fallback_reason: str | None = None
        try:
            backend = self.backend_selector(requested_backend)
        except BackendUnavailable:
            if requested_backend != "cuda":
                raise
            backend = self.backend_selector("cpu")
            fallback_reason = "cuda_unavailable"
        if (
            backend.name == "cpu"
            and requested_backend in {"auto", "cuda"}
            and backend.unavailable_reason
        ):
            fallback_reason = "cuda_unavailable"

        context = AnalysisContext(
            site_id=claim.site_id,
            site_timezone=claim.site_timezone,
            profile_version=str(claim.profile_version),
            pipeline_version=claim.pipeline_version,
            cable_type=claim.cable_type,
            device_config=requested_backend,
        )
        try:
            result = self.analyzer(
                source,
                reader,
                profile,
                backend,
                self.settings.app_data_dir / "artifacts",
                context,
            )
            return result, backend.name, fallback_reason
        except Exception as error:
            if backend.name != "cuda" or not _is_cuda_out_of_memory(error):
                raise
            backend.clear_cache()
            cpu = self.backend_selector("cpu")
            result = self.analyzer(
                source,
                reader,
                profile,
                cpu,
                self.settings.app_data_dir / "artifacts",
                context,
            )
            return result, "cpu", "cuda_out_of_memory"

    def _start_heartbeat(self, claim: WorkerClaim) -> None:
        self.client.heartbeat(claim.id, self.worker_id, 0, 1)
        stop = threading.Event()

        def heartbeat_loop() -> None:
            while not stop.wait(self.heartbeat_interval_seconds):
                try:
                    self.client.heartbeat(claim.id, self.worker_id, 0, 1)
                except Exception:  # noqa: BLE001 - heartbeat failure stops this helper
                    return

        thread = threading.Thread(
            target=heartbeat_loop,
            name=f"das-heartbeat-{claim.id}",
            daemon=True,
        )
        self._heartbeat_stop = stop
        self.heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(self) -> None:
        stop = self._heartbeat_stop
        thread = self.heartbeat_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self._heartbeat_stop = None
        self.heartbeat_thread = None

    def _safe_error_message(
        self,
        error: BaseException,
        claim: WorkerClaim,
    ) -> str:
        message = str(error).strip() or type(error).__name__
        for secret in (
            self.settings.worker_token,
            claim.source_path,
        ):
            if secret:
                message = message.replace(secret, "[redacted]")
        return message[:MAX_FAILURE_MESSAGE]

    def _report_failure(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
        retry_eligible: bool,
    ) -> bool:
        try:
            self.client.fail(
                job_id,
                self.worker_id,
                error_code,
                error_message,
                retry_eligible,
            )
        except Exception:  # noqa: BLE001 - failure reporting must not kill polling
            return False
        return True


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, CudaOutOfMemoryError):
        return True
    error_type = type(error)
    return (
        error_type.__name__ == "OutOfMemoryError"
        and error_type.__module__.startswith("cupy.")
    )


def _failure_code(error: BaseException) -> str:
    if isinstance(error, ResultSubmissionError):
        return "persistence_failed"
    if isinstance(error, SourceChangedError):
        return "source_changed"
    if isinstance(error, (ValueError, TypeError)):
        return "invalid_claim"
    return "analysis_failed"


def _failure_retry_eligible(error: BaseException) -> bool:
    return not isinstance(error, (SourceChangedError, ValueError, TypeError))


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars/containers into JSON-serializable natives."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _result_payload(
    claim: WorkerClaim,
    source_start_time: datetime,
    sample_rate_hz: float,
    analysis: FileAnalysis,
    *,
    requested_backend: str,
    actual_backend: str,
    fallback_reason: str | None,
) -> dict[str, Any]:
    artifacts = [
        {
            "event_uid": manifest.event_id,
            "kind": artifact.kind,
            "path": str(artifact.path),
            "media_type": artifact.media_type,
            "sha256": artifact.sha256,
        }
        for manifest in analysis.artifacts
        for artifact in manifest.files.values()
    ]
    events = []
    for event in analysis.events:
        events.append(
            {
                "event_uid": event.event_id,
                "start_sample": event.start_sample,
                "end_sample": event.end_sample,
                "channel_start": event.channel_start,
                "channel_stop": event.channel_stop,
                "features": event.features,
                "evidence": [
                    {
                        "detector": item.detector,
                        "score": item.score,
                        "payload": {
                            "start_sample": item.start_sample,
                            "end_sample": item.end_sample,
                            "peak_sample": item.peak_sample,
                            "channel_start": item.channel_start,
                            "channel_stop": item.channel_stop,
                            "metrics": item.metrics,
                            "quality_flags": list(item.quality_flags),
                        },
                    }
                    for item in event.evidence
                ],
                "classification": {
                    "label": event.label.value,
                    "decision_score": event.decision_score,
                    "method": "rules",
                    "reasons": list(event.reasons),
                },
            }
        )
    return {
        "worker_id": claim.worker_id,
        "source_file_id": claim.source_file_id,
        "processing_profile_id": claim.processing_profile_id,
        "profile_version": claim.profile_version,
        "pipeline_version": claim.pipeline_version,
        "source_fingerprint": claim.source_fingerprint,
        "source_start_time_utc": source_start_time.astimezone(UTC).isoformat(),
        "sample_rate_hz": sample_rate_hz,
        "requested_backend": requested_backend,
        "actual_backend": actual_backend,
        "backend_fallback_reason": fallback_reason,
        "timings_ms": _json_safe(analysis.timings_ms),
        "events": _json_safe(events),
        "artifacts": _json_safe(artifacts),
    }
