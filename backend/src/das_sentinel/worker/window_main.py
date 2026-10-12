"""Worker for persistent whole-window jobs; it never claims legacy file jobs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx

from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.services.discovery import _open_anchored_source
from das_sentinel.worker.main import (
    ResultSubmissionError,
    WorkerClaimLostError,
    WorkerFatalProtocolError,
    WorkerOutcome,
    WorkerTransientError,
)

logger = logging.getLogger(__name__)


class HttpxWindowClient:
    def __init__(self, api_url, worker_token, *, transport=None, timeout=60.0):
        self._client = httpx.Client(
            base_url=api_url.rstrip("/"),
            headers={"X-Worker-Token": worker_token},
            transport=transport,
            timeout=timeout,
        )

    def close(self):
        self._client.close()

    def _post(self, suffix, payload):
        try:
            response = self._client.post(
                f"/api/v1/internal/window-jobs/{suffix}", json=payload
            )
        except httpx.TransportError as error:
            raise WorkerTransientError("window API transport failure") from error
        code = response.status_code
        if code >= 500 or code == 429:
            raise WorkerTransientError(f"window API HTTP {code}")
        if code in (404, 409) and suffix != "claim":
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if detail in (
                "window job not found",
                "window job is not running",
                "worker is not the window job owner",
                "window job attempt changed",
                "job is not running",
                "worker is not the job owner",
                "window job claim lost",
                "claim_lost",
                "analysis_not_found",
            ):
                raise WorkerClaimLostError("window job claim lost")
            if (
                code == 409
                and suffix.endswith(("/result", "/skip"))
                and detail == "source_profile_changed"
            ):
                raise ResultSubmissionError("window source or profile changed")
            if code == 409 and suffix.endswith("/skip") and detail == "skip_not_allowed":
                raise ResultSubmissionError("window skip is no longer allowed")
        if code == 422 and suffix.endswith("/result"):
            raise ResultSubmissionError("window result rejected (HTTP 422)")
        if code >= 400:
            raise WorkerFatalProtocolError(f"window API HTTP {code}")
        if code == 204:
            return None
        try:
            return response.json()
        except ValueError as error:
            raise WorkerFatalProtocolError(
                "window API returned invalid JSON"
            ) from error

    def claim(self, worker_id):
        return self._post("claim", {"worker_id": worker_id, "supports_empty_skip": True})

    def heartbeat(self, job_id, worker_id, attempt_count):
        return self._post(
            f"{job_id}/heartbeat",
            {
                "worker_id": worker_id,
                "attempt_count": attempt_count,
            },
        )

    def result(self, job_id, worker_id, attempt_count):
        return self._post(
            f"{job_id}/result",
            {
                "worker_id": worker_id,
                "attempt_count": attempt_count,
            },
        )

    def skip(self, job_id, worker_id, attempt_count, plan_sha256):
        payload = {
            "worker_id": worker_id,
            "attempt_count": attempt_count,
            "plan_sha256": plan_sha256,
            "reason": "no_qualified_regions",
        }
        try:
            return self._post(f"{job_id}/skip", payload)
        except WorkerTransientError:
            # The first request may have committed; replay the same tombstone identity.
            return self._post(f"{job_id}/skip", payload)

    def fail(self, job_id, worker_id, attempt_count, error_code, error_message):
        return self._post(
            f"{job_id}/fail",
            {
                "worker_id": worker_id,
                "attempt_count": attempt_count,
                "error_code": error_code,
                "error_message": error_message,
            },
        )


def _analyze(*args, **kwargs):
    from das_sentinel.whole_window import analyze_window

    return analyze_window(*args, **kwargs)


class WindowWorker:
    def __init__(
        self, settings, client, worker_id, *, analyzer=None, heartbeat_seconds=15
    ):
        if not worker_id.strip() or len(worker_id) > 200:
            raise ValueError("worker_id must be nonblank and at most 200 characters")
        if not 0 < heartbeat_seconds <= 60:
            raise ValueError("heartbeat interval must be in (0, 60]")
        self.settings = settings
        self.client = client
        self.worker_id = worker_id
        self.analyzer = analyzer or _analyze
        self.heartbeat_seconds = heartbeat_seconds

    @staticmethod
    def _validate(claim):
        try:
            for field in ("id", "window_id", "site_id"):
                if str(UUID(claim[field])) != claim[field]:
                    raise ValueError("noncanonical identifier")
            attempt = claim["attempt_count"]
            if type(attempt) is not int or attempt < 1:
                raise ValueError("invalid attempt")
            start, end = (
                datetime.fromisoformat(claim[key]) for key in ("start_utc", "end_utc")
            )
            if start.utcoffset() is None or end.utcoffset() is None:
                raise ValueError("missing timezone")
            start = start.astimezone(UTC)
            duration = (end - start).total_seconds()
            if (
                duration not in (30, 60)
                or start.second % duration
                or start.microsecond
            ):
                raise ValueError("invalid primary window")
            skip_empty = claim.get("skip_empty", False)
            if type(skip_empty) is not bool:
                raise ValueError("invalid empty skip authorization")
            if skip_empty and (
                duration != 30
                or not isinstance(claim.get("plan_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", claim["plan_sha256"]) is None
                or not isinstance(claim.get("skip_policy_version"), str)
                or not claim["skip_policy_version"].strip()
            ):
                raise ValueError("invalid empty skip authorization")
            if (
                not isinstance(claim["sources"], list)
                or not 1 <= len(claim["sources"]) <= 1000
            ):
                raise ValueError("invalid source list")
            profile = ProcessingProfile.model_validate_json(
                json.dumps(claim["profile_payload"], allow_nan=False),
                strict=True,
            )
            root = Path(claim["site_data_root"])
            if not root.is_absolute() or ".." in root.parts:
                raise ValueError("invalid source root")
            for source in claim["sources"]:
                if not isinstance(source, dict) or not isinstance(source["path"], str):
                    raise TypeError("invalid source snapshot")
                path = Path(source["path"])
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or path == root
                    or not path.is_relative_to(root)
                ):
                    raise ValueError("invalid source path")
                if (
                    not isinstance(source["fingerprint"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", source["fingerprint"]) is None
                    or type(source["size_bytes"]) is not int
                    or source["size_bytes"] <= 0
                    or type(source["mtime_ns"]) is not int
                    or source["mtime_ns"] < 0
                ):
                    raise ValueError("invalid source snapshot")
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise WorkerFatalProtocolError("invalid whole-window claim") from error
        return start, profile, root

    def run_once(self):
        claim = self.client.claim(self.worker_id)
        if claim is None:
            return WorkerOutcome(status="idle")
        start, profile, root = self._validate(claim)
        job_id, attempt = claim["id"], claim["attempt_count"]
        stop = threading.Event()
        heartbeat_error = []
        thread = None

        def heartbeat():
            while not stop.wait(self.heartbeat_seconds):
                try:
                    self.client.heartbeat(job_id, self.worker_id, attempt)
                except Exception as error:  # noqa: BLE001 - propagate heartbeat failures to the owner
                    heartbeat_error.append(error)
                    return

        try:
            self.client.heartbeat(job_id, self.worker_id, attempt)
            thread = threading.Thread(
                target=heartbeat, daemon=True, name=f"window-heartbeat-{job_id}"
            )
            thread.start()
            if not root.resolve().is_relative_to(self.settings.data_root.resolve()):
                raise WorkerFatalProtocolError("claimed site root is outside data root")
            paths = []
            with ExitStack() as stack:
                for source in claim["sources"]:
                    path = Path(source["path"])
                    anchored = stack.enter_context(
                        _open_anchored_source(
                            root,
                            path,
                            required_suffix=path.suffix,
                        )
                    )
                    snapshot = anchored.snapshot().discovered()
                    if any(
                        getattr(snapshot, key) != source[key]
                        for key in (
                            "fingerprint",
                            "size_bytes",
                            "mtime_ns",
                        )
                    ):
                        raise ValueError("source changed since window planning")
                    paths.append(path)
                output = (
                    self.settings.app_data_dir
                    / "window-artifacts"
                    / job_id
                    / f"attempt-{attempt}"
                )
                options = {"skip_empty": True} if claim.get("skip_empty") else {}
                analysis = self.analyzer(
                    paths, start, profile, output,
                    site_timezone=claim.get("site_timezone", "UTC"),
                    window_seconds=int((datetime.fromisoformat(claim["end_utc"]) - start).total_seconds()),
                    **options,
                )
            stop.set()
            thread.join()
            if heartbeat_error:
                raise heartbeat_error[0]
            # Recheck the lease after long I/O before publishing any result.
            self.client.heartbeat(job_id, self.worker_id, attempt)
            if claim.get("skip_empty") and isinstance(analysis, dict) and analysis.get("skipped") is True:
                self.client.skip(job_id, self.worker_id, attempt, claim["plan_sha256"])
                return WorkerOutcome(status="skipped", job_id=job_id)
            self.client.result(job_id, self.worker_id, attempt)
            return WorkerOutcome(status="succeeded", job_id=job_id)
        except WorkerClaimLostError:
            return WorkerOutcome(status="claim_lost", job_id=job_id)
        except (WorkerFatalProtocolError, WorkerTransientError):
            raise
        except Exception as error:  # noqa: BLE001 - isolate one window's analysis failure
            stop.set()
            if thread is not None:
                thread.join()
            if heartbeat_error and isinstance(
                heartbeat_error[0],
                (
                    WorkerFatalProtocolError,
                    WorkerTransientError,
                    WorkerClaimLostError,
                ),
            ):
                if isinstance(heartbeat_error[0], WorkerClaimLostError):
                    return WorkerOutcome(status="claim_lost", job_id=job_id)
                raise heartbeat_error[0]
            code = (
                "result_rejected"
                if isinstance(error, ResultSubmissionError)
                else "window_analysis_failed"
            )
            message = str(error) or type(error).__name__
            for secret in (
                self.settings.worker_token,
                *(str(source.get("path", "")) for source in claim["sources"]),
            ):
                if secret:
                    message = message.replace(secret, "[redacted]")
            try:
                self.client.fail(job_id, self.worker_id, attempt, code, message[:1024])
            except WorkerClaimLostError:
                return WorkerOutcome(status="claim_lost", job_id=job_id)
            return WorkerOutcome(status="failed", job_id=job_id, error_code=code)
        finally:
            stop.set()
            if thread is not None:
                thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", default=f"window-{os.getpid()}")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    settings = Settings()
    client = HttpxWindowClient(settings.api_url, settings.worker_token)
    worker = WindowWorker(settings, client, args.worker_id)
    processed = 0
    try:
        while True:
            try:
                result = worker.run_once()
            except WorkerTransientError:
                logger.warning("Window API temporarily unavailable; backing off")
                if args.once:
                    raise
                time.sleep(5)
                continue
            if result.status != "idle":
                processed += 1
                logger.info("Window %s %s", result.job_id, result.status)
            if args.once or (args.max_jobs > 0 and processed >= args.max_jobs):
                break
            if result.status == "idle":
                time.sleep(3)
    finally:
        client.close()


if __name__ == "__main__":
    main()
