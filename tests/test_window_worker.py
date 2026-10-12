from __future__ import annotations

import importlib
import importlib.util
import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from das_sentinel.config import Settings
from das_sentinel.services.discovery import DiscoveredFile
from das_sentinel.worker.main import (
    ResultSubmissionError,
    WorkerClaimLostError,
    WorkerFatalProtocolError,
    WorkerTransientError,
)


def module():
    assert importlib.util.find_spec("das_sentinel.worker.window_main") is not None
    return importlib.import_module("das_sentinel.worker.window_main")


def environment(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "a.h5"
    source.write_bytes(b"stable source fixture")
    snapshot = DiscoveredFile.from_path(source)
    settings = Settings(
        data_root=raw,
        app_data_dir=tmp_path / "app-data",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'test.db'}",
        session_secret="s" * 40,
        worker_token="w" * 40,
        secure_cookies=False,
    )
    claim = {
        "id": str(uuid4()),
        "window_id": str(uuid4()),
        "site_id": str(uuid4()),
        "site_data_root": str(raw),
        "site_timezone": "Asia/Shanghai",
        "start_utc": datetime(2023, 9, 27, 3, 2, tzinfo=UTC).isoformat(),
        "end_utc": datetime(2023, 9, 27, 3, 3, tzinfo=UTC).isoformat(),
        "profile_payload": {},
        "attempt_count": 1,
        "sources": [
            {
                "path": str(source),
                "fingerprint": snapshot.fingerprint,
                "size_bytes": snapshot.size_bytes,
                "mtime_ns": snapshot.mtime_ns,
            }
        ],
    }
    return settings, claim


@pytest.mark.parametrize("seconds", [30, 60])
def test_window_worker_continues_after_one_result_rejection(tmp_path, seconds):
    implementation = module()
    settings, claim = environment(tmp_path)
    start = datetime.fromisoformat(claim["start_utc"]) + timedelta(seconds=30 if seconds == 30 else 0)
    claim.update(start_utc=start.isoformat(), end_utc=(start + timedelta(seconds=seconds)).isoformat())
    second = {**claim, "id": str(uuid4()), "attempt_count": 2}
    queued = [claim, second]
    failed = []
    submitted = []
    output_dirs = []

    def respond(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/claim"):
            return httpx.Response(200, json=queued.pop(0))
        assert body["attempt_count"] in (1, 2)
        if request.url.path.endswith("/result"):
            submitted.append(request.url.path)
            if len(submitted) == 1:
                return httpx.Response(422, json={"detail": "result exceeds size limit"})
        if request.url.path.endswith("/fail"):
            failed.append(body)
        return httpx.Response(200, json={"status": "ok"})

    def analyze(paths, start_utc, profile, output_dir, *, site_timezone, window_seconds):
        assert paths == [Path(claim["sources"][0]["path"])]
        assert site_timezone == "Asia/Shanghai"
        assert window_seconds == seconds
        assert start_utc == start
        output_dirs.append(output_dir)
        return {}

    client = implementation.HttpxWindowClient(
        "http://api",
        settings.worker_token,
        transport=httpx.MockTransport(respond),
    )
    try:
        worker = implementation.WindowWorker(
            settings, client, "worker-a", analyzer=analyze
        )
        assert [worker.run_once().status, worker.run_once().status] == [
            "failed",
            "succeeded",
        ]
        assert len(failed) == 1
        assert failed[0]["attempt_count"] == 1
        assert output_dirs == [
            settings.app_data_dir / "window-artifacts" / claim["id"] / "attempt-1",
            settings.app_data_dir / "window-artifacts" / second["id"] / "attempt-2",
        ]
    finally:
        client.close()


def test_changed_source_fails_before_analyzer(tmp_path):
    implementation = module()
    settings, claim = environment(tmp_path)
    Path(claim["sources"][0]["path"]).write_bytes(b"changed")
    failures = []

    class Client:
        def claim(self, worker_id):
            return claim

        def heartbeat(self, *args):
            pass

        def fail(self, *args):
            failures.append(args)

    worker = implementation.WindowWorker(
        settings,
        Client(),
        "worker-a",
        analyzer=lambda *args: pytest.fail("changed source was analyzed"),
    )
    assert worker.run_once().status == "failed"
    assert len(failures) == 1


@pytest.mark.parametrize("status", [401, 403])
def test_window_worker_authorization_error_is_fatal(status):
    implementation = module()
    client = implementation.HttpxWindowClient(
        "http://api",
        "w" * 40,
        transport=httpx.MockTransport(lambda request: httpx.Response(status)),
    )
    try:
        with pytest.raises(WorkerFatalProtocolError):
            client.claim("worker")
    finally:
        client.close()


@pytest.mark.parametrize(
    ("suffix", "detail", "expected"),
    [
        ("result", "source_profile_changed", ResultSubmissionError),
        ("result", "claim_lost", WorkerClaimLostError),
        ("result", "result_identity_mismatch", WorkerFatalProtocolError),
        ("result", "input_signature_mismatch", WorkerFatalProtocolError),
        ("result", "unknown_conflict", WorkerFatalProtocolError),
        ("heartbeat", "source_profile_changed", WorkerFatalProtocolError),
    ],
)
def test_window_result_only_isolates_known_stale_input(suffix, detail, expected):
    implementation = module()
    client = implementation.HttpxWindowClient(
        "http://api", "w" * 40,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(409, json={"detail": detail})
        ),
    )
    try:
        with pytest.raises(expected):
            getattr(client, suffix)(str(uuid4()), "worker", 1)
    finally:
        client.close()


def test_cancelled_claim_does_not_submit_a_result(tmp_path):
    implementation = module()
    settings, claim = environment(tmp_path)
    class Client:
        def claim(self, worker_id):
            return claim

        def heartbeat(self, *args):
            raise WorkerClaimLostError("cancelled")

        def result(self, *args):
            pytest.fail("cancelled claim submitted")

    worker = implementation.WindowWorker(
        settings,
        Client(),
        "worker-a",
        analyzer=lambda *args: pytest.fail("cancelled"),
    )
    assert worker.run_once().status == "claim_lost"


def test_invalid_claim_identifier_is_fatal(tmp_path):
    implementation = module()
    settings, claim = environment(tmp_path)
    claim["id"] = "../escape"

    class Client:
        def claim(self, worker_id):
            return claim

    with pytest.raises(WorkerFatalProtocolError):
        implementation.WindowWorker(settings, Client(), "worker-a").run_once()


@pytest.mark.parametrize(
    "invalid",
    [
        "not_mapping", "missing_path", "relative", "outside", "traversal",
        "missing_snapshot", "bad_fingerprint", "string_size", "bool_mtime",
    ],
)
def test_invalid_source_claim_is_fatal_before_heartbeat(tmp_path, invalid):
    implementation = module()
    settings, claim = environment(tmp_path)
    source = claim["sources"][0]
    if invalid == "not_mapping":
        claim["sources"] = [None]
    elif invalid == "missing_path":
        del source["path"]
    elif invalid == "relative":
        source["path"] = "a.h5"
    elif invalid == "outside":
        source["path"] = str(tmp_path / "outside.h5")
    elif invalid == "traversal":
        source["path"] = str(settings.data_root / ".." / "raw" / "a.h5")
    elif invalid == "missing_snapshot":
        del source["mtime_ns"]
    elif invalid == "bad_fingerprint":
        source["fingerprint"] = "not-a-sha256"
    elif invalid == "string_size":
        source["size_bytes"] = str(source["size_bytes"])
    elif invalid == "bool_mtime":
        source["mtime_ns"] = True

    calls = []

    class Client:
        def claim(self, worker_id):
            return claim

        def heartbeat(self, *args):
            calls.append("heartbeat")

        def fail(self, *args):
            calls.append("fail")

    with pytest.raises(WorkerFatalProtocolError):
        implementation.WindowWorker(settings, Client(), "worker-a").run_once()
    assert calls == []


@pytest.mark.parametrize(
    "heartbeat_failure",
    [WorkerFatalProtocolError, WorkerTransientError, WorkerClaimLostError],
)
def test_analysis_failure_waits_for_inflight_heartbeat(
    tmp_path, monkeypatch, heartbeat_failure
):
    implementation = module()
    settings, claim = environment(tmp_path)
    started = threading.Event()
    release = threading.Event()
    original_thread = threading.Thread

    class Thread(original_thread):
        def join(self, *args, **kwargs):
            # Complete the in-flight response only once the owner waits for it.
            release.set()
            return super().join(*args, **kwargs)

    monkeypatch.setattr(implementation.threading, "Thread", Thread)
    failed = []
    heartbeat_count = 0

    class Client:
        def claim(self, worker_id):
            return claim

        def heartbeat(self, *args):
            nonlocal heartbeat_count
            heartbeat_count += 1
            if heartbeat_count == 1:
                return
            started.set()
            assert release.wait(5)
            raise heartbeat_failure("in-flight heartbeat rejected")

        def fail(self, *args):
            failed.append(args)

    def analyze(*args, **kwargs):
        assert started.wait(5)
        raise ValueError("local analysis failed")

    worker = implementation.WindowWorker(
        settings, Client(), "worker-a", analyzer=analyze, heartbeat_seconds=0.01
    )
    if heartbeat_failure is WorkerClaimLostError:
        assert worker.run_once().status == "claim_lost"
    else:
        with pytest.raises(heartbeat_failure):
            worker.run_once()
    assert not failed


def empty_skip_claim(tmp_path):
    settings, claim = environment(tmp_path)
    start = datetime.fromisoformat(claim["start_utc"])
    claim.update(
        end_utc=(start + timedelta(seconds=30)).isoformat(),
        skip_empty=True,
        plan_sha256="a" * 64,
        skip_policy_version="empty-regions-v1",
    )
    return settings, claim


@pytest.mark.parametrize("empty", [False, True])
def test_authorized_worker_submits_skip_only_for_empty_analysis(tmp_path, empty):
    implementation = module()
    settings, claim = empty_skip_claim(tmp_path)
    requests = []

    def respond(request):
        operation = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content)
        requests.append((operation, body))
        if operation == "claim":
            return httpx.Response(200, json=claim)
        return httpx.Response(200, json={"id": claim["id"], "status": "skipped"})

    def analyze(paths, start, profile, output, *, site_timezone, window_seconds, skip_empty):
        assert skip_empty is True and window_seconds == 30
        return {"skipped": True, "reason": "no_qualified_regions"} if empty else {}

    client = implementation.HttpxWindowClient(
        "http://api", settings.worker_token, transport=httpx.MockTransport(respond)
    )
    try:
        outcome = implementation.WindowWorker(
            settings, client, "worker-a", analyzer=analyze
        ).run_once()
        assert outcome.status == ("skipped" if empty else "succeeded")
        assert requests[0] == ("claim", {"worker_id": "worker-a", "supports_empty_skip": True})
        assert [operation for operation, _ in requests] == [
            "claim", "heartbeat", "heartbeat", "skip" if empty else "result"
        ]
        if empty:
            assert requests[-1][1] == {
                "worker_id": "worker-a", "attempt_count": 1,
                "plan_sha256": "a" * 64, "reason": "no_qualified_regions",
            }
        assert not (settings.app_data_dir / "window-artifacts").exists()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("source_profile_changed", ResultSubmissionError),
        ("skip_not_allowed", ResultSubmissionError),
        ("claim_lost", WorkerClaimLostError),
        ("unknown_conflict", WorkerFatalProtocolError),
    ],
)
def test_skip_conflicts_preserve_worker_error_categories(detail, expected):
    implementation = module()
    client = implementation.HttpxWindowClient(
        "http://api", "w" * 40,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(409, json={"detail": detail})
        ),
    )
    try:
        with pytest.raises(expected):
            client.skip(str(uuid4()), "worker", 1, "a" * 64)
    finally:
        client.close()


def test_final_heartbeat_loss_prevents_skip_submission(tmp_path):
    implementation = module()
    settings, claim = empty_skip_claim(tmp_path)
    operations = []

    def respond(request):
        operation = request.url.path.rsplit("/", 1)[-1]
        operations.append(operation)
        if operation == "claim":
            return httpx.Response(200, json=claim)
        if operations.count("heartbeat") == 2:
            return httpx.Response(409, json={"detail": "claim_lost"})
        return httpx.Response(200, json={"status": "ok"})

    client = implementation.HttpxWindowClient(
        "http://api", settings.worker_token, transport=httpx.MockTransport(respond)
    )
    try:
        worker = implementation.WindowWorker(
            settings, client, "worker-a",
            analyzer=lambda *args, **kwargs: {"skipped": True, "reason": "no_qualified_regions"},
        )
        assert worker.run_once().status == "claim_lost"
        assert operations == ["claim", "heartbeat", "heartbeat"]
    finally:
        client.close()


def test_skip_retries_identical_payload_after_lost_response():
    implementation = module()
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            raise httpx.ReadTimeout("skip committed but response lost", request=request)
        return httpx.Response(200, json={"status": "skipped"})

    client = implementation.HttpxWindowClient(
        "http://api", "w" * 40, transport=httpx.MockTransport(respond)
    )
    try:
        assert client.skip(str(uuid4()), "worker", 1, "a" * 64) == {"status": "skipped"}
        assert payloads == [
            {"worker_id": "worker", "attempt_count": 1,
             "plan_sha256": "a" * 64, "reason": "no_qualified_regions"}
        ] * 2
    finally:
        client.close()


@pytest.mark.parametrize("invalid", ["duration", "flag", "hash", "policy"])
def test_invalid_skip_authorization_is_fatal_before_heartbeat(tmp_path, invalid):
    implementation = module()
    settings, claim = empty_skip_claim(tmp_path)
    if invalid == "duration":
        start = datetime.fromisoformat(claim["start_utc"])
        claim["end_utc"] = (start + timedelta(seconds=60)).isoformat()
    elif invalid == "flag":
        claim["skip_empty"] = "true"
    elif invalid == "hash":
        claim["plan_sha256"] = "invalid"
    else:
        claim["skip_policy_version"] = None

    class Client:
        def claim(self, worker_id):
            return claim

        def heartbeat(self, *args):
            pytest.fail("invalid authorization reached heartbeat")

    with pytest.raises(WorkerFatalProtocolError):
        implementation.WindowWorker(settings, Client(), "worker-a").run_once()


@pytest.mark.parametrize("detail", ["skip_not_allowed", "source_profile_changed"])
def test_skip_rejection_isolated_and_protected_claim_runs_normally(tmp_path, detail):
    implementation = module()
    settings, claim = empty_skip_claim(tmp_path)
    protected = {**claim, "id": str(uuid4()), "skip_empty": False, "skip_policy_version": None}
    claims = [claim, protected]
    operations = []
    failures = []

    def respond(request):
        operation = request.url.path.rsplit("/", 1)[-1]
        operations.append(operation)
        if operation == "claim":
            return httpx.Response(200, json=claims.pop(0))
        if operation == "skip":
            return httpx.Response(409, json={"detail": detail})
        if operation == "fail":
            failures.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "ok"})

    def analyze(*args, **kwargs):
        assert kwargs["skip_empty"] is True
        return {"skipped": True, "reason": "no_qualified_regions"}

    def legacy_analyze(paths, start, profile, output, *, site_timezone, window_seconds):
        return {}

    client = implementation.HttpxWindowClient(
        "http://api", settings.worker_token, transport=httpx.MockTransport(respond)
    )
    try:
        worker = implementation.WindowWorker(settings, client, "worker-a", analyzer=analyze)
        outcome = worker.run_once()
        assert outcome.status == "failed" and outcome.error_code == "result_rejected"
        worker.analyzer = legacy_analyze
        assert worker.run_once().status == "succeeded"
        assert len(failures) == 1 and failures[0]["error_code"] == "result_rejected"
        assert operations == [
            "claim", "heartbeat", "heartbeat", "skip", "fail",
            "claim", "heartbeat", "heartbeat", "result",
        ]
    finally:
        client.close()


def test_skip_transient_retry_is_bounded():
    implementation = module()
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(503)

    client = implementation.HttpxWindowClient(
        "http://api", "w" * 40, transport=httpx.MockTransport(respond)
    )
    try:
        with pytest.raises(WorkerTransientError):
            client.skip(str(uuid4()), "worker", 1, "a" * 64)
        assert attempts == 2
    finally:
        client.close()
