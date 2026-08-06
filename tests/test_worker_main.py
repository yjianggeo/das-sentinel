from __future__ import annotations

from types import SimpleNamespace

import pytest
from das_sentinel.worker import __main__ as worker_main
from das_sentinel.worker.__main__ import run_forever
from das_sentinel.worker.main import (
    WorkerFatalProtocolError,
    WorkerOutcome,
    WorkerTransientError,
)


class _StubWorker:
    def __init__(self, statuses: list[str]) -> None:
        self._statuses = list(statuses)
        self.calls = 0

    def run_once(self) -> WorkerOutcome:
        self.calls += 1
        status = self._statuses.pop(0) if self._statuses else "idle"
        return WorkerOutcome(status=status)


class _SequenceWorker:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def run_once(self):
        value = next(self.outcomes)
        if isinstance(value, BaseException):
            raise value
        return WorkerOutcome(status=value)


def test_run_forever_executes_bounded_iterations_without_sleeping_long() -> None:
    worker = _StubWorker(["idle", "processed"])

    iterations = run_forever(worker, idle_seconds=0.0, max_iterations=2)

    assert iterations == 2
    assert worker.calls == 2


def test_transient_failures_back_off_and_success_resets_delay() -> None:
    sleeps = []
    worker = _SequenceWorker(
        [
            WorkerTransientError("offline"),
            WorkerTransientError("offline"),
            "idle",
            WorkerTransientError("offline"),
        ]
    )
    run_forever(worker, idle_seconds=0.0, max_iterations=4, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0, 0.0, 1.0]


def test_transient_backoff_caps_at_thirty_seconds() -> None:
    sleeps = []
    worker = _SequenceWorker([WorkerTransientError("offline")] * 7)
    run_forever(worker, max_iterations=7, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_fatal_protocol_error_exits_without_retry() -> None:
    worker = _SequenceWorker([WorkerFatalProtocolError("unauthorized")])
    with pytest.raises(WorkerFatalProtocolError, match="unauthorized"):
        run_forever(worker, max_iterations=1, sleep=lambda _: None)


def test_cuda_smoke_runs_before_http_client_construction(monkeypatch) -> None:
    calls = []
    settings = SimpleNamespace(
        worker_device="cuda",
        api_url="http://api",
        worker_token="w" * 40,
    )

    class FakeClient:
        def __init__(self, api_url, worker_token):
            del api_url, worker_token
            calls.append("client")

    monkeypatch.setattr(worker_main, "Settings", lambda: settings)
    monkeypatch.setattr(
        worker_main,
        "smoke_test_cuda",
        lambda: calls.append("smoke"),
        raising=False,
    )
    monkeypatch.setattr(worker_main, "HttpxWorkerClient", FakeClient)
    monkeypatch.setattr(worker_main, "Worker", lambda **kwargs: object())
    monkeypatch.setattr(worker_main, "run_forever", lambda worker: None)

    worker_main.main()

    assert calls == ["smoke", "client"]
