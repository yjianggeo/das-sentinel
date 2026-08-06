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


@pytest.mark.parametrize(
    ("worker_device", "capability", "expected_calls"),
    [
        ("auto", "cuda", ["select:auto", "smoke", "client", "claim"]),
        ("auto", "cpu", ["select:auto", "client", "claim"]),
        ("cpu", None, ["client", "claim"]),
        ("cuda", None, ["smoke", "client", "claim"]),
    ],
    ids=("auto-cuda", "auto-cpu", "explicit-cpu", "explicit-cuda"),
)
def test_worker_startup_gates_cuda_smoke_before_client_and_claim(
    monkeypatch: pytest.MonkeyPatch,
    worker_device: str,
    capability: str | None,
    expected_calls: list[str],
) -> None:
    calls = []
    settings = SimpleNamespace(
        worker_device=worker_device,
        api_url="http://api",
        worker_token="w" * 40,
    )

    class FakeClient:
        def __init__(self, api_url, worker_token):
            del api_url, worker_token
            calls.append("client")

        def claim(self, worker_id):
            del worker_id
            calls.append("claim")

    class FakeWorker:
        def __init__(self, *, client, **kwargs):
            del kwargs
            self.client = client

        def run_once(self):
            self.client.claim("worker")
            return WorkerOutcome(status="idle")

    def select(device: str) -> SimpleNamespace:
        calls.append(f"select:{device}")
        if capability is None:
            pytest.fail(f"unexpected backend probe: {device}")
        return SimpleNamespace(name=capability)

    monkeypatch.setattr(worker_main, "Settings", lambda: settings)
    monkeypatch.setattr(worker_main, "select_backend", select)
    monkeypatch.setattr(worker_main, "smoke_test_cuda", lambda: calls.append("smoke"))
    monkeypatch.setattr(worker_main, "HttpxWorkerClient", FakeClient)
    monkeypatch.setattr(worker_main, "Worker", FakeWorker)
    monkeypatch.setattr(worker_main, "run_forever", lambda worker: worker.run_once())

    worker_main.main()

    assert calls == expected_calls
