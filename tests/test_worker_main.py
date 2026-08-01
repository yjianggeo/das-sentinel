from __future__ import annotations

from das_sentinel.worker.__main__ import run_forever
from das_sentinel.worker.main import WorkerOutcome


class _StubWorker:
    def __init__(self, statuses: list[str]) -> None:
        self._statuses = list(statuses)
        self.calls = 0

    def run_once(self) -> WorkerOutcome:
        self.calls += 1
        status = self._statuses.pop(0) if self._statuses else "idle"
        return WorkerOutcome(status=status)


def test_run_forever_executes_bounded_iterations_without_sleeping_long() -> None:
    worker = _StubWorker(["idle", "processed"])

    iterations = run_forever(worker, idle_seconds=0.0, max_iterations=2)

    assert iterations == 2
    assert worker.calls == 2
