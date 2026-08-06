"""Long-running worker entrypoint used by the container deployment.

``python -m das_sentinel.worker`` polls the internal API for jobs, analyses
one file per claim, and sleeps briefly when the queue is idle. The loop is
factored behind :func:`run_forever` so tests can drive a bounded number of
iterations without spawning a real process.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Callable
from typing import Protocol

from das_sentinel.config import Settings
from das_sentinel.signal.backend import smoke_test_cuda
from das_sentinel.worker.main import (
    HttpxWorkerClient,
    Worker,
    WorkerOutcome,
    WorkerProtocol,
    WorkerTransientError,
)

DEFAULT_IDLE_SECONDS = 5.0
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)


class _Runnable(Protocol):
    def run_once(self) -> WorkerOutcome: ...


def run_forever(
    worker: _Runnable,
    *,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
    max_iterations: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll until stopped; return the number of iterations executed."""
    iterations = 0
    transient_failures = 0
    while max_iterations is None or iterations < max_iterations:
        try:
            outcome = worker.run_once()
        except WorkerTransientError:
            delay = BACKOFF_SECONDS[
                min(transient_failures, len(BACKOFF_SECONDS) - 1)
            ]
            transient_failures += 1
            sleep(delay)
        else:
            transient_failures = 0
            if outcome.status == "idle":
                sleep(idle_seconds)
        iterations += 1
    return iterations


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]  # secrets come from DAS_* env
    worker_id = os.environ.get("DAS_WORKER_ID") or socket.gethostname()
    if settings.worker_device == "cuda":
        smoke_test_cuda()
    client: WorkerProtocol = HttpxWorkerClient(settings.api_url, settings.worker_token)
    worker = Worker(settings=settings, client=client, worker_id=worker_id)
    run_forever(worker)


if __name__ == "__main__":
    main()
