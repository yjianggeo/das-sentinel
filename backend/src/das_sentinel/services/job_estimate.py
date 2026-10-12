"""Read-only estimates from recent completion throughput, across entire queues."""
import math
from datetime import timedelta

from sqlalchemy import func, select

from das_sentinel.persistence.models import utc_now
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowSkip,
)
from das_sentinel.services.event_windows import LEASE_SECONDS

LOOKBACK = timedelta(hours=2)
MAX_SAMPLES = 31
MIN_SAMPLES = 4


def _queue(session, *, key, label, model, completed_at, now, lease, current=()):
    counts = dict(session.execute(
        select(model.status, func.count()).where(*current).group_by(model.status)
    ).all())
    queued, running = counts.get("queued", 0), counts.get("running", 0)
    result = {"key": key, "label": label, "queued": queued, "running": running,
                  "sample_count": 0, "remaining_seconds": None, "status": "complete"}
    if queued + running == 0:
        return result | {"remaining_seconds": 0}
    # Whole-window workers send a final heartbeat immediately before publishing.
    # Queue age is never used.
    completed = session.scalars(select(completed_at).where(
        *current, model.status == "succeeded", completed_at >= now - LOOKBACK,
        completed_at <= now,
    ).order_by(completed_at.desc()).limit(MAX_SAMPLES)).all()
    if model is WindowAnalysis:
        skipped = session.scalars(select(WindowSkip.finished_at).where(
            WindowSkip.finished_at >= now - LOOKBACK, WindowSkip.finished_at <= now,
        ).order_by(WindowSkip.finished_at.desc()).limit(MAX_SAMPLES)).all()
        completed = sorted([*completed, *skipped], reverse=True)[:MAX_SAMPLES]
    result["sample_count"] = len(completed)
    heartbeat = session.scalar(select(func.max(model.heartbeat_at)).where(
        *current, model.status == "running", model.heartbeat_at <= now,
    ))
    activity = [value for value in (heartbeat, completed[0] if completed else None) if value is not None]
    if not activity or max(activity) < now - lease:
        return result | {"status": "waiting_worker"}
    if len(completed) < MIN_SAMPLES or completed[0] <= completed[-1]:
        return result | {"status": "estimating"}
    # Observed throughput already includes concurrency; do not divide by workers again.
    seconds_per_job = (completed[0] - completed[-1]).total_seconds() / (len(completed) - 1)
    return result | {"status": "running", "remaining_seconds": math.ceil((queued + running) * seconds_per_job)}


def estimate_jobs(session):
    now = utc_now()
    current_window = WindowAnalysis.id.in_(select(EventWindow.latest_analysis_id))
    queues = [
        _queue(session, key="windows", label="整窗任务", model=WindowAnalysis,
               completed_at=WindowAnalysis.heartbeat_at, now=now,
               lease=timedelta(seconds=LEASE_SECONDS), current=(current_window,)),
    ]
    remaining = None
    states = {queue["status"] for queue in queues}
    if "waiting_worker" in states:
        status = "waiting_worker"
    elif "estimating" in states:
        status = "estimating"
    else:
        remaining = max(queue["remaining_seconds"] for queue in queues)
        status = "running" if remaining else "complete"
    return {"as_of": now, "status": status, "queued": sum(q["queued"] for q in queues),
                "running": sum(q["running"] for q in queues), "remaining_seconds": remaining,
                "estimated_finish_at": now + timedelta(seconds=remaining) if remaining else None,
                "queues": queues}
