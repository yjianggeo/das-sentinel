from datetime import UTC, datetime, timedelta
from itertools import count
from uuid import uuid4

import pytest
from das_sentinel.persistence.models import Job, ProcessingProfile
from das_sentinel.persistence.window_models import EventWindow, WindowAnalysis

from tests.test_worker_protocol import api_client as api_client  # noqa: PLC0414


@pytest.fixture
def queue(api_client):
    client, _, _, source_id, profile_id = api_client
    now = datetime.now(UTC)
    sequence = count()

    def add(status, *, age=0, kind="windows", current=True):
        with client.app.state.session_factory.begin() as session:
            if kind == "files":
                row = Job(source_file_id=source_id, processing_profile_id=profile_id,
                          pipeline_version=str(uuid4()), status=status, worker_id="file-worker",
                          heartbeat_at=now - timedelta(seconds=age), updated_at=now - timedelta(seconds=age))
            else:
                wid, aid = str(uuid4()), str(uuid4())
                start = now + timedelta(seconds=30 * next(sequence))
                session.add(EventWindow(id=wid, site_id=session.get(ProcessingProfile, profile_id).site_id,
                    start_utc=start, end_utc=start + timedelta(seconds=30), window_seconds=30,
                    latest_analysis_id=aid if current else None))
                session.flush()
                row = WindowAnalysis(id=aid, window_id=wid, profile_id=profile_id,
                    plan_json="{}", plan_sha256=uuid4().hex, algorithm_version="whole-window-v2",
                    status=status, worker_id="window-worker", heartbeat_at=now - timedelta(seconds=age))
            session.add(row)

    return client, add


def estimate(client):
    response = client.get("/api/v1/jobs/estimate")
    assert response.status_code == 200
    return response.json()


def test_whole_queue_estimate_ignores_pagination_and_terminal_jobs(queue):
    client, add = queue
    for _ in range(99):
        add("queued")
    add("running")
    for age in [30, 120, 210, 300]:
        add("succeeded", age=age)
    for status in ["failed", "cancelled"]:
        add(status)
    add("queued", current=False)
    data = estimate(client)
    assert data["queued"] == 99 and data["running"] == 1
    assert data["status"] == "running" and data["remaining_seconds"] == 9000
    assert (datetime.fromisoformat(data["estimated_finish_at"]) - datetime.fromisoformat(data["as_of"])).total_seconds() == 9000
    assert data["queues"][0]["sample_count"] == 4
    response = client.get("/api/v1/jobs/estimate", params={"limit": 1, "offset": 80, "status": "failed"})
    assert response.json()["remaining_seconds"] == 9000


def test_retired_file_queue_is_not_included_in_window_estimate(queue):
    client, add = queue
    for kind, gap, queued in [("windows", 90, 19), ("files", 60, 9)]:
        add("running", kind=kind)
        for _ in range(queued):
            add("queued", kind=kind)
        for age in [30 + gap * i for i in range(4)]:
            add("succeeded", kind=kind, age=age)
    data = estimate(client)
    assert data["remaining_seconds"] == 1800
    assert data["queued"] == 19 and data["running"] == 1
    assert [q["key"] for q in data["queues"]] == ["windows"]


@pytest.mark.parametrize("active,history,status", [
    ("queued", [], "waiting_worker"),
    ("running", [], "estimating"),
    ("running", [9000, 9100, 9200, 9300], "estimating"),
])
def test_no_fabricated_eta_without_recent_activity_and_samples(queue, active, history, status):
    client, add = queue
    add(active)
    for age in history:
        add("succeeded", age=age)
    data = estimate(client)
    assert data["status"] == status
    assert data["remaining_seconds"] is None and data["estimated_finish_at"] is None


def test_expired_heartbeat_does_not_show_a_running_countdown(queue):
    client, add = queue
    add("running", age=600)
    for age in [600, 690, 780, 870]:
        add("succeeded", age=age)
    assert estimate(client)["status"] == "waiting_worker"


def test_empty_queue_is_complete_even_with_failed_and_cancelled_records(queue):
    client, add = queue
    add("failed")
    add("cancelled")
    data = estimate(client)
    assert data["status"] == "complete" and data["remaining_seconds"] == 0
    assert data["estimated_finish_at"] is None


def test_retired_file_queue_does_not_block_window_finish_time(queue):
    client, add = queue
    add("running")
    add("queued", kind="files")
    for age in [30, 120, 210, 300]:
        add("succeeded", age=age)
    data = estimate(client)
    assert data["status"] == "running" and data["remaining_seconds"] == 90
    assert data["queues"][0]["remaining_seconds"] == 90


def test_estimate_requires_login(queue):
    client, _ = queue
    client.cookies.clear()
    assert client.get("/api/v1/jobs/estimate").status_code == 401


def test_recent_sample_limit_excludes_older_slower_work(queue):
    client, add = queue
    add("running")
    for age in [7000, 6000, 5000] + [30 + i * 60 for i in range(31)]:
        add("succeeded", age=age)
    add("succeeded", age=-600)
    data = estimate(client)
    assert data["remaining_seconds"] == 60
    assert data["queues"][0]["sample_count"] == 31


def test_queued_work_between_claims_uses_recent_completion_activity(queue):
    client, add = queue
    add("queued")
    for age in [1, 61, 121, 181]:
        add("succeeded", age=age)
    data = estimate(client)
    assert data["status"] == "running" and data["remaining_seconds"] == 60


def test_skipped_empty_windows_count_toward_throughput_without_phantom_pending_jobs(queue):
    from das_sentinel.persistence.window_models import WindowSkip
    client, add = queue
    add("queued")
    add("succeeded", age=121)
    with client.app.state.session_factory.begin() as session:
        window = session.query(EventWindow).first()
        now = datetime.now(UTC)
        for age in [1, 61, 181, 9000, -300]:
            session.add(WindowSkip(analysis_id=str(uuid4()), window_id=str(uuid4()),
                site_id=window.site_id, start_utc=now, window_seconds=30,
                plan_sha256="a" * 64, policy_version="empty-regions-v1",
                worker_id="worker", attempt_count=1, finished_at=now - timedelta(seconds=age),
                reason="no_qualified_regions"))
    data = estimate(client)
    assert data["queued"] == 1 and data["running"] == 0
    assert data["status"] == "running"
    assert 59 <= data["remaining_seconds"] <= 61
    assert data["queues"][0]["sample_count"] == 4
