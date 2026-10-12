import importlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
)
from das_sentinel.services.event_windows import WindowError
from sqlalchemy import func, select

from tests.test_event_windows import context as context  # noqa: PLC0414

MODEL = {"id": "fixed-model", "name": "300 m", "weights_sha256": "a" * 64,
         "metadata_sha256": "b" * 64, "threshold": .04}


def service(context, tmp_path):
    assert importlib.util.find_spec("das_sentinel.services.model_scans"), "persistent model scans missing"
    cls = importlib.import_module("das_sentinel.services.model_scans").ModelScanService
    return cls(context[0], tmp_path)


def create(api, context, tmp_path, request="one"):
    return api.create(context[1].id, context[2].id, request, [str(tmp_path)], MODEL, [])


def test_create_replay_freezes_model_and_rejects_conflicting_request(context, tmp_path):
    api = service(context, tmp_path)
    first = create(api, context, tmp_path)
    assert create(api, context, tmp_path)["id"] == first["id"]
    assert api.get(first["id"])["model"] == MODEL
    assert api.get(first["id"])["counts"]["judged"] == 0
    with pytest.raises(WindowError, match="request_conflict"):
        api.create(context[1].id, context[2].id, "one", [str(tmp_path)], MODEL | {"threshold": .9}, [])


def test_pause_allows_owned_completion_cancel_fences_late_work(context, tmp_path):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = api.claim("worker")
    api.control(batch["id"], "pause")
    assert api.claim("other") is None
    api.heartbeat(job["id"], "worker", job["attempt_count"])
    with pytest.raises(WindowError, match="claim_lost"):
        api.heartbeat(job["id"], "other", job["attempt_count"])
    api.control(batch["id"], "cancel")
    with pytest.raises(WindowError, match="claim_lost"):
        api.heartbeat(job["id"], "worker", job["attempt_count"])
    with pytest.raises(WindowError, match="cancelled"):
        api.control(batch["id"], "resume")
    with pytest.raises(WindowError, match="cancelled"):
        api.control(batch["id"], "retry")


def add_infer(api, context, batch):
    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    unit = ModelScanUnit(scan_id=batch["id"], kind="infer", unit_key="block",
                         payload_json=json.dumps({"windows": [{"start_utc": "2023-09-27T00:00:00+00:00",
                            "end_utc": "2023-09-27T00:00:30+00:00", "sources": [], "coverage": "complete"}]}))
    context[0].add(unit)
    context[0].flush()
    return api.claim("worker", kind="infer")


def test_negative_replay_does_not_create_events_or_human_labels(context, tmp_path):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = add_infer(api, context, batch)
    result = {"start_utc": "2023-09-27T00:00:00+00:00", "status": "succeeded",
              "model_id": MODEL["id"], "score": .01, "model_input_sha256": "c" * 64}
    first = api.record_prediction(job["id"], "worker", job["attempt_count"], result)
    again = api.record_prediction(job["id"], "worker", job["attempt_count"], result)
    assert first["id"] == again["id"]
    assert first["status"] == "negative"
    assert api.get(batch["id"])["counts"]["negative"] == 1
    for table in (EventWindow, WindowAnalysis, WindowReview):
        assert context[0].scalar(select(func.count()).select_from(table)) == 0
    with pytest.raises(WindowError, match="result_conflict"):
        api.record_prediction(job["id"], "worker", job["attempt_count"], result | {"score": .02})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 1.1, True])
def test_invalid_score_never_becomes_negative(context, tmp_path, score):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = add_infer(api, context, batch)
    with pytest.raises(WindowError, match="invalid_prediction"):
        api.record_prediction(job["id"], "worker", job["attempt_count"], {
            "start_utc": "2023-09-27T00:00:00+00:00", "status": "succeeded",
            "model_id": MODEL["id"], "score": score, "model_input_sha256": "c" * 64})
    assert api.get(batch["id"])["counts"]["judged"] == 0


def test_threshold_equality_selects_and_failure_has_no_class(context, tmp_path):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = add_infer(api, context, batch)
    row = api.record_prediction(job["id"], "worker", job["attempt_count"], {
        "start_utc": "2023-09-27T00:00:00+00:00", "status": "succeeded",
        "model_id": MODEL["id"], "score": .04, "model_input_sha256": "c" * 64})
    assert row["status"] == "selected"
    assert api.get(batch["id"])["counts"]["pending_artifacts"] == 1
    assert api.claim("worker", kind="materialize")["payload"]["receipt_id"] == row["id"]


def test_expired_owner_fenced_after_reclaim(context, tmp_path):
    api = service(context, tmp_path)
    create(api, context, tmp_path)
    first = api.claim("old")
    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    unit = context[0].get(ModelScanUnit, first["id"])
    unit.heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    context[0].flush()
    second = api.claim("new")
    assert second["id"] == first["id"] and second["attempt_count"] == 2
    with pytest.raises(WindowError, match="claim_lost"):
        api.heartbeat(first["id"], "old", 1)


def test_failed_input_has_no_score_and_transaction_can_roll_back(context, tmp_path):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = add_infer(api, context, batch)
    with context[0].begin_nested() as savepoint:
        row = api.record_prediction(job["id"], "worker", job["attempt_count"], {
            "start_utc": "2023-09-27T00:00:00+00:00", "status": "failed",
            "error_code": "coverage_incomplete", "error_message": "gap"})
        assert row["status"] == "unjudgeable" and row["score"] is None
        savepoint.rollback()
    assert api.get(batch["id"])["counts"]["unjudgeable"] == 0
    row = api.record_prediction(job["id"], "worker", job["attempt_count"], {
        "start_utc": "2023-09-27T00:00:00+00:00", "status": "failed",
        "error_code": "coverage_incomplete", "error_message": "gap"})
    assert row["status"] == "unjudgeable"


def test_new_migration_is_repeatable_and_keeps_foreign_keys(tmp_path):
    from das_sentinel.persistence.db import create_local_engine, upgrade_database
    url = f"sqlite+pysqlite:///{tmp_path / 'migration.sqlite3'}"
    upgrade_database(url)
    upgrade_database(url)
    engine = create_local_engine(url)
    with engine.connect() as conn:
        assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        assert conn.exec_driver_sql("PRAGMA integrity_check").scalar_one() == "ok"
        assert conn.exec_driver_sql("SELECT count(*) FROM model_scan_receipts").scalar_one() == 0
    engine.dispose()


def test_expired_lease_cannot_be_revived_without_reclaim(context, tmp_path):
    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    api = service(context, tmp_path)
    create(api, context, tmp_path)
    first = api.claim("old")
    context[0].get(ModelScanUnit, first["id"]).heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    context[0].flush()
    with pytest.raises(WindowError, match="claim_lost"):
        api.heartbeat(first["id"], "old", 1)


def test_reclaim_skips_committed_results_and_retry_reopens_only_failed_windows(context, tmp_path):
    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = add_infer(api, context, batch)
    unit = context[0].get(ModelScanUnit, job["id"])
    plans = json.loads(unit.payload_json)["windows"]
    plans.append(plans[0] | {"start_utc": "2023-09-27T00:00:30+00:00", "end_utc": "2023-09-27T00:01:00+00:00"})
    unit.payload_json = json.dumps({"windows": plans})
    api.record_prediction(job["id"], "worker", 1, {"start_utc": plans[0]["start_utc"], "status": "failed", "error_code": "temporary_read"})
    unit.heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    context[0].flush()
    reclaimed = api.claim("new", kind="infer")
    assert [p["start_utc"] for p in reclaimed["payload"]["windows"]] == [plans[1]["start_utc"]]
    api.record_prediction(unit.id, "new", 2, {"start_utc": plans[1]["start_utc"], "status": "succeeded",
        "model_id": MODEL["id"], "score": .01, "model_input_sha256": "c" * 64})
    api.control(batch["id"], "retry")
    retry = api.claim("retry", kind="infer")
    assert [p["start_utc"] for p in retry["payload"]["windows"]] == [plans[0]["start_utc"]]


def test_scan_estimate_requires_inventory_activity_and_real_samples(context, tmp_path):
    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    from sqlalchemy import event
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    assert api.get(batch["id"])["estimate_status"] == "discovering"
    now = datetime.now(UTC)
    for row in context[0].scalars(select(ModelScanUnit)):
        row.status = "succeeded"
    planner = api._add_unit(batch["id"], "plan", "coverage", {"cursor": None})
    planner.pages_json = json.dumps({"total": 100})
    for i in range(4):
        row = api._add_unit(batch["id"], "infer", str(i), {"windows": []})
        row.status = "succeeded"
        row.window_count = 10
        row.started_at = now - timedelta(seconds=100 - i * 20)
        row.finished_at = row.started_at + timedelta(seconds=20)
    active = api._add_unit(batch["id"], "infer", "active", {"windows": []})
    active.status, active.heartbeat_at = "running", now
    context[0].flush()
    planner_queries = []

    def capture_planner(_connection, _cursor, statement, parameters, _execution_context, _many):
        if "FROM model_scan_units" in statement and "plan" in parameters:
            planner_queries.append(statement)

    engine = context[0].get_bind()
    event.listen(engine, "before_cursor_execute", capture_planner)
    try:
        estimate = api.get(batch["id"])
    finally:
        event.remove(engine, "before_cursor_execute", capture_planner)
    assert estimate["total"] == 100
    assert estimate["remaining_seconds"] == 200
    assert estimate["estimated_finish_at"] is not None
    assert len(planner_queries) == 1
    projection = planner_queries[0].split("FROM", 1)[0]
    assert "model_scan_units.pages_json" in projection
    assert "model_scan_units.payload_json" not in projection
    assert projection.split("SELECT", 1)[1].strip() == "model_scan_units.pages_json"
    api.control(batch["id"], "pause")
    assert api.get(batch["id"])["remaining_seconds"] is None
    assert api.get(batch["id"])["estimate_status"] == "paused"
