"""Deletion is limited to explicitly authorized empty drafts; history survives."""
import json
from datetime import timedelta

import pytest
from das_sentinel.persistence.models import Admin, utc_now
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowLabelImport,
    WindowReview,
)
from das_sentinel.services.event_windows import WindowError, enqueue_windows
from sqlalchemy import func, select

from tests import test_event_windows as original

context = original.context


def plan():
    return original.plan() | {"end_utc": "2023-09-27T00:00:30+00:00"}


def queued(context):
    session, site, profile, _service = context
    return enqueue_windows(session, site.id, profile.id, [plan()])[0]


def imported(session, analysis, label):
    session.add(Admin(id=1, password_hash="test"))
    session.flush()
    record = {"event_id": "legacy-event", "history": [{"label": label}]}
    row = WindowLabelImport(window_id=analysis.window_id, analysis_id=analysis.id,
        payload_json=json.dumps({"records": [record]}), admin_id=1)
    session.add(row)
    session.flush()
    return row


def test_empty_skip_deletes_draft_and_is_idempotent_and_prevents_reenqueue(context):
    session, site, profile, service = context
    a = queued(context)
    aid, wid, signature = a.id, a.window_id, a.plan_sha256
    imported(session, a, "noise")
    claim = service.claim("new", supports_empty_skip=True)
    assert claim["skip_empty"] is True
    assert claim["plan_sha256"] == signature
    assert service.skip(aid, "new", 1, signature) == {"id": aid, "status": "skipped"}
    assert session.get(WindowAnalysis, aid) is None
    assert session.get(EventWindow, wid) is None
    assert session.get(WindowLabelImport, wid) is None
    assert session.scalar(select(func.count()).select_from(WindowReview)) == 0
    assert service.skip(aid, "new", 1, signature)["status"] == "skipped"
    with pytest.raises(WindowError, match="claim_lost"):
        service.skip(aid, "other", 1, signature)
    assert enqueue_windows(session, site.id, profile.id, [plan()]) == []
    assert session.get(EventWindow, wid) is None
    changed = plan() | {"algorithm_version": "changed"}
    assert len(enqueue_windows(session, site.id, profile.id, [changed])) == 1


@pytest.mark.parametrize("protection", ["old_worker", "60s", "artifact", "result", "history", "positive", "review"])
def test_protected_windows_never_receive_skip_permission(context, protection):
    session, site, profile, service = context
    a = (enqueue_windows(session, site.id, profile.id, [original.plan()])[0]
         if protection == "60s" else queued(context))
    if protection == "artifact":
        output = service.artifact_root / a.id / "attempt-1"
        output.mkdir(parents=True)
        (output / "waterfall.png").write_bytes(b"existing image")
    elif protection == "result":
        a.result_json = "{}"
    elif protection == "history":
        a = enqueue_windows(session, site.id, profile.id, [plan() | {"algorithm_version": "new"}])[0]
    elif protection == "positive":
        imported(session, a, "earthquake")
    elif protection == "review":
        session.add(Admin(id=1, password_hash="test"))
        session.flush()
        session.add(WindowReview(window_id=a.window_id, analysis_id=a.id,
            input_sha256="a"*64, labels_json="{}", request_id="review", admin_id=1))
    session.flush()
    claim = service.claim("worker", supports_empty_skip=protection != "old_worker")
    assert claim["skip_empty"] is False
    with pytest.raises(WindowError, match="skip_not_allowed"):
        service.skip(a.id, "worker", 1, a.plan_sha256)
    assert session.get(EventWindow, a.window_id) is not None


def test_original_running_lease_reclaim_does_not_enable_skip(context):
    session, _, _, service = context
    a = queued(context)
    service.claim("old")
    a.heartbeat_at = utc_now() - timedelta(minutes=10)
    session.flush()
    assert service.claim("new", supports_empty_skip=True)["skip_empty"] is False


@pytest.mark.parametrize("original_supports_empty_skip", [False, True])
def test_retry_preserves_original_empty_skip_authorization(context, original_supports_empty_skip):
    session, _, _, service = context
    a = queued(context)
    aid, wid, signature = a.id, a.window_id, a.plan_sha256
    service.claim("original", supports_empty_skip=original_supports_empty_skip)
    service.fail(aid, "original", "window_analysis_failed", "retry fixture", 1)
    service.retry(aid)

    claim = service.claim("new", supports_empty_skip=True)
    assert claim["attempt_count"] == 2
    assert claim["skip_empty"] is original_supports_empty_skip
    if original_supports_empty_skip:
        assert service.skip(aid, "new", 2, signature)["status"] == "skipped"
    else:
        assert claim["skip_policy_version"] is None
        with pytest.raises(WindowError, match="skip_not_allowed"):
            service.skip(aid, "new", 2, signature)
        assert session.get(WindowAnalysis, aid) is not None
        assert session.get(EventWindow, wid) is not None


@pytest.mark.parametrize("change", ["cancelled", "lease", "profile", "source", "positive", "artifact", "new_version", "signature"])
def test_skip_revalidates_final_state_without_deleting(context, change):
    session, site, profile, service = context
    a = queued(context)
    service.claim("new", supports_empty_skip=True)
    signature = a.plan_sha256
    if change == "cancelled":
        service.cancel(a.id)
    elif change == "lease":
        a.heartbeat_at = utc_now() - timedelta(minutes=10)
    elif change == "profile":
        profile.is_current = False
    elif change == "source":
        from pathlib import Path
        Path(json.loads(a.plan_json)["sources"][0]["path"]).write_bytes(b"changed")
    elif change == "positive":
        imported(session, a, "traffic")
    elif change == "artifact":
        output = service.artifact_root / a.id
        output.mkdir(parents=True)
        (output / "result.json").write_text("{}")
    elif change == "new_version":
        enqueue_windows(session, site.id, profile.id, [plan() | {"algorithm_version": "new"}])
    elif change == "signature":
        signature = "0" * 64
    session.flush()
    with pytest.raises(WindowError):
        service.skip(a.id, "new", 1, signature)
    assert session.get(WindowAnalysis, a.id) is not None
    assert session.get(EventWindow, a.window_id) is not None


def test_skip_transaction_rollback_restores_draft(context):
    session, _, _, service = context
    a = queued(context)
    aid, wid, signature = a.id, a.window_id, a.plan_sha256
    service.claim("new", supports_empty_skip=True)
    session.commit()
    service.skip(aid, "new", 1, signature)
    session.rollback()
    assert session.get(WindowAnalysis, aid).status == "running"
    assert session.get(EventWindow, wid) is not None


def test_authorized_expired_attempt_can_be_reclaimed_but_old_attempt_cannot_skip(context):
    session, _, _, service = context
    a = queued(context)
    service.claim("new", supports_empty_skip=True)
    a.heartbeat_at = utc_now() - timedelta(minutes=10)
    session.flush()
    assert service.claim("new", supports_empty_skip=True)["skip_empty"] is True
    with pytest.raises(WindowError, match="claim_lost"):
        service.skip(a.id, "new", 1, a.plan_sha256)
    assert service.skip(a.id, "new", 2, a.plan_sha256)["status"] == "skipped"


def test_upgrade_existing_schema_preserves_plan_and_running_job(context):
    from das_sentinel.persistence.db import upgrade_database
    session, _, _, service = context
    a = queued(context)
    aid, signature = a.id, a.plan_sha256
    service.claim("old")
    session.commit()
    engine = session.bind
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE window_skips")
        connection.exec_driver_sql("ALTER TABLE window_analyses DROP COLUMN skip_policy_version")
        connection.exec_driver_sql("UPDATE alembic_version SET version_num='0005_window_duration'")
    upgrade_database(str(engine.url))
    session.expire_all()
    existing = session.get(WindowAnalysis, aid)
    assert existing.plan_sha256 == signature and existing.status == "running"
    assert existing.skip_policy_version is None
    assert service.heartbeat(aid, "old", 1)["status"] == "running"
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
