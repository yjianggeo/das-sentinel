from datetime import UTC, datetime

import pytest
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
    WindowSkip,
)
from das_sentinel.services.event_windows import WindowError, enqueue_windows

from tests.test_event_windows import context as context  # noqa: PLC0414
from tests.test_model_scans import MODEL, add_infer, create, service


def selected(context, tmp_path):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = add_infer(api, context, batch)
    row = api.record_prediction(job["id"], "worker", 1, {
        "start_utc": "2023-09-27T00:00:00+00:00", "status": "succeeded",
        "model_id": MODEL["id"], "score": .6, "model_input_sha256": "c" * 64})
    return api, batch, row, job["payload"]["windows"][0]


def test_model_selected_empty_rule_window_is_not_skipped_or_claimed_by_legacy_worker(context, tmp_path):
    api, batch, _row, plan = selected(context, tmp_path)
    session, site, profile, windows = context
    old = enqueue_windows(session, site.id, profile.id, [plan])[0]
    wid = old.window_id
    session.add(WindowSkip(analysis_id=old.id, window_id=wid, site_id=site.id,
        start_utc=datetime(2023, 9, 27, tzinfo=UTC), window_seconds=30, plan_sha256=old.plan_sha256,
        policy_version="empty-regions-v1", worker_id="old", attempt_count=1, reason="no_qualified_regions"))
    session.delete(old)
    session.flush()
    session.delete(session.get(EventWindow, wid))
    session.flush()
    job = api.claim("model-worker", kind="materialize")
    assert "analysis" in job, "model selection must create a protected materialization claim"
    assert job["analysis"]["skip_empty"] is False
    assert windows.claim("old-worker", supports_empty_skip=True) is None
    analysis = session.get(WindowAnalysis, job["analysis"]["id"])
    assert analysis.skip_policy_version is None
    assert session.get(WindowSkip, old.id) is not None
    assert windows.job(analysis)["model_scan_id"] == batch["id"]
    with pytest.raises(WindowError, match="use_model_scan_control"):
        windows.cancel(analysis.id)


def test_model_selection_never_repoints_a_window_with_human_review(context, tmp_path):
    api, batch, _row, plan = selected(context, tmp_path)
    session, site, profile, windows = context
    old = enqueue_windows(session, site.id, profile.id, [plan])[0]
    from das_sentinel.persistence.models import Admin
    admin = Admin(id=1, password_hash="test")
    session.add(admin)
    session.flush()
    old.status, old.input_sha256 = "succeeded", "d" * 64
    review = WindowReview(window_id=old.window_id, analysis_id=old.id, input_sha256=old.input_sha256,
                         labels_json='{"earthquake":"present"}', note="human", request_id="one", admin_id=admin.id)
    session.add(review)
    session.flush()
    job = api.claim("model-worker", kind="materialize")
    assert job["payload"]["protected"] is True
    assert session.get(EventWindow, old.window_id).latest_analysis_id == old.id
    assert windows.job(old)["status"] == "succeeded"
    assert api.get(batch["id"])["counts"]["protected"] == 1


def test_cancel_invalidates_both_model_and_analysis_ownership(context, tmp_path):
    api, batch, _, _ = selected(context, tmp_path)
    job = api.claim("model-worker", kind="materialize")
    assert "analysis" in job
    api.control(batch["id"], "cancel")
    analysis = context[0].get(WindowAnalysis, job["analysis"]["id"])
    assert analysis.status == "cancelled"
    with pytest.raises(WindowError, match="claim_lost"):
        api.heartbeat(job["id"], "model-worker", job["attempt_count"])


def test_cancelling_old_failed_batch_never_cancels_new_owner_analysis(context, tmp_path):
    api, first, _, plan = selected(context, tmp_path)
    old = api.claim("old", kind="materialize")
    api.fail(old["id"], "old", old["attempt_count"], "disk_error", "disk error")
    second = create(api, context, tmp_path, request="second")
    infer = add_infer(api, context, second)
    api.record_prediction(infer["id"], "worker", 1, {"start_utc": plan["start_utc"], "status": "succeeded",
        "model_id": MODEL["id"], "score": .6, "model_input_sha256": "c" * 64})
    current = api.claim("new", kind="materialize")
    assert current["analysis"]["id"] == old["analysis"]["id"]
    api.control(first["id"], "cancel")
    analysis = context[0].get(WindowAnalysis, current["analysis"]["id"])
    assert analysis.status == "running" and analysis.worker_id == "new"


def test_explicit_retry_preserves_monotonic_claim_generation(context, tmp_path):
    api, batch, _, _ = selected(context, tmp_path)
    old = api.claim("same", kind="materialize")
    api.fail(old["id"], "same", 1, "disk_error", "disk error")
    api.control(batch["id"], "retry")
    new = api.claim("same", kind="materialize")
    assert new["attempt_count"] > old["attempt_count"]
    with pytest.raises(WindowError, match="claim_lost"):
        api.heartbeat(old["id"], "same", old["attempt_count"])
