import json

import pytest
from das_sentinel.api.window_schemas import WindowReviewCreate
from das_sentinel.persistence.models import Admin
from das_sentinel.persistence.window_models import WindowReview
from das_sentinel.services.event_windows import enqueue_windows
from das_sentinel.window_training import binary_labels

from tests import test_event_windows

context = test_event_windows.context


@pytest.mark.parametrize("state", ["present", "absent", "unknown"])
def test_binary_review_round_trip_and_training_projection(context, state):
    session, site, profile, service = context
    analysis = enqueue_windows(session, site.id, profile.id, [test_event_windows.plan()])[0]
    analysis.status = "succeeded"
    analysis.input_sha256 = "a" * 64
    analysis.result_json = "{}"
    session.add(Admin(id=1, password_hash="test"))
    session.flush()
    payload = WindowReviewCreate(
        analysis_id=analysis.id, input_sha256=analysis.input_sha256,
        labels={"earthquake": state}, note="保留备注", request_id="binary-test",
    ).model_dump()
    saved = service.review(analysis.window_id, **payload, admin_id=1)
    assert saved["labels"] == {"earthquake": state}
    assert saved["background_only"] is (state == "absent")
    assert json.loads(session.get(WindowReview, saved["id"]).labels_json) == {
        "earthquake": state,
    }
    assert binary_labels(saved["labels"]) == {"earthquake": state}
    assert service.review(analysis.window_id, **payload, admin_id=1)["id"] == saved["id"]


def test_legacy_blast_response_is_positive_and_vehicle_absence_is_not_inferred(context):
    session, site, profile, service = context
    analysis = enqueue_windows(session, site.id, profile.id, [test_event_windows.plan()])[0]
    analysis.status = "succeeded"
    analysis.input_sha256 = "b" * 64
    session.add(Admin(id=1, password_hash="test"))
    session.flush()
    for key, labels, expected in [
        ("blast", {"earthquake": "unknown", "blast": "present", "traffic": "present"}, "present"),
        ("vehicle", {"earthquake": "unknown", "blast": "unknown", "traffic": "present"}, "unknown"),
        ("noise", {"earthquake": "absent", "blast": "absent", "traffic": "present"}, "absent"),
    ]:
        record = WindowReview(window_id=analysis.window_id, analysis_id=analysis.id,
                              input_sha256=analysis.input_sha256, labels_json=json.dumps(labels),
                              request_id=key, admin_id=1, note="original")
        session.add(record)
        session.flush()
        assert service._review_dict(record, analysis)["labels"] == {"earthquake": expected}


def test_latest_unknown_review_remains_pending_until_determined(context):
    session, site, profile, service = context
    analysis = enqueue_windows(session, site.id, profile.id, [test_event_windows.plan()])[0]
    analysis.status = "succeeded"
    analysis.input_sha256 = "c" * 64
    analysis.result_json = '{"candidate": true}'
    session.add(Admin(id=1, password_hash="test"))
    session.flush()
    for index, state in enumerate(["unknown", "present", "unknown", "absent"]):
        service.review(
            analysis.window_id, analysis_id=analysis.id,
            input_sha256=analysis.input_sha256, labels={"earthquake": state},
            admin_id=1, note="", request_id=f"pending-{index}",
        )
        pending = service.list(window_seconds=60, review_status="unreviewed", candidate_only=True)
        reviewed = service.list(window_seconds=60, review_status="reviewed")
        assert pending["total"] == (1 if state == "unknown" else 0)
        assert reviewed["total"] == (0 if state == "unknown" else 1)
        assert service.list(window_seconds=60, review_status="all")["total"] == 1


@pytest.mark.parametrize("earthquake,blast,determined", [
    ("unknown", "present", True),
    ("present", "unknown", True),
    ("absent", "absent", True),
    ("unknown", "unknown", False),
    ("absent", "unknown", False),
    ("unknown", "absent", False),
])
def test_legacy_review_filter_uses_binary_interpretation(context, earthquake, blast, determined):
    session, site, profile, service = context
    analysis = enqueue_windows(session, site.id, profile.id, [test_event_windows.plan()])[0]
    analysis.status = "succeeded"
    analysis.input_sha256 = "d" * 64
    session.add(Admin(id=1, password_hash="test"))
    session.flush()
    service.review(
        analysis.window_id, analysis_id=analysis.id, input_sha256=analysis.input_sha256,
        labels={"earthquake": earthquake, "blast": blast, "traffic": "present"},
        admin_id=1, note="", request_id="legacy-pending",
    )
    assert service.list(window_seconds=60, review_status="reviewed")["total"] == int(determined)
    assert service.list(window_seconds=60, review_status="unreviewed")["total"] == int(not determined)
