import json

import pytest
from das_sentinel.persistence.models import Admin, ProcessingProfile, Site
from das_sentinel.persistence.window_models import WindowReview
from das_sentinel.services.event_windows import WindowService

from tests.test_window_favorites import context as context  # noqa: PLC0414
from tests.test_window_favorites import ready

pytest_plugins = ("tests.test_events_api",)


def review(service, analysis, state, request_id):
    return service.review(analysis.window_id, analysis_id=analysis.id,
                          input_sha256=analysis.input_sha256,
                          labels={"earthquake": state}, note="keep this review",
                          request_id=request_id, admin_id=1)


def test_review_class_filters_split_latest_determined_labels(context):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    positive = ready(session, site.id, profile.id, minute=0)
    negative = ready(session, site.id, profile.id, minute=1)
    uncertain = ready(session, site.id, profile.id, minute=2)
    ready(session, site.id, profile.id, minute=3)  # No manual review.
    review(service, positive, "present", "positive")
    review(service, negative, "present", "old-positive")
    review(service, negative, "absent", "new-negative")
    review(service, uncertain, "unknown", "uncertain")

    earthquake = service.list(review_status="reviewed_earthquake")
    noise = service.list(review_status="reviewed_noise")
    assert earthquake["total"] == 1
    assert [item["id"] for item in earthquake["items"]] == [positive.window_id]
    assert noise["total"] == 1
    assert [item["id"] for item in noise["items"]] == [negative.window_id]
    assert service.list(review_status="reviewed")["total"] == 2
    assert service.list(review_status="unreviewed")["total"] == 2


@pytest.mark.parametrize("stale_reason", ["input", "profile"])
def test_review_class_filters_exclude_stale_reviews(context, stale_reason):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    for minute, state in enumerate(("present", "absent")):
        analysis = ready(session, site.id, profile.id, minute=minute)
        review(service, analysis, state, state)
        if stale_reason == "input":
            analysis.input_sha256 = "b" * 64
    if stale_reason == "profile":
        profile.is_current = False
    session.flush()
    assert service.list(review_status="reviewed_earthquake")["total"] == 0
    assert service.list(review_status="reviewed_noise")["total"] == 0
    assert service.list(review_status="unreviewed")["total"] == 2


def test_review_class_filters_preserve_legacy_blast_and_traffic_semantics(context):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    ids = []
    for minute, labels in enumerate([
        {"earthquake": "absent", "blast": "present", "traffic": "present"},
        {"earthquake": "absent", "blast": "absent", "traffic": "present"},
        {"earthquake": "unknown", "blast": "unknown", "traffic": "present"},
    ]):
        analysis = ready(session, site.id, profile.id, minute=minute)
        ids.append(analysis.window_id)
        session.add(WindowReview(window_id=analysis.window_id, analysis_id=analysis.id,
                                 input_sha256=analysis.input_sha256, labels_json=json.dumps(labels),
                                 request_id=f"legacy-{minute}", admin_id=1))
    session.flush()
    assert [item["id"] for item in service.list(review_status="reviewed_earthquake")["items"]] == [ids[0]]
    assert [item["id"] for item in service.list(review_status="reviewed_noise")["items"]] == [ids[1]]
    assert service.list(review_status="unreviewed")["items"][0]["id"] == ids[2]


def test_class_favorite_and_site_filters_apply_before_count_and_pagination(context):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    rows = [ready(session, site.id, profile.id, minute=i, priority=10-i) for i in range(4)]
    for index, analysis in enumerate(rows):
        review(service, analysis, "absent" if index == 0 else "present", f"review-{index}")
        if index != 3:
            service.set_favorite(analysis.window_id, True, admin_id=1)
    other_site = Site(name="other", data_path=site.data_path, timezone="UTC", cable_type="downhole")
    session.add(other_site)
    session.flush()
    other_profile = ProcessingProfile(site_id=other_site.id, version=1, payload_json="{}", is_current=True)
    session.add(other_profile)
    session.flush()
    other = ready(session, other_site.id, other_profile.id)
    review(service, other, "present", "other")
    service.set_favorite(other.window_id, True, admin_id=1)
    page = service.list(review_status="reviewed_earthquake", site_id=site.id,
                        favorites_only=True, admin_id=1, limit=1, offset=1)
    assert page["total"] == 2
    assert [item["id"] for item in page["items"]] == [rows[2].window_id]
    assert page["items"][0]["favorite"] is True


def test_api_exposes_both_review_class_filters(event_api):
    client, settings, sessions, _csrf, ids = event_api
    with sessions.begin() as session:
        service = WindowService(session, settings.app_data_dir / "window-artifacts")
        positive = ready(session, ids["site_a"], ids["profile_a"], minute=0)
        negative = ready(session, ids["site_a"], ids["profile_a"], minute=1)
        review(service, positive, "present", "positive")
        review(service, negative, "absent", "negative")
        expected = {"reviewed_earthquake": positive.window_id, "reviewed_noise": negative.window_id}
    for status, wid in expected.items():
        response = client.get("/api/v1/event-windows", params={"review_status": status})
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["items"][0]["id"] == wid
