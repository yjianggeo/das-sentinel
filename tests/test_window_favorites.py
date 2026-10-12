import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from das_sentinel.persistence.models import Admin, ProcessingProfile, Site
from das_sentinel.persistence.window_models import EventWindow, WindowReview
from das_sentinel.services.event_windows import (
    WindowError,
    WindowService,
    enqueue_windows,
)
from sqlalchemy import func, select

from tests import test_event_windows as windows

context = windows.context
pytest_plugins = ("tests.test_events_api",)


def ready(session, site_id, profile_id, *, seconds=30, minute=0, priority=0):
    plan = windows.plan()
    start = datetime.fromisoformat(plan["start_utc"]) + timedelta(minutes=minute)
    plan.update(start_utc=start.isoformat(), end_utc=(start + timedelta(seconds=seconds)).isoformat())
    analysis = enqueue_windows(session, site_id, profile_id, [plan], priority=priority)[0]
    analysis.status = "succeeded"
    analysis.input_sha256 = "a" * 64
    analysis.result_json = json.dumps({"features": {}, "artifacts": [
        {"kind": "waterfall", "path": "waterfall.png", "media_type": "image/png"},
    ]})
    session.flush()
    return analysis


def test_favorite_persists_without_changing_review_or_analysis(context):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    analysis = ready(session, site.id, profile.id, priority=123)
    service.review(analysis.window_id, analysis_id=analysis.id, input_sha256=analysis.input_sha256,
                   labels={"earthquake": "present"}, note="keep this annotation", request_id="review", admin_id=1)
    original = (analysis.result_json, analysis.input_sha256,
                [(r.id, r.labels_json, r.note) for r in session.scalars(select(WindowReview))])

    first = service.set_favorite(analysis.window_id, True, admin_id=1)
    assert first["favorite"] is True and first["created_at"] is not None
    assert service.set_favorite(analysis.window_id, True, admin_id=1) == first
    session.commit()
    session.expire_all()
    assert service.detail(analysis.window_id, admin_id=1)["favorite"] is True
    assert service.list(review_status="all", admin_id=1)["items"][0]["favorite"] is True
    assert service.list(review_status="all")["items"][0]["favorite"] is False
    assert (analysis.result_json, analysis.input_sha256,
            [(r.id, r.labels_json, r.note) for r in session.scalars(select(WindowReview))]) == original
    assert session.get(EventWindow, analysis.window_id).priority == 123
    assert service.set_favorite(analysis.window_id, False, admin_id=1)["favorite"] is False
    assert service.set_favorite(analysis.window_id, False, admin_id=1)["created_at"] is None


@pytest.mark.parametrize("status,seconds,artifacts", [
    ("queued", 30, True), ("running", 30, True), ("failed", 30, True),
    ("cancelled", 30, True), ("succeeded", 60, True), ("succeeded", 30, False),
])
def test_new_favorite_requires_a_completed_thirty_second_waterfall(context, status, seconds, artifacts):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    analysis = ready(session, site.id, profile.id, seconds=seconds)
    analysis.status = status
    if not artifacts:
        analysis.result_json = '{"artifacts": []}'
    session.flush()
    with pytest.raises(WindowError, match="favorite_requires_completed_window"):
        service.set_favorite(analysis.window_id, True, admin_id=1)


def test_api_favorite_requires_session_and_csrf_and_rejects_supplied_identity(event_api):
    client, _settings, sessions, csrf, ids = event_api
    with sessions.begin() as session:
        analysis = ready(session, ids["site_a"], ids["profile_a"])
        wid = analysis.window_id
    path = f"/api/v1/event-windows/{wid}/favorite"
    assert client.put(path, json={"favorite": True}).status_code == 403
    headers = {"X-CSRF-Token": csrf}
    assert client.put("/api/v1/event-windows/missing/favorite", json={"favorite": True}, headers=headers).status_code == 404
    assert client.put(path, json={"favorite": True, "admin_id": 1}, headers=headers).status_code == 422
    assert client.put(path, json={"favorite": "false"}, headers=headers).status_code == 422
    response = client.put(path, json={"favorite": True}, headers=headers)
    assert response.status_code == 200
    assert response.json()["window_id"] == wid and response.json()["favorite"] is True
    assert client.get(f"/api/v1/event-windows/{wid}").json()["favorite"] is True
    page = client.get("/api/v1/event-windows", params={"favorites_only": True, "review_status": "all"}).json()
    assert page["total"] == 1 and page["items"][0]["favorite"] is True
    client.cookies.clear()
    assert client.put(path, json={"favorite": False}, headers=headers).status_code == 401


@pytest.mark.parametrize("sort", ["default", "borehole_coverage"])
def test_favorite_filter_applies_before_count_and_pagination(context, sort):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    ready(session, site.id, profile.id, priority=999)
    first = ready(session, site.id, profile.id, minute=1, priority=20)
    second = ready(session, site.id, profile.id, minute=2, priority=10)
    for analysis in (first, second):
        service.set_favorite(analysis.window_id, True, admin_id=1)
    service.review(first.window_id, analysis_id=first.id, input_sha256=first.input_sha256,
                   labels={"earthquake": "present"}, note="reviewed favorite", request_id="review", admin_id=1)
    page = service.list(favorites_only=True, review_status="all", admin_id=1, limit=1, offset=1, sort=sort)
    assert page["total"] == 2 and [v["id"] for v in page["items"]] == [second.window_id]
    assert service.list(favorites_only=True, review_status="reviewed", admin_id=1, sort=sort)["items"][0]["id"] == first.window_id
    assert service.list(favorites_only=True, review_status="all", admin_id=999, sort=sort)["total"] == 0
    assert service.detail(first.window_id, admin_id=999)["favorite"] is False


def test_existing_favorite_survives_reanalysis_and_can_be_removed(context):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    analysis = ready(session, site.id, profile.id)
    first = service.set_favorite(analysis.window_id, True, admin_id=1)
    new_plan = json.loads(analysis.plan_json) | {"algorithm_version": "future"}
    replacement = enqueue_windows(session, site.id, profile.id, [new_plan])[0]
    assert replacement.status == "queued"
    assert service.detail(analysis.window_id, admin_id=1)["favorite"] is True
    assert service.set_favorite(analysis.window_id, True, admin_id=1) == first
    assert service.set_favorite(analysis.window_id, False, admin_id=1)["favorite"] is False


def test_favorite_protects_window_from_empty_skip(context):
    session, site, profile, service = context
    session.add(Admin(id=1, password_hash="test"))
    analysis = ready(session, site.id, profile.id)
    service.set_favorite(analysis.window_id, True, admin_id=1)
    # A replay without retained results must still respect the bookmark.
    analysis.status = "queued"
    analysis.result_json = None
    analysis.input_sha256 = None
    session.flush()
    assert service.claim("worker", supports_empty_skip=True)["skip_empty"] is False


def test_favorites_do_not_create_training_labels_or_change_existing_export(context):
    from das_sentinel.window_training import export_rows

    from tests.test_window_training_export import reviewed_window

    session, site, service, _existing = reviewed_window(context)
    profile = context[2]
    analysis = ready(session, site.id, profile.id)
    before = export_rows(session, service.artifact_root, site.id, window_seconds=60)
    assert len(before["rows"]) == 1
    service.set_favorite(analysis.window_id, True, admin_id=1)
    assert session.scalar(select(func.count()).select_from(WindowReview)) == 1
    assert export_rows(session, service.artifact_root, site.id, window_seconds=30)["rows"] == []
    assert export_rows(session, service.artifact_root, site.id, window_seconds=60) == before


def test_upgrade_from_previous_schema_preserves_data_and_cascades_removed_window(tmp_path):
    from alembic import command
    from alembic.config import Config
    from das_sentinel.persistence import db
    from sqlalchemy import inspect, text

    url = f"sqlite+pysqlite:///{tmp_path / 'upgrade.sqlite3'}"
    config = Config()
    config.set_main_option("script_location", str(Path(db.__file__).parent / "migrations"))
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "0007_model_scans")
    engine = db.create_local_engine(url)
    sessions = db.session_factory(engine)
    assert "window_favorites" not in inspect(engine).get_table_names()
    with sessions.begin() as session:
        session.add(Admin(id=1, password_hash="test"))
        site = Site(name="kept", data_path=str(tmp_path), timezone="Asia/Shanghai", cable_type="downhole")
        session.add(site)
        session.flush()
        profile = ProcessingProfile(site_id=site.id, version=1, payload_json="{}", is_current=True)
        session.add(profile)
        session.flush()
        analysis = ready(session, site.id, profile.id)
        aid, wid = analysis.id, analysis.window_id
        session.add(WindowReview(window_id=wid, analysis_id=aid, input_sha256=analysis.input_sha256,
                                 labels_json='{"earthquake":"present"}', note="must survive migration",
                                 request_id="kept", admin_id=1))
    with engine.connect() as connection:
        before = {table: connection.execute(text(f"SELECT * FROM {table}")).all()
                  for table in ("event_windows", "window_analyses", "window_reviews")}
    db.upgrade_database(url)
    with engine.connect() as connection:
        assert all(connection.execute(text(f"SELECT * FROM {table}")).all() == rows
                   for table, rows in before.items())
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0012_model_scan_coverage_index"
    with sessions.begin() as session:
        service = WindowService(session, tmp_path / "window-artifacts")
        service.set_favorite(wid, True, admin_id=1)
        session.execute(text("DELETE FROM window_reviews WHERE window_id=:wid"), {"wid": wid})
        session.execute(text("DELETE FROM window_analyses WHERE window_id=:wid"), {"wid": wid})
        session.execute(text("DELETE FROM event_windows WHERE id=:wid"), {"wid": wid})
        assert session.execute(text("SELECT count(*) FROM window_favorites")).scalar() == 0
        assert session.execute(text("PRAGMA foreign_key_check")).all() == []
    engine.dispose()
