from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.model_scan_models import ModelScan
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
)
from fastapi.testclient import TestClient


@pytest.fixture
def dashboard_api(tmp_path, monkeypatch):
    data, app_data = tmp_path / 'data', tmp_path / 'app-data'
    data.mkdir()
    app_data.mkdir()
    settings = Settings(data_root=data, app_data_dir=app_data,
        database_url=f'sqlite+pysqlite:///{app_data / "dashboard.sqlite3"}',
        session_secret='s' * 32, worker_token='w' * 40, secure_cookies=False)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        admin = repositories.admins.add(password_hash=PasswordHasher().hash('correct horse'))
        site = repositories.sites.add(name='current', data_path='/data/current', timezone='Asia/Shanghai', cable_type='urban')
        profile = repositories.profiles.add(site_id=site.id, version=1,
            payload=ProcessingProfile().model_dump(mode='json'), is_current=True)
        identities = (site.id, profile.id, admin.id)
    monkeypatch.setattr('das_sentinel.api.app.check_database_startup', lambda _: None)
    with TestClient(create_app(settings)) as client:
        client.post('/api/v1/session', json={'password': 'correct horse'})
        yield client, sessions, identities
    engine.dispose()


def add_window(session, identities, index, status='succeeded', seconds=30):
    site_id, profile_id, _ = identities
    start = datetime(2026, 8, 6, tzinfo=UTC) + timedelta(minutes=index)
    window = EventWindow(id=str(uuid4()), site_id=site_id, start_utc=start,
        end_utc=start + timedelta(seconds=seconds), window_seconds=seconds)
    session.add(window)
    session.flush()
    analysis = WindowAnalysis(window_id=window.id, profile_id=profile_id, status=status,
        plan_json='{}', plan_sha256='c' * 64, algorithm_version='whole-window-v2',
        input_sha256='a' * 64, result_json='{}')
    session.add(analysis)
    session.flush()
    window.latest_analysis_id = analysis.id
    return window, analysis


def add_review(session, identities, window, analysis, label, *, created=0, sha=None, labels=None, review_id=None):
    review = WindowReview(id=review_id or str(uuid4()), window_id=window.id,
        analysis_id=analysis.id, input_sha256=sha or analysis.input_sha256,
        labels_json=json.dumps(labels or {'earthquake': label}), admin_id=identities[2],
        request_id=str(uuid4()), created_at=datetime(2026, 8, 6, tzinfo=UTC) + timedelta(seconds=created))
    session.add(review)
    return review


def test_dashboard_requires_an_authenticated_session(dashboard_api):
    client, _, _ = dashboard_api
    client.cookies.clear()
    assert client.get('/api/v1/dashboard').status_code == 401


def test_dashboard_counts_current_30s_analyses_and_scan_batches_not_legacy_jobs(dashboard_api):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        repositories = Repositories(session)
        source = repositories.source_files.add(site_id=identities[0], path='/data/current/old.h5',
            fingerprint='f' * 64, size_bytes=1, mtime_ns=1)
        repositories.jobs.add(source_file_id=source.id, processing_profile_id=identities[1],
            pipeline_version='old', status='running')
        for index, state in enumerate(('queued', 'running', 'succeeded', 'failed')):
            add_window(session, identities, index, state)
        add_window(session, identities, 10, 'running', seconds=60)
        old_window, _ = add_window(session, identities, 11, 'succeeded')
        replacement = WindowAnalysis(window_id=old_window.id, profile_id=identities[1],
            status='queued', plan_json='{}', plan_sha256='d' * 64, algorithm_version='whole-window-v2')
        session.add(replacement)
        session.flush()
        old_window.latest_analysis_id = replacement.id
        for index, state in enumerate(('queued', 'queued', 'running', 'paused', 'cancelled', 'succeeded')):
            session.add(ModelScan(site_id=identities[0], profile_id=identities[1], request_id=f'scan-{index}',
                config_json='{}', config_sha256='e' * 64, status=state))
    response = client.get('/api/v1/dashboard')
    assert response.status_code == 200
    assert response.json() == {
        'window_status_counts': {'queued': 2, 'running': 1, 'succeeded': 1, 'failed': 1},
        'model_scan_status_counts': {'queued': 2, 'running': 1, 'paused': 1, 'cancelled': 1, 'succeeded': 1},
        'review_counts': {'earthquake': 0, 'noise': 0, 'pending': 1},
    }


def test_dashboard_latest_known_reviews_must_match_latest_succeeded_input(dashboard_api):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        positive, pa = add_window(session, identities, 0)
        add_review(session, identities, positive, pa, 'absent')
        add_review(session, identities, positive, pa, 'present', created=1)
        negative, na = add_window(session, identities, 1)
        add_review(session, identities, negative, na, 'present', review_id='review-a')
        add_review(session, identities, negative, na, 'absent', review_id='review-z')
        unknown, ua = add_window(session, identities, 2)
        add_review(session, identities, unknown, ua, 'present')
        add_review(session, identities, unknown, ua, 'unknown', created=1)
        stale, sa = add_window(session, identities, 3)
        add_review(session, identities, stale, sa, 'present')
        add_review(session, identities, stale, sa, 'absent', created=1, sha='b' * 64)
        changed, ca = add_window(session, identities, 4)
        add_review(session, identities, changed, ca, 'present')
        replacement = WindowAnalysis(window_id=changed.id, profile_id=identities[1], status='succeeded',
            plan_json='{}', plan_sha256='d' * 64, algorithm_version='whole-window-v2', input_sha256='a' * 64)
        session.add(replacement)
        session.flush()
        changed.latest_analysis_id = replacement.id
        failed, fa = add_window(session, identities, 5, 'failed')
        add_review(session, identities, failed, fa, 'present')
        old, oa = add_window(session, identities, 6, seconds=60)
        add_review(session, identities, old, oa, 'present')
    response = client.get('/api/v1/dashboard')
    assert response.status_code == 200
    assert response.json()['review_counts'] == {'earthquake': 1, 'noise': 1, 'pending': 3}


def test_dashboard_projects_inherited_binary_classes_without_assuming_unknown_absent(dashboard_api):
    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        for index, blast in enumerate(('present', 'absent', 'unknown')):
            window, analysis = add_window(session, identities, index)
            add_review(session, identities, window, analysis, 'absent', labels={
                'earthquake': 'absent', 'blast': blast, 'traffic': 'present'})
    response = client.get('/api/v1/dashboard')
    assert response.status_code == 200
    assert response.json()['review_counts'] == {'earthquake': 1, 'noise': 1, 'pending': 1}

def test_dashboard_uses_covering_analysis_projection_instead_of_wide_record_reads(dashboard_api):
    from sqlalchemy import event

    client, sessions, identities = dashboard_api
    with sessions.begin() as session:
        window, analysis = add_window(session, identities, 0)
        add_review(session, identities, window, analysis, 'present')
    engine = client.app.state.session_factory.kw['bind']
    queries = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        if 'window_analyses' in statement and 'count(' in statement.lower():
            queries.append((statement, parameters))

    event.listen(engine, 'before_cursor_execute', capture)
    try:
        response = client.get('/api/v1/dashboard')
        assert response.status_code == 200
    finally:
        event.remove(engine, 'before_cursor_execute', capture)
    assert len(queries) == 2
    with engine.connect() as connection:
        for statement, parameters in queries:
            plan = connection.exec_driver_sql('EXPLAIN QUERY PLAN ' + statement, parameters).all()
            assert any('SCAN window_analyses USING COVERING INDEX ix_window_analysis_list' in row[3]
                       for row in plan), plan
