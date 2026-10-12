"""Exercise worker error isolation across actual FastAPI/service boundaries."""

import httpx
from das_sentinel.api import create_app
from das_sentinel.config import Settings
from das_sentinel.persistence.models import ProcessingProfile
from das_sentinel.persistence.window_models import WindowAnalysis
from das_sentinel.services.event_windows import WindowService, enqueue_windows
from das_sentinel.worker.window_main import HttpxWindowClient, WindowWorker
from fastapi.testclient import TestClient

from tests import test_event_windows as original

context = original.context


def test_profile_change_fails_one_api_task_and_worker_processes_next(context):
    session, site, profile, service = context
    analysis = enqueue_windows(session, site.id, profile.id, [original.plan()])[0]
    first_id, site_id = analysis.id, site.id
    app_data = service.artifact_root.parent
    settings = Settings(
        data_root=site.data_path,
        app_data_dir=app_data,
        database_url=str(session.bind.url),
        session_secret="s" * 40,
        worker_token="w" * 40,
        secure_cookies=False,
    )
    session.commit()
    app = create_app(settings)
    sessions = app.state.session_factory
    replacement_id = None

    def analyzer(paths, start, config, output, **kwargs):
        nonlocal replacement_id
        with sessions.begin() as local:
            current = local.get(WindowAnalysis, output.parent.name)
            original.make_result(WindowService(local, service.artifact_root), current)
            if replacement_id is None:
                old = local.get(ProcessingProfile, current.profile_id)
                old.is_current = False
                local.flush()
                replacement = ProcessingProfile(
                    site_id=site_id, version=2,
                    payload_json=old.payload_json, is_current=True,
                )
                local.add(replacement)
                local.flush()
                replacement_id = replacement.id

    with TestClient(app) as api:
        replies = []

        def bridge(request):
            response = api.post(
                request.url.path,
                content=request.content,
                headers=dict(request.headers),
            )
            if request.url.path.endswith("/result"):
                replies.append((response.status_code, response.json()))
            return httpx.Response(response.status_code, content=response.content)

        client = HttpxWindowClient(
            "http://isolated.invalid", settings.worker_token,
            transport=httpx.MockTransport(bridge),
        )
        try:
            worker = WindowWorker(settings, client, "worker-api-test", analyzer=analyzer)
            assert worker.run_once().status == "failed"
            assert replies == [(409, {"detail": "source_profile_changed"})]
            with sessions.begin() as local:
                failed = local.get(WindowAnalysis, first_id)
                assert failed.status == "failed"
                assert failed.error_code == "result_rejected"
                next_analysis = enqueue_windows(
                    local, site_id, replacement_id, [original.plan()]
                )[0]
                next_id = next_analysis.id
            assert worker.run_once().status == "succeeded"
            with sessions() as local:
                assert local.get(WindowAnalysis, next_id).status == "succeeded"
        finally:
            client.close()


def test_empty_window_worker_skip_api_transaction_and_lost_response_replay(context):
    from das_sentinel.persistence.window_models import EventWindow, WindowSkip
    session, site, profile, service = context
    planned = original.plan() | {"end_utc": "2023-09-27T00:00:30+00:00"}
    a = enqueue_windows(session, site.id, profile.id, [planned])[0]
    aid, wid = a.id, a.window_id
    settings = Settings(data_root=site.data_path, app_data_dir=service.artifact_root.parent,
        database_url=str(session.bind.url), session_secret="s"*40,
        worker_token="w"*40, secure_cookies=False)
    session.commit()
    app = create_app(settings)
    calls = []
    dropped = False

    def analyzer(paths, start, config, output, **kwargs):
        assert kwargs["skip_empty"] is True
        assert not output.exists()
        return {"skipped": True, "reason": "no_qualified_regions"}

    with TestClient(app) as api:
        assert api.post(f"/api/v1/internal/window-jobs/{aid}/skip", json={}).status_code in (401, 403)

        def bridge(request):
            nonlocal dropped
            calls.append(request.url.path.rsplit("/", 1)[-1])
            response = api.post(request.url.path, content=request.content, headers=dict(request.headers))
            if request.url.path.endswith("/skip") and not dropped:
                assert response.status_code == 200
                dropped = True
                raise httpx.ReadTimeout("response lost after successful deletion", request=request)
            return httpx.Response(response.status_code, content=response.content)

        client = HttpxWindowClient("http://isolated.invalid", settings.worker_token,
                                   transport=httpx.MockTransport(bridge))
        try:
            result = WindowWorker(settings, client, "skip-worker", analyzer=analyzer).run_once()
            assert result.status == "skipped"
            assert calls.count("skip") == 2
            assert "result" not in calls and "fail" not in calls
            assert not (service.artifact_root / aid).exists()
            with app.state.session_factory.begin() as local:
                assert local.get(WindowAnalysis, aid) is None
                assert local.get(EventWindow, wid) is None
                assert local.get(WindowSkip, aid).reason == "no_qualified_regions"
                assert enqueue_windows(local, site.id, profile.id, [planned]) == []
        finally:
            client.close()
