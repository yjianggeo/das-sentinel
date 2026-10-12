from unittest.mock import AsyncMock, Mock

from das_sentinel.api.routes import event_windows

pytest_plugins = ('tests.test_events_api',)


def test_prediction_routes_require_auth_and_retry_requires_csrf(event_api):
    client, _, _, csrf, ids = event_api
    service = Mock()
    service.active.return_value = {'id': 'model', 'name': 'previous 300 m'}
    service.prediction.return_value = {'status': 'pending'}
    client.app.state.window_inference_service = service
    url = '/api/v1/event-windows/window/prediction'
    binding = {'analysis_id': 'analysis', 'input_sha256': 'a' * 64}
    assert client.get('/api/v1/window-models/active', params={'site_id': ids['site_a']}).json() == service.active.return_value
    assert client.get(url, params=binding).json() == {'status': 'pending'}
    service.prediction.assert_called_once_with('window', **binding)
    assert client.post(url, json=binding).status_code == 403
    assert client.post(url, json=binding, headers={'X-CSRF-Token': csrf}).status_code == 200
    service.prediction.assert_called_with('window', **binding, retry=True)
    client.cookies.clear()
    assert client.get(url).status_code == 401
    assert client.get('/api/v1/window-models/active', params={'site_id': ids['site_a']}).status_code == 401


def test_accepted_scan_triggers_inference_but_inference_failure_does_not_fail_scan(event_api, monkeypatch):
    client, settings, _, _, _ = event_api
    service = Mock()
    client.app.state.window_inference_service = service
    completed = {'window_id': 'window', 'id': 'analysis', 'status': 'succeeded', 'input_sha256': 'a' * 64}

    async def accepted(*args, **kwargs):
        return completed

    monkeypatch.setattr(event_windows, 'write', accepted)
    coverage = AsyncMock()
    monkeypatch.setattr(event_windows, 'schedule_spatial_coverage', coverage)
    service.prediction.side_effect = RuntimeError('inference unavailable')
    response = client.post('/api/v1/internal/window-jobs/analysis/result',
        json={'worker_id': 'worker', 'attempt_count': 1}, headers={'X-Worker-Token': settings.worker_token})
    assert response.status_code == 200 and response.json() == completed
    coverage.assert_awaited_once()
    assert coverage.call_args.args[1:] == ('window', 'analysis', 'a' * 64)
    service.prediction.assert_called_once_with('window', analysis_id='analysis')
