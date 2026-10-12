from __future__ import annotations

import pytest

from tests.test_low_frequency_scans import completed, payload

pytest_plugins = ("tests.test_events_api",)


def test_low_frequency_routes_require_session_and_csrf(event_api):
    client, _, _, _, _ = event_api
    assert client.get("/api/v1/low-frequency/scans").status_code == 200
    assert client.post("/api/v1/low-frequency/scans", json={}).status_code == 403
    client.cookies.clear()
    assert client.get("/api/v1/low-frequency/scans").status_code == 401


def test_low_frequency_request_rejects_short_window_and_naive_time():
    from das_sentinel.api.routes.low_frequency import ScanRequest
    from pydantic import ValidationError

    from tests.test_low_frequency_scans import payload

    request = payload()
    assert ScanRequest.model_validate(request).distance_start_m == 499
    with pytest.raises(ValidationError):
        ScanRequest.model_validate({**request, "end_utc": "2026-06-16T09:05:30Z"})
    with pytest.raises(ValidationError):
        ScanRequest.model_validate({**request, "start_utc": "2026-06-16T09:05:00"})
    with pytest.raises(ValidationError):
        ScanRequest.model_validate({**request, "distance_stop_m": 300})


def test_create_scan_review_and_read_persisted_result(event_api, monkeypatch):
    from uuid import uuid4

    client, settings, _, csrf, ids = event_api
    (settings.data_root / "day").mkdir()
    service = client.app.state.low_frequency_service
    monkeypatch.setattr(
        service,
        "_analyze",
        lambda *args: {"candidates": [{"id": "one", "score": 0.8}], "plots": {}},
    )
    body = {**payload(), "site_id": ids["site_a"]}
    response = client.post(
        "/api/v1/low-frequency/scans", json=body, headers={"x-csrf-token": csrf}
    )
    assert response.status_code == 202
    scan_id = response.json()["id"]
    assert completed(service, scan_id)["status"] == "succeeded"
    url = f"/api/v1/low-frequency/scans/{scan_id}"
    assert client.get(url).json()["result"]["candidates"][0]["latest_review"] is None
    review = {
        "label": "earthquake",
        "notes": "same-phase long wave",
        "request_id": str(uuid4()),
    }
    assert client.post(url + "/candidates/one/reviews", json=review).status_code == 403
    first = client.post(
        url + "/candidates/one/reviews", json=review, headers={"x-csrf-token": csrf}
    )
    second = client.post(
        url + "/candidates/one/reviews", json=review, headers={"x-csrf-token": csrf}
    )
    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    assert (
        client.get(url).json()["result"]["candidates"][0]["latest_review"]["label"]
        == "earthquake"
    )
    assert client.get(url + "/plots/not-a-plot").status_code == 404
    assert client.get("/api/v1/low-frequency/scans/not-a-uuid").status_code == 404


def test_scan_rejects_directory_escape_and_missing_site(event_api):
    client, _, _, csrf, ids = event_api
    headers = {"x-csrf-token": csrf}
    body = {**payload(), "site_id": ids["site_a"], "relative_path": "../outside"}
    assert (
        client.post(
            "/api/v1/low-frequency/scans", json=body, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/low-frequency/scans",
            json={**body, "site_id": "missing"},
            headers=headers,
        ).status_code
        == 404
    )
    assert client.get("/api/v1/low-frequency/scans").json() == []
