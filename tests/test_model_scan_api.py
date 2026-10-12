import importlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from tests.test_events_api import WORKER_TOKEN

pytest_plugins = ("tests.test_events_api",)


def test_scan_routes_use_existing_auth_and_csrf(event_api):
    client, _settings, _sessions, _csrf, ids = event_api
    response = client.get("/api/v1/model-scans")
    assert response.status_code == 200, "persistent scan API missing"
    assert response.json()["total"] == 0
    assert client.post(f"/api/v1/sites/{ids['site_a']}/model-scans", json={}).status_code == 403
    assert client.post("/api/v1/internal/model-scans/claim", json={"worker_id": "reader"}).status_code == 401
    assert client.post("/api/v1/internal/model-scans/claim", headers={"X-Worker-Token": WORKER_TOKEN},
                       json={"worker_id": "reader"}).status_code == 204
    client.cookies.clear()
    assert client.get("/api/v1/model-scans").status_code == 401


def test_create_freezes_active_model_and_range_then_pause(event_api, monkeypatch):
    assert importlib.util.find_spec("das_sentinel.api.routes.model_scans"), "model scan API missing"
    from das_sentinel.api.routes import model_scans
    client, settings, _sessions, csrf, ids = event_api
    model = {"id": str(uuid4()), "name": "frozen", "weights_sha256": "a" * 64,
             "metadata_sha256": "b" * 64, "threshold": .04}
    monkeypatch.setattr(model_scans, "read_active", lambda *args: model)
    (settings.data_root / "jjk2023").mkdir()
    (settings.app_data_dir / "model-scan-exclusions.json").write_text('{"event_windows": []}')
    body = {"request_id": "scan-one", "directories": ["jjk2023"], "include_rule_skips": True,
            "expected_model_id": model["id"]}
    headers = {"X-CSRF-Token": csrf}
    response = client.post(f"/api/v1/sites/{ids['site_a']}/model-scans", headers=headers, json=body)
    assert response.status_code == 201, response.text
    batch = response.json()
    assert batch["model"] == model and batch["remaining_seconds"] is None
    again = client.post(f"/api/v1/sites/{ids['site_a']}/model-scans", headers=headers, json=body)
    assert again.json()["id"] == batch["id"]
    assert client.post(f"/api/v1/model-scans/{batch['id']}/pause", headers=headers).json()["status"] == "paused"
    assert client.post("/api/v1/internal/model-scans/claim", headers={"X-Worker-Token": WORKER_TOKEN},
                       json={"worker_id": "reader"}).status_code == 204
    for directory in ("../escape", "@Recycle", "eq", str(settings.data_root)):
        bad = client.post(f"/api/v1/sites/{ids['site_a']}/model-scans", headers=headers,
                          json=body | {"request_id": str(uuid4()), "directories": [directory]})
        assert bad.status_code == 422


@pytest.mark.parametrize("sort", ["default", "borehole_coverage"])
def test_model_recommendations_ignore_rule_gate_and_old_priority(event_api, sort):
    client, _settings, sessions, _csrf, ids = event_api
    from das_sentinel.persistence.model_scan_models import (
        ModelScan,
        ModelScanReceipt,
        ModelScanUnit,
    )
    from das_sentinel.persistence.models import ProcessingProfile
    from das_sentinel.services.event_windows import enqueue_windows
    from sqlalchemy import select
    model = {"id": "frozen-model", "name": "Frozen 300 m", "threshold": .04}
    with sessions.begin() as session:
        profile = session.scalar(select(ProcessingProfile).where(ProcessingProfile.site_id == ids["site_a"]))
        scan = ModelScan(site_id=ids["site_a"], profile_id=profile.id, request_id="seed",
                         config_json=json.dumps({"model": model}), config_sha256="a" * 64)
        session.add(scan)
        session.flush()
        unit = ModelScanUnit(scan_id=scan.id, kind="infer", unit_key="seed", payload_json="{}", status="succeeded")
        session.add(unit)
        session.flush()
        expected = []
        for index, score in enumerate((.1, .9)):
            start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=30 * index)
            analysis = enqueue_windows(session, ids["site_a"], profile.id, [{"start_utc": start.isoformat(),
                "end_utc": (start + timedelta(seconds=30)).isoformat(), "sources": [], "coverage": "complete"}],
                priority=100 if index == 0 else 0)[0]
            analysis.status, analysis.input_sha256 = "succeeded", str(index) * 64
            analysis.result_json = '{"candidate":false,"score":0,"features":{}}'
            session.add(ModelScanReceipt(scan_id=scan.id, infer_unit_id=unit.id, start_utc=start,
                inference_key=str(index) * 64, status="ready", score=score, final_score=score,
                result_json="{}", result_sha256="b" * 64, window_id=analysis.window_id,
                analysis_id=analysis.id, input_sha256=analysis.input_sha256))
            expected.insert(0, analysis.window_id)
    response = client.get("/api/v1/event-windows", params={"model_recommended": True, "limit": 1, "sort": sort})
    assert response.status_code == 200
    assert response.json()["total"] == 2
    assert response.json()["items"][0]["id"] == expected[0]
    item = response.json()["items"][0]
    assert item["candidate"] is False and item["model_recommendation"]["score"] == .9
    assert item["model_recommendation"]["model_id"] == "frozen-model"
    assert client.get("/api/v1/event-windows?model_recommended=true&candidate_only=true").status_code == 422


def test_real_reader_to_review_model_positive_without_rule_regions(event_api, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import httpx
    import numpy as np
    from das_sentinel.window_inference import file_sha
    from das_sentinel.worker.model_scan_main import (
        HttpxModelScanClient,
        ModelScanWorker,
    )

    from tests.test_whole_window import BASE
    from tests.test_whole_window import write as write_hdf

    client, settings, _sessions, csrf, ids = event_api
    directory = settings.data_root / "jjk2023"
    directory.mkdir()
    write_hdf(directory / "source.h5")
    (settings.app_data_dir / "model-scan-exclusions.json").write_text('{"event_windows": []}')
    model_id = str(uuid4())
    folder = settings.app_data_dir / "window-models" / ids["site_a"] / "deployments" / model_id
    folder.mkdir(parents=True)
    (folder / "weights.pt").write_bytes(b"API test: prediction is provided by a deterministic predictor")
    distance = np.arange(10, dtype="<f8") * 100
    metadata = {"schema_version": 1, "task": "earthquake_including_blast_vs_noise", "candidate": "residual_waveform",
        "window_seconds": 30, "sample_rate_hz": 100, "distance_start_m": 300,
        "preprocess": "whole-window-signed-waveform-v1-30s", "unit": "strain_rate",
        "weights_sha256": file_sha(folder / "weights.pt"), "metrics": {"threshold": .04},
        "training_grid": {"version": "native-channel-union-v1", "start_locus": 0, "spacing_m": 100.,
                          "channels": 10, "grid_sha256": hashlib.sha256(distance.tobytes()).hexdigest()}}
    (folder / "model.json").write_text(json.dumps(metadata))
    model = {"id": model_id, "site_id": ids["site_a"], "name": "Frozen test model", "threshold": .04,
             "metadata_sha256": file_sha(folder / "model.json"), "weights_sha256": metadata["weights_sha256"]}
    (folder / "deployment.json").write_text(json.dumps(model))
    (folder.parent.parent / "active.json").write_text(json.dumps({"id": model_id}))
    monkeypatch.setattr("das_sentinel.services.model_scan_discovery.time.sleep", lambda seconds: None)
    monkeypatch.setattr("das_sentinel.whole_window.screen_block", lambda *args: SimpleNamespace(candidates=[], events=[]))
    response = client.post(f"/api/v1/sites/{ids['site_a']}/model-scans", headers={"X-CSRF-Token": csrf}, json={
        "request_id": "integration", "directories": ["jjk2023"], "expected_model_id": model_id,
        "time_ranges": [{"start_utc": BASE.isoformat(), "end_utc": (BASE + timedelta(seconds=30)).isoformat()}]})
    assert response.status_code == 201, response.text
    scan_id = response.json()["id"]
    def handler(request):
        response = client.request(request.method, request.url.path, content=request.content,
                                  headers={"X-Worker-Token": WORKER_TOKEN, "Content-Type": "application/json"})
        return httpx.Response(response.status_code, content=response.content)
    worker_client = HttpxModelScanClient("http://test", WORKER_TOKEN, transport=httpx.MockTransport(handler))
    predictor = SimpleNamespace(predict=lambda arrays: {"status": "succeeded", "score": .8, "model_id": model_id})
    worker = ModelScanWorker(settings, worker_client, "integration-reader", predictor=predictor)
    outcomes = []
    for _ in range(12):
        outcome = worker.run_once()
        outcomes.append(outcome.status)
        if outcome.status == "idle":
            break
    assert "failed" not in outcomes, outcomes
    progress = client.get(f"/api/v1/model-scans/{scan_id}").json()
    # The shared fixture also registers an intentionally invalid HDF source.
    assert progress["status"] == "completed_with_errors", progress
    assert progress["counts"]["source_errors"] > 0
    assert progress["counts"]["ready"] == 1
    queue = client.get(f"/api/v1/event-windows?model_recommended=true&model_scan_id={scan_id}").json()
    assert queue["total"] == 1
    event = queue["items"][0]
    assert event["candidate"] is False and event["latest_review"] is None
    detail = client.get(f"/api/v1/event-windows/{event['id']}").json()
    assert detail["model_recommendation"]["model_id"] == model_id
    for artifact in detail["artifacts"]:
        assert client.get(artifact["url"]).status_code == 200
    worker.close()
    worker_client.close()
