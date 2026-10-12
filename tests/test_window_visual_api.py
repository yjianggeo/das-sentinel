import hashlib
import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from das_sentinel.services.event_windows import WindowError, enqueue_windows
from das_sentinel.whole_window_plots import create_visualization

from tests import test_event_windows as windows

context = windows.context
pytest_plugins = ("tests.test_events_api",)


def install_visualization(service, analysis, *, regions_baked=False):
    from das_sentinel.config import ProcessingProfile as ProfileConfig
    from das_sentinel.whole_window import compute_input_sha256

    result = windows.make_result(service, analysis)
    result["profile"] = ProfileConfig.model_validate(
        json.loads(analysis.plan_json)["profile_payload"]
    ).model_dump(mode="json")
    result["input_sha256"] = compute_input_sha256(result)
    folder = service.artifact_root / analysis.id / "attempt-1"
    with np.load(folder / "waveform.npz") as arrays:
        raw = arrays["waveform"].T
        distance = arrays["distance_m"]
        recognition = arrays["recognition_channel_mask"]
    evidence = [{"start_sample": 400, "end_sample": 500, "peak_sample": 450,
                     "channel_start": 0, "channel_stop": 2, "detector": "stalta:short",
                     "score": 3.0, "qualified": True}]
    (folder / "evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    visual, files = create_visualization(raw, raw.astype("f4") * 0.5, distance,
                                         datetime.fromisoformat(result["start_utc"]), recognition,
                                         result["unit"], evidence, merge_gap_seconds=result["profile"]["merge_gap_seconds"])
    result.update(visual)
    if regions_baked is None:
        result.pop("waterfall_regions_baked", None)
    else:
        result["waterfall_regions_baked"] = regions_baked
    if regions_baked is not False:
        from das_sentinel.whole_window_plots import render_waterfall_preview

        for name, samples, filtered in (("waterfall_raw_v2.png", raw, False),
                                        ("waterfall_filtered_v2.png", raw.astype("f4") * 0.5, True)):
            files[name] = render_waterfall_preview(samples, distance,
                datetime.fromisoformat(result["start_utc"]), recognition, result["unit"],
                filtered=filtered, regions=visual["detection_regions"])
    result.update(candidate=True, evidence_count=1)
    for name, payload in files.items():
        (folder / name).write_bytes(payload)
    (folder / "metadata_visual_v2.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "artifacts"}), encoding="utf-8")
    result["artifacts"] = [{"kind": kind, "path": name, "media_type": media,
                                "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest()}
                           for kind, name, media in (
                               ("waveform", "waveform.npz", "application/octet-stream"),
                               ("evidence", "evidence.json", "application/json"),
                               ("metadata", "metadata_visual_v2.json", "application/json"),
                               ("waterfall", "waterfall_raw_v2.png", "image/png"),
                               ("waterfall_filtered", "waterfall_filtered_v2.png", "image/png"),
                               ("filtered_waveform", "filtered_waveform_v2.npz", "application/octet-stream"))]
    windows.write_result(service, analysis, result)
    return result


def test_candidate_filter_runs_before_pagination_and_excludes_unfinished(context):
    session, site, profile, service = context
    start = datetime.fromisoformat(windows.plan()["start_utc"])
    jobs = []
    for i in range(4):
        plan = windows.plan() | {"start_utc": (start + timedelta(minutes=i)).isoformat(),
                                "end_utc": (start + timedelta(minutes=i, seconds=30)).isoformat()}
        job = enqueue_windows(session, site.id, profile.id, [plan])[0]
        job.status = "succeeded" if i < 3 else "running"
        job.result_json = json.dumps({"candidate": i != 0, "score": i})
        jobs.append(job)
    session.flush()
    result = service.list(candidate_only=True, review_status="all", limit=1, offset=1)
    assert result["total"] == 2 and result["items"][0]["id"] == jobs[1].window_id
    assert result["items"][0]["candidate"] is True
    assert service.list(review_status="all")["total"] == 4


def test_coverage_api_measures_local_cache_without_changing_artifacts_or_labels(event_api, context):
    from das_sentinel.services.event_windows import WindowService

    client, settings, sessions, csrf, ids = event_api
    with sessions.begin() as session:
        analysis = enqueue_windows(session, ids["site_a"], ids["profile_a"], [windows.plan()])[0]
        analysis.attempt_count = 1
        service = WindowService(session, settings.app_data_dir / "window-artifacts")
        result = install_visualization(service, analysis)
        analysis.status, analysis.input_sha256 = "succeeded", result["input_sha256"]
        analysis.result_json = json.dumps(result)
        wid, aid, signature, frozen = analysis.window_id, analysis.id, analysis.input_sha256, analysis.result_json
    body = {"analysis_id": aid, "input_sha256": signature}
    response = client.post(f"/api/v1/event-windows/{wid}/spatial-coverage",
        json=body, headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200, response.json()
    # This fixture has no channels below the surface; unknown must never become zero.
    assert response.json()["score"] is None
    assert response.json()["algorithm_version"] == "borehole-coverage-v3"
    assert response.json()["coverage_score"] is None
    assert response.json()["propagation_status"] == "insufficient_support"
    assert response.json()["common_mode_status"] == "unknown"
    assert response.json()["common_mode_penalty"] == 0
    assert response.json()["review_protected"] is False
    assert response.json()["common_mode_energy_fraction"] is None
    assert response.json()["common_mode_channel_fraction"] is None
    assert response.json()["residual_coverage_score"] is None
    assert response.json()["valid_coverage_fraction"] == 0
    assert client.post(f"/api/v1/event-windows/{wid}/spatial-coverage",
        json=body, headers={"X-CSRF-Token": csrf}).json() == response.json()
    detail = client.get(f"/api/v1/event-windows/{wid}").json()
    assert detail["spatial_coverage"] == response.json() and detail["latest_review"] is None
    with sessions() as session:
        from das_sentinel.persistence.window_models import WindowAnalysis
        analysis = session.get(WindowAnalysis, aid)
        assert analysis.result_json == frozen and analysis.input_sha256 == signature
        service = WindowService(session, settings.app_data_dir / "window-artifacts")
        for artifact in result["artifacts"]:
            service.artifact(wid, artifact["kind"], aid)


def test_coverage_api_protects_positive_reviews_without_rewriting_measurement(event_api, context):
    from das_sentinel.persistence.models import ProcessingProfile, Site
    from das_sentinel.persistence.window_models import (
        WindowReview,
        WindowSpatialCoverage,
    )
    from das_sentinel.services.event_windows import WindowService
    from sqlalchemy import select

    client, settings, sessions, csrf, ids = event_api
    with sessions.begin() as session:
        service = WindowService(session, settings.app_data_dir / "window-artifacts")
        local_context = (session, session.get(Site, ids["site_a"]),
            session.get(ProcessingProfile, ids["profile_a"]), service)
        positive = windows.common_mode_window(local_context, 0, version="borehole-coverage-v3",
            labels={"earthquake": "present"})
        other = windows.common_mode_window(local_context, 1, score=80, version="borehole-coverage-v3")
        wid, aid, signature = positive.window_id, positive.id, positive.input_sha256
        before = (positive.result_json, positive.input_sha256,
            session.scalar(select(WindowReview.labels_json)))
    page = client.get("/api/v1/event-windows", params={
        "review_status": "all", "window_seconds": 60, "sort": "borehole_coverage"}).json()
    assert page["total"] == 2
    assert [item["id"] for item in page["items"]] == [wid, other.window_id]
    assert page["items"][0]["spatial_coverage"]["score"] == 90
    response = client.post(f"/api/v1/event-windows/{wid}/spatial-coverage",
        json={"analysis_id": aid, "input_sha256": signature}, headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200, response.json()
    protected = response.json()
    assert protected["score"] == 90 and protected["common_mode_penalty"] == 0
    assert protected["review_protected"] is True
    assert protected["common_mode_status"] == "component_supported"
    assert client.get(f"/api/v1/event-windows/{wid}").json()["spatial_coverage"] == protected
    with sessions() as session:
        row = session.get(WindowSpatialCoverage, aid)
        assert row.score == 10
        assert json.loads(row.payload_json)["common_mode_penalty"] == 80
        assert "review_protected" not in json.loads(row.payload_json)
        from das_sentinel.persistence.window_models import WindowAnalysis
        analysis = session.get(WindowAnalysis, aid)
        assert (analysis.result_json, analysis.input_sha256,
            session.scalar(select(WindowReview.labels_json))) == before


def test_successful_worker_result_automatically_saves_bound_coverage(event_api, context):
    from das_sentinel.persistence.window_models import (
        WindowAnalysis,
        WindowSpatialCoverage,
    )
    from das_sentinel.services.event_windows import WindowService

    from tests.test_events_api import WORKER_TOKEN

    client, settings, sessions, _, ids = event_api
    with sessions.begin() as session:
        analysis = enqueue_windows(session, ids["site_a"], ids["profile_a"], [windows.plan()])[0]
        service = WindowService(session, settings.app_data_dir / "window-artifacts")
        service.claim("coverage-worker")
        expected = install_visualization(service, analysis)
        aid = analysis.id
    response = client.post(f"/api/v1/internal/window-jobs/{aid}/result",
        json={"worker_id": "coverage-worker", "attempt_count": 1},
        headers={"X-Worker-Token": WORKER_TOKEN})
    assert response.status_code == 200
    with sessions() as session:
        analysis = session.get(WindowAnalysis, aid)
        coverage = session.get(WindowSpatialCoverage, aid)
        assert analysis.status == "succeeded" and coverage is not None
        assert coverage.input_sha256 == analysis.input_sha256 == expected["input_sha256"]
        assert coverage.score is None
        assert json.loads(analysis.result_json) == expected


@pytest.mark.parametrize("regions_baked", [False, None, True])
def test_accepts_visual_manifest_preserves_input_and_serves_two_local_previews(context, regions_baked):
    session, site, profile, service = context
    analysis = enqueue_windows(session, site.id, profile.id, [windows.plan()])[0]
    service.claim("w")
    result = install_visualization(service, analysis, regions_baked=regions_baked)
    service.result(analysis.id, "w", 1)
    assert analysis.input_sha256 == result["input_sha256"]
    detail = service.detail(analysis.window_id)
    assert detail["detection_regions"] == result["detection_regions"]
    assert detail["plot_geometry"] == result["plot_geometry"]
    assert detail["waterfall_regions_baked"] is (True if regions_baked is None else regions_baked)
    raw = service.region_preview(analysis.window_id, analysis.id, 1, False)
    filtered = service.region_preview(analysis.window_id, analysis.id, 1, True)
    assert raw.startswith(b"\x89PNG") and raw != filtered
    saved_result = analysis.result_json
    folder = service.artifact_root / analysis.id / "attempt-1"
    with np.load(folder / "waveform.npz") as arrays:
        raw_samples = arrays["waveform"].T
        _, clean = create_visualization(
            raw_samples, raw_samples.astype("f4") * 0.5, arrays["distance_m"],
            datetime.fromisoformat(result["start_utc"]), arrays["recognition_channel_mask"],
            result["unit"], [], merge_gap_seconds=0,
        )
    assert service.waterfall_preview(analysis.window_id, analysis.id, False) == clean["waterfall_raw_v2.png"]
    assert service.waterfall_preview(analysis.window_id, analysis.id, True) == clean["waterfall_filtered_v2.png"]
    for name in ("waterfall_raw_v2.png", "waterfall_filtered_v2.png"):
        assert ((folder / name).read_bytes() == clean[name]) is (regions_baked is False)
    assert service.region_preview(analysis.window_id, analysis.id, 1, False, show_regions=False) != raw
    assert service.region_preview(analysis.window_id, analysis.id, 1, True, show_regions=False) != filtered
    assert analysis.result_json == saved_result and analysis.input_sha256 == result["input_sha256"]
    with pytest.raises(WindowError, match="region_not_found"):
        service.region_preview(analysis.window_id, analysis.id, 200, False)
    other = enqueue_windows(session, site.id, profile.id, [windows.plan() | {"algorithm_version": "future"}])[0]
    with pytest.raises(WindowError):
        service.region_preview(analysis.window_id, other.id, 1, False)
    with pytest.raises(WindowError):
        service.waterfall_preview(analysis.window_id, other.id, False)
    folder = service.artifact_root / analysis.id / "attempt-1"
    (folder / "filtered_waveform_v2.npz").write_bytes(b"changed")
    with pytest.raises(WindowError, match="artifact_hash_mismatch"):
        service.region_preview(analysis.window_id, analysis.id, 1, True)
    with pytest.raises(WindowError, match="artifact_hash_mismatch"):
        service.waterfall_preview(analysis.window_id, analysis.id, True)
    (folder / "waveform.npz").write_bytes(b"changed")
    with pytest.raises(WindowError, match="artifact_hash_mismatch"):
        service.waterfall_preview(analysis.window_id, analysis.id, False)


def test_image_control_routes_hide_baked_regions_and_require_authentication(event_api, context):
    from das_sentinel.services.event_windows import WindowService

    client, settings, sessions, _, ids = event_api
    with sessions.begin() as session:
        analysis = enqueue_windows(session, ids["site_a"], ids["profile_a"], [windows.plan()])[0]
        analysis.attempt_count = 1
        service = WindowService(session, settings.app_data_dir / "window-artifacts")
        result = install_visualization(service, analysis)
        analysis.status = "succeeded"
        analysis.result_json = json.dumps(result)
        wid, aid = analysis.window_id, analysis.id
    params = {"analysis_id": aid, "region_id": 1, "filtered": "false"}
    full_url = f"/api/v1/event-windows/{wid}/waterfall-preview"
    local_url = f"/api/v1/event-windows/{wid}/region-preview"
    full = client.get(full_url, params=params)
    assert full.status_code == 200 and full.headers["content-type"] == "image/png"
    marked = client.get(local_url, params=params)
    clean = client.get(local_url, params=params | {"show_regions": "false"})
    assert marked.status_code == clean.status_code == 200
    assert marked.content != clean.content and clean.content.startswith(b"\x89PNG")
    client.cookies.clear()
    assert client.get(full_url, params=params).status_code == 401
    assert client.get(local_url, params=params).status_code == 401


def test_rejects_regions_that_do_not_match_qualified_evidence(context):
    _, site, profile, service = context
    analysis = enqueue_windows(service.session, site.id, profile.id, [windows.plan()])[0]
    service.claim("w")
    result = install_visualization(service, analysis)
    result["detection_regions"][0]["start_sample"] = -1
    folder = service.artifact_root / analysis.id / "attempt-1"
    metadata = folder / "metadata_visual_v2.json"
    metadata.write_text(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), encoding="utf-8")
    for item in result["artifacts"]:
        if item["kind"] == "metadata":
            item["sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    windows.write_result(service, analysis, result)
    with pytest.raises(WindowError, match="invalid_visualization"):
        service.result(analysis.id, "w", 1)


@pytest.mark.parametrize("filtered", [False, True])
def test_plain_waterfall_preview_reuses_verified_png_without_loading_or_rendering(context, monkeypatch, filtered):
    _, site, profile, service = context
    analysis = enqueue_windows(service.session, site.id, profile.id, [windows.plan()])[0]
    service.claim("w")
    install_visualization(service, analysis)
    service.result(analysis.id, "w", 1)
    name = "waterfall_filtered_v2.png" if filtered else "waterfall_raw_v2.png"
    path = service.artifact_root / analysis.id / "attempt-1" / name
    expected = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("plain waterfall preview must reuse the PNG without loading NPZ or rendering")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr("das_sentinel.whole_window_plots.render_waterfall_preview", forbidden)
    assert service.waterfall_preview(analysis.window_id, analysis.id, filtered) == expected
    path.write_bytes(b"changed")
    with pytest.raises(WindowError, match="artifact_hash_mismatch"):
        service.waterfall_preview(analysis.window_id, analysis.id, filtered)


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", [], {}])
def test_complete_rejects_non_boolean_waterfall_regions_baked(context, invalid):
    _, site, profile, service = context
    analysis = enqueue_windows(service.session, site.id, profile.id, [windows.plan()])[0]
    service.claim("w")
    result = install_visualization(service, analysis)
    result["waterfall_regions_baked"] = invalid
    folder = service.artifact_root / analysis.id / "attempt-1"
    metadata = folder / "metadata_visual_v2.json"
    metadata.write_text(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), encoding="utf-8")
    for item in result["artifacts"]:
        if item["kind"] == "metadata":
            item["sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    windows.write_result(service, analysis, result)
    with pytest.raises(WindowError, match="invalid_waterfall_regions_baked"):
        service.result(analysis.id, "w", 1)
