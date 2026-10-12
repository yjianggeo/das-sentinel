from __future__ import annotations

import json
import time
from uuid import uuid4

import pytest


def payload():
    return {
        "site_id": str(uuid4()),
        "relative_path": "day",
        "start_utc": "2026-06-16T09:05:00+00:00",
        "end_utc": "2026-06-16T09:15:00+00:00",
        "distance_start_m": 499.0,
        "distance_stop_m": None,
    }


def service(tmp_path, monkeypatch):
    from das_sentinel.services.low_frequency_scans import LowFrequencyScanService

    raw = tmp_path / "raw"
    (raw / "day").mkdir(parents=True)
    instance = LowFrequencyScanService(tmp_path / "scans", raw)

    def analyze(scan, directory):
        assert directory == raw / "day"
        return {"candidates": [{"id": "candidate-one", "score": 0.8}], "plots": {}}

    monkeypatch.setattr(instance, "_analyze", analyze)
    return instance, raw


def completed(instance, scan_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = instance.get(scan_id)
        if result["status"] in {"succeeded", "failed"}:
            return result
        time.sleep(0.01)
    pytest.fail("scan did not finish")


def test_scan_persists_result_and_idempotent_review(tmp_path, monkeypatch):
    instance, raw = service(tmp_path, monkeypatch)
    try:
        scan = instance.create(payload(), raw)
        result = completed(instance, scan["id"])
        assert result["status"] == "succeeded"
        assert result["result"]["candidates"][0]["latest_review"] is None
        request_id = str(uuid4())
        review = instance.review(
            scan["id"], "candidate-one", "earthquake", "低频波列", request_id
        )
        assert (
            instance.review(
                scan["id"], "candidate-one", "earthquake", "低频波列", request_id
            )
            == review
        )
        with pytest.raises(ValueError, match="重复"):
            instance.review(scan["id"], "candidate-one", "noise", "", request_id)
        assert (
            instance.get(scan["id"])["result"]["candidates"][0]["latest_review"][
                "label"
            ]
            == "earthquake"
        )
        saved = json.loads(
            (tmp_path / "scans" / scan["id"] / "reviews.json").read_text(
                encoding="utf-8"
            )
        )
        assert len(saved) == 1
    finally:
        instance.close()
    from das_sentinel.services.low_frequency_scans import LowFrequencyScanService

    recovered = LowFrequencyScanService(tmp_path / "scans", raw)
    try:
        assert (
            recovered.get(scan["id"])["result"]["candidates"][0]["latest_review"]
            == review
        )
    finally:
        recovered.close()


@pytest.mark.parametrize(
    "relative", ["..", "../outside", "/absolute", "C:\\outside", "day/../day", "."]
)
def test_rejects_unsafe_source_directory(tmp_path, monkeypatch, relative):
    instance, raw = service(tmp_path, monkeypatch)
    try:
        with pytest.raises(ValueError):
            instance.create({**payload(), "relative_path": relative}, raw)
        assert instance.list() == []
    finally:
        instance.close()


def test_restart_marks_unfinished_scan_failed_without_resuming(tmp_path):
    from das_sentinel.services.low_frequency_scans import LowFrequencyScanService

    root = tmp_path / "scans"
    scan_id = str(uuid4())
    folder = root / scan_id
    folder.mkdir(parents=True)
    (folder / "scan.json").write_text(
        json.dumps(
            {
                **payload(),
                "id": scan_id,
                "status": "running",
                "created_at": "2026-06-16T00:00:00Z",
                "error_message": None,
            }
        )
    )
    instance = LowFrequencyScanService(root, tmp_path)
    try:
        assert instance.get(scan_id)["status"] == "failed"
        assert "中断" in instance.get(scan_id)["error_message"]
    finally:
        instance.close()


def test_analysis_error_is_saved_and_unknown_candidate_cannot_be_reviewed(
    tmp_path, monkeypatch
):
    instance, raw = service(tmp_path, monkeypatch)

    def fail(*args):
        raise ValueError("连续数据中存在缺口")

    monkeypatch.setattr(instance, "_analyze", fail)
    try:
        scan = instance.create(payload(), raw)
        result = completed(instance, scan["id"])
        assert result["status"] == "failed"
        assert result["error_message"] == "连续数据中存在缺口"
        with pytest.raises(ValueError):
            instance.review(scan["id"], "candidate-one", "noise", "", str(uuid4()))
    finally:
        instance.close()
