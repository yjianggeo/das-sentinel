from datetime import timedelta

import pytest
from das_sentinel.model_scan_contract import digest
from das_sentinel.persistence.model_scan_models import ModelScanReceipt
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
)
from das_sentinel.services.model_scan_discovery import plan_source_windows
from das_sentinel.whole_window import inspect_sources
from sqlalchemy import func, select

from tests.test_event_windows import context as context  # noqa: PLC0414
from tests.test_model_scans import create, service
from tests.test_whole_window import BASE, write


def old_source(tmp_path):
    return {
        "path": str(tmp_path / "old.h5"), "fingerprint": "old", "size_bytes": 100, "mtime_ns": 1,
        "start_utc": (BASE - timedelta(seconds=60)).isoformat(),
        "end_utc": (BASE + timedelta(seconds=120)).isoformat(),
        "sample_rate_hz": 100, "unit": "strain_rate",
        "distance_m": [i * 2.044989824295044 for i in range(1663)],
    }


def test_excluded_sources_need_no_file_reads_or_hashing(tmp_path, monkeypatch):
    from das_sentinel.services import model_scan_discovery as discovery
    def forbidden(*args, **kwargs):
        raise AssertionError("excluded sources must not be opened or hashed")
    monkeypatch.setattr(discovery.DiscoveredFile, "from_path", forbidden)
    monkeypatch.setattr(discovery, "deduplicate_sources", forbidden)
    source = old_source(tmp_path)
    blocks = plan_source_windows([source], cache_root=tmp_path / "cache")
    assert len([p for b in blocks for p in b["windows"]]) == 6
    assert all(p["sources"] == [source] for b in blocks for p in b["windows"])


def test_excluded_copy_does_not_break_normal_2023_coverage(tmp_path):
    valid = inspect_sources([write(tmp_path / "normal.h5")])[0]
    blocks = plan_source_windows([valid, old_source(tmp_path)], cache_root=tmp_path / "cache")
    plan = next(p for b in blocks for p in b["windows"] if p["start_utc"] == BASE.isoformat())
    assert plan["coverage"] == "complete"
    assert plan["sources"] == [valid]


@pytest.mark.parametrize("reviewed", [False, True])
def test_two_metre_windows_are_excluded_before_inference_and_not_retryable(context, tmp_path, reviewed):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = api.claim("worker", kind="discover")
    source = old_source(tmp_path)
    page = {"sources": [source], "directories": [], "errors": []}
    api.commit_discovery_page(job["id"], "worker", 1, 0, page)
    api.finish_discovery(job["id"], "worker", 1, 1, digest([digest(page)]))
    planner = api.claim("worker", kind="plan")
    blocks = plan_source_windows(planner["payload"]["sources"], cache_root=tmp_path / "cache",
        start_utc=planner["payload"]["start_utc"], end_utc=planner["payload"]["end_utc"])
    if reviewed:
        from das_sentinel.persistence.models import Admin
        from das_sentinel.services.event_windows import enqueue_windows
        analysis = enqueue_windows(context[0], context[1].id, context[2].id, [blocks[0]["windows"][2]])[0]
        context[0].add(Admin(id=1, password_hash="test"))
        context[0].flush()
        analysis.status, analysis.input_sha256 = "succeeded", "a" * 64
        context[0].add(WindowReview(window_id=analysis.window_id, analysis_id=analysis.id,
            input_sha256=analysis.input_sha256, labels_json='{"earthquake":"present"}',
            request_id="human", admin_id=1))
        context[0].flush()
    api.finish_plan(planner["id"], "worker", planner["attempt_count"], blocks)
    assert api.claim("worker", kind="infer") is None
    receipts = context[0].scalars(select(ModelScanReceipt)).all()
    assert len(receipts) == 6
    assert sum(r.status == "protected" for r in receipts) == int(reviewed)
    assert all(r.status in ("excluded", "protected") and r.score is None for r in receipts)
    assert all(r.error_code == "excluded_2023_two_metre" for r in receipts if r.status == "excluded")
    api.control(batch["id"], "retry")
    assert api.claim("worker", kind="infer") is None
    for table in (EventWindow, WindowAnalysis, WindowReview):
        assert context[0].scalar(select(func.count()).select_from(table)) == int(reviewed)


def test_exclusion_requires_specific_geometry_and_2023(tmp_path):
    from das_sentinel.model_scan_contract import excluded_two_metre_source
    source = old_source(tmp_path)
    assert excluded_two_metre_source(source)
    assert not excluded_two_metre_source(source | {"distance_m": [i * 4.089979648590088 for i in range(832)]})
    assert not excluded_two_metre_source(source | {"start_utc": "2024-01-01T00:00:00+00:00"})
    assert not excluded_two_metre_source(source | {"distance_m": source["distance_m"][:-1]})


@pytest.mark.parametrize("old_first", [False, True])
def test_geometry_boundary_keeps_normal_window_as_incomplete(tmp_path, old_first):
    old = old_source(tmp_path)
    if old_first:
        old["end_utc"] = BASE.isoformat()
        normal = inspect_sources([write(tmp_path / "normal.h5", start=0, duration=120)])[0]
        excluded_start, normal_start = BASE - timedelta(seconds=30), BASE
    else:
        old["start_utc"] = BASE.isoformat()
        normal = inspect_sources([write(tmp_path / "normal.h5", start=-60, duration=60)])[0]
        excluded_start, normal_start = BASE, BASE - timedelta(seconds=30)
    plans = {p["start_utc"]: p for b in plan_source_windows([old, normal], cache_root=tmp_path / "cache")
             for p in b["windows"]}
    assert plans[excluded_start.isoformat()]["sources"] == [old]
    assert plans[normal_start.isoformat()]["sources"] == [normal]
    assert plans[normal_start.isoformat()]["coverage"] == "incomplete"
