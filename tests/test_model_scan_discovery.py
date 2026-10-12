import importlib
import shutil

import pytest
from das_sentinel.model_scan_contract import digest
from das_sentinel.services.event_windows import WindowError
from das_sentinel.whole_window import inspect_sources
from sqlalchemy import func, select

from tests.test_event_windows import context as context  # noqa: PLC0414
from tests.test_model_scans import create, service
from tests.test_whole_window import BASE, write


def module():
    assert importlib.util.find_spec("das_sentinel.services.model_scan_discovery"), "model discovery missing"
    return importlib.import_module("das_sentinel.services.model_scan_discovery")


def test_plan_is_half_minute_unique_and_bounded(tmp_path):
    api = module()
    paths = [write(tmp_path / "a.h5", -60, 120), write(tmp_path / "b.h5", 60, 120)]
    blocks = api.plan_source_windows(inspect_sources(paths), block_seconds=60, cache_root=tmp_path / "cache")
    plans = [p for block in blocks for p in block["windows"]]
    assert BASE.isoformat() in [p["start_utc"] for p in plans]
    assert len(plans) == len({p["start_utc"] for p in plans})
    assert all(len(b["windows"]) <= 2 for b in blocks)


def test_identical_copy_collapses_but_conflicting_recording_does_not(tmp_path):
    api = module()
    original = write(tmp_path / "a.h5")
    duplicate = tmp_path / "b.h5"
    shutil.copyfile(original, duplicate)
    blocks = api.plan_source_windows(inspect_sources([original, duplicate]), cache_root=tmp_path / "cache")
    assert next(p for b in blocks for p in b["windows"] if p["start_utc"] == BASE.isoformat())["coverage"] == "complete"
    import h5py
    with h5py.File(duplicate, "r+") as stream:
        stream["Acquisition/Raw[0]/RawData"][0, 0] += 1
    blocks = api.plan_source_windows(inspect_sources([original, duplicate]), cache_root=tmp_path / "cache")
    assert all(p["coverage"] == "incomplete" for b in blocks for p in b["windows"])


def test_discovery_uses_source_clock_and_reports_bad_file(tmp_path):
    api = module()
    data = tmp_path / "20441231"
    data.mkdir()
    write(data / "wrong-name.h5")
    (data / "bad.h5").write_bytes(b"bad hdf")
    pages = list(api.discover_directory(tmp_path, data, stable_seconds=0))
    sources = [s for p in pages for s in p["sources"]]
    errors = [e for p in pages for e in p["errors"]]
    assert len(sources) == 1 and sources[0]["start_utc"].startswith("2023-09-27")
    assert len(errors) == 1 and errors[0]["path"].endswith("bad.h5")


@pytest.mark.parametrize("context_only", [False, True])
def test_empty_hdf_does_not_abort_remaining_sources(tmp_path, context_only):
    from tests.fixtures import write_zd_h5
    api = module()
    data = tmp_path / "data"
    data.mkdir()
    empty = data / "a-empty.h5"
    write_zd_h5(empty, n_time=0)
    valid = write(data / "z-valid.h5")
    pages = list(api.discover_context(tmp_path, [str(empty), str(valid)], cache_root=tmp_path / "cache")
                 if context_only else api.discover_directory(tmp_path, data, stable_seconds=0, cache_root=tmp_path / "cache"))
    sources = [s for page in pages for s in page["sources"]]
    errors = [e for page in pages for e in page["errors"]]
    assert len(sources) == 1 and sources[0]["path"] == str(valid)
    assert len(errors) == 1 and errors[0]["path"] == str(empty)
    assert "empty" in errors[0]["error"]


def test_discovery_pages_replay_but_incomplete_manifest_cannot_finish(context, tmp_path):
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    job = api.claim("reader")
    source = inspect_sources([write(tmp_path / "a.h5")])[0]
    payload = {"sources": [source], "directories": [], "errors": []}
    first = api.commit_discovery_page(job["id"], "reader", 1, 0, payload)
    assert api.commit_discovery_page(job["id"], "reader", 1, 0, payload) == first
    with pytest.raises(WindowError, match="incomplete_discovery"):
        api.finish_discovery(job["id"], "reader", 1, 2, digest([digest(payload)]))
    api.finish_discovery(job["id"], "reader", 1, 1, digest([digest(payload)]))
    from das_sentinel.persistence.model_scan_models import ModelScanSource
    assert context[0].scalar(select(func.count()).select_from(ModelScanSource)) == 1
    assert api.get(batch["id"])["discovery_complete"]
    plan = api.claim("reader", kind="plan")
    assert plan is not None and len(plan["payload"]["sources"]) == 1
    with pytest.raises(WindowError, match="page_conflict"):
        api.commit_discovery_page(job["id"], "reader", 1, 0, payload | {"errors": [{"path": "x", "error": "bad"}]})


def test_planned_receipts_deduplicate_and_deleted_window_is_excluded(context, tmp_path):
    api = service(context, tmp_path)
    from tests.test_model_scans import MODEL
    batch = api.create(context[1].id, context[2].id, "once", [str(tmp_path)], MODEL,
                       [{"start_utc": BASE.isoformat(), "window_seconds": 30}])
    job = api.claim("reader")
    source = inspect_sources([write(tmp_path / "a.h5")])[0]
    payload = {"sources": [source], "directories": [], "errors": []}
    api.commit_discovery_page(job["id"], "reader", 1, 0, payload)
    api.finish_discovery(job["id"], "reader", 1, 1, digest([digest(payload)]))
    job = api.claim("reader", kind="plan")
    blocks = module().plan_source_windows(job["payload"]["sources"], cache_root=tmp_path / "cache",
        start_utc=job["payload"]["start_utc"], end_utc=job["payload"]["end_utc"])
    api.finish_plan(job["id"], "reader", job["attempt_count"], blocks)
    infer = api.claim("reader", kind="infer")
    assert BASE.isoformat() not in [p["start_utc"] for p in infer["payload"]["windows"]]
    assert api.get(batch["id"])["counts"]["excluded"] == 1
    for plan in infer["payload"]["windows"]:
        api.record_prediction(infer["id"], "reader", infer["attempt_count"], {
            "start_utc": plan["start_utc"], "status": "failed", "error_code": "test_read_error"})
    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    assert context[0].get(ModelScanUnit, infer["id"]).status == "succeeded"


def test_unaligned_ranges_do_not_lose_block_boundary_and_missing_plan_rejected(context, tmp_path):
    from datetime import timedelta

    from tests.test_model_scans import MODEL
    api = service(context, tmp_path)
    api.create(context[1].id, context[2].id, "range", [str(tmp_path)], MODEL, [], time_ranges=[{
        "start_utc": (BASE + timedelta(seconds=15)).isoformat(),
        "end_utc": (BASE + timedelta(seconds=915)).isoformat()}])
    source = inspect_sources([write(tmp_path / "a.h5")])[0]
    source.update(start_utc=BASE.isoformat(), end_utc=(BASE + timedelta(seconds=1000)).isoformat())
    job = api.claim("reader")
    page = {"sources": [source], "directories": [], "errors": []}
    api.commit_discovery_page(job["id"], "reader", 1, 0, page)
    api.finish_discovery(job["id"], "reader", 1, 1, digest([digest(page)]))
    job = api.claim("reader", kind="plan")
    assert job["payload"]["start_utc"] == (BASE + timedelta(seconds=30)).isoformat()
    with pytest.raises(WindowError, match="incomplete_window_plan"):
        api.finish_plan(job["id"], "reader", job["attempt_count"], [])
    blocks = module().plan_source_windows(job["payload"]["sources"], cache_root=tmp_path / "cache",
        start_utc=job["payload"]["start_utc"], end_utc=job["payload"]["end_utc"])
    api.finish_plan(job["id"], "reader", job["attempt_count"], blocks)
    next_job = api.claim("reader", kind="plan")
    assert next_job["payload"]["start_utc"] == job["payload"]["end_utc"]


def test_registered_context_is_discovered_without_expanding_requested_dates(context, tmp_path):
    from das_sentinel.persistence.models import SourceFile
    source = inspect_sources([write(tmp_path / "eq.h5")])[0]
    context[0].add(SourceFile(site_id=context[1].id, **{k: source[k] for k in
        ("path", "fingerprint", "size_bytes", "mtime_ns")}))
    context[0].flush()
    api = service(context, tmp_path)
    batch = create(api, context, tmp_path)
    import json

    from das_sentinel.persistence.model_scan_models import ModelScanUnit
    units = context[0].scalars(select(ModelScanUnit).where(ModelScanUnit.scan_id == batch["id"])).all()
    assert any(json.loads(u.payload_json).get("context_paths") == [source["path"]] for u in units)


def test_previous_negative_is_reused_by_exact_frozen_input_identity(context, tmp_path):
    from das_sentinel.persistence.model_scan_models import ModelScanReceipt

    from tests.test_model_scans import MODEL
    source = inspect_sources([write(tmp_path / "a.h5")])[0]
    api = service(context, tmp_path)
    for index in range(2):
        batch = create(api, context, tmp_path, request=str(index))
        job = api.claim("reader", kind="discover")
        page = {"sources": [source], "directories": [], "errors": []}
        api.commit_discovery_page(job["id"], "reader", job["attempt_count"], 0, page)
        api.finish_discovery(job["id"], "reader", job["attempt_count"], 1, digest([digest(page)]))
        job = api.claim("reader", kind="plan")
        blocks = module().plan_source_windows(job["payload"]["sources"], cache_root=tmp_path / "cache",
            start_utc=job["payload"]["start_utc"], end_utc=job["payload"]["end_utc"])
        api.finish_plan(job["id"], "reader", job["attempt_count"], blocks)
        infer = api.claim("reader", kind="infer")
        if index == 0:
            for plan in infer["payload"]["windows"]:
                api.record_prediction(infer["id"], "reader", 1, {"start_utc": plan["start_utc"],
                    "status": "succeeded", "model_id": MODEL["id"], "score": .01, "model_input_sha256": "a" * 64})
            planner = api.claim("reader", kind="plan")
            api.finish_plan(planner["id"], "reader", planner["attempt_count"], [])
        else:
            assert infer is None
            assert api.get(batch["id"])["counts"]["reused"] > 0
    assert context[0].scalar(select(func.count()).select_from(ModelScanReceipt).where(ModelScanReceipt.status == "negative")) > 0


def test_interrupted_discovery_rebuilds_manifest_before_planning(context, tmp_path):
    from datetime import UTC, datetime, timedelta

    from das_sentinel.persistence.model_scan_models import (
        ModelScanSource,
        ModelScanUnit,
    )
    api = service(context, tmp_path)
    create(api, context, tmp_path)
    first = api.claim("old")
    source = inspect_sources([write(tmp_path / "a.h5")])[0]
    page = {"sources": [source], "directories": [], "errors": []}
    api.commit_discovery_page(first["id"], "old", 1, 0, page)
    context[0].get(ModelScanUnit, first["id"]).heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    context[0].flush()
    retry = api.claim("new")
    assert context[0].scalar(select(func.count()).select_from(ModelScanSource)) == 0
    empty = {"sources": [], "directories": [], "errors": []}
    api.commit_discovery_page(retry["id"], "new", 2, 0, empty)
    api.finish_discovery(retry["id"], "new", 2, 1, digest([digest(empty)]))
    planner = api.claim("new", kind="plan")
    api.finish_plan(planner["id"], "new", planner["attempt_count"], [])
    assert api.finish_discovery(retry["id"], "new", 2, 1, digest([digest(empty)])) == {"status": "succeeded"}


def test_stale_optional_context_does_not_block_valid_requested_sources(tmp_path):
    source = inspect_sources([write(tmp_path / "valid.h5")])[0]
    stale = source | {"path": str(tmp_path / "removed-optional-copy.h5")}
    blocks = module().plan_source_windows([source, stale], requested_sources=[source], cache_root=tmp_path / "cache")
    plan = next(p for b in blocks for p in b["windows"] if p["start_utc"] == BASE.isoformat())
    assert plan["coverage"] == "complete"
    assert plan["sources"] == [source]


def test_changed_requested_source_keeps_windows_as_unjudgeable_plans(tmp_path):
    source = inspect_sources([write(tmp_path / "valid.h5")])[0]
    source["mtime_ns"] -= 1
    blocks = module().plan_source_windows([source], requested_sources=[source], cache_root=tmp_path / "cache")
    assert blocks and all(p["coverage"] == "incomplete" for b in blocks for p in b["windows"])


def test_metadata_cache_reuses_directory_headers_for_registered_context(tmp_path, monkeypatch):
    api = module()
    planning = importlib.import_module("das_sentinel.services.window_planning")
    path = write(tmp_path / "sample.h5")
    cache = tmp_path / "metadata"
    first = list(api.discover_directory(tmp_path, tmp_path, stable_seconds=0, cache_root=cache))
    source = first[0]["sources"][0]
    assert source["start_utc"] == "2023-09-27T03:01:00+00:00"
    assert source["distance_m"] == [0, 100, 200, 300, 400, 500, 600, 700]

    def unexpected_read(*args):
        pytest.fail("unchanged cached source must not reopen HDF5 metadata")

    monkeypatch.setattr(planning, "inspect_sources", unexpected_read)
    monkeypatch.setattr(api, "inspect_sources", unexpected_read)
    second = list(api.discover_context(tmp_path, [str(path)], cache_root=cache))
    third = list(api.discover_context(tmp_path, [str(path)], cache_root=cache))
    assert second == third == [{"sources": [source], "directories": [], "errors": []}]


def test_cached_metadata_never_bypasses_root_boundary(tmp_path):
    api = module()
    root = tmp_path / "allowed"
    root.mkdir()
    path = write(tmp_path / "outside.h5")
    cache = tmp_path / "metadata"
    list(api.discover_context(tmp_path, [str(path)], cache_root=cache))
    # Exhaust discovery before inspecting the first page.
    page = list(api.discover_context(root, [str(path)], cache_root=cache))[0]  # noqa: RUF015
    assert page["sources"] == []
    assert len(page["errors"]) == 1 and page["errors"][0]["path"] == str(path)


def test_worker_discovery_uses_current_version_cache_and_ignores_v1(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from uuid import uuid4

    from das_sentinel.services.discovery import DiscoveredFile
    from das_sentinel.whole_window import ALGORITHM_VERSION
    from das_sentinel.worker.model_scan_main import ModelScanWorker
    api = module()
    planning = importlib.import_module("das_sentinel.services.window_planning")
    data = tmp_path / "raw"
    data.mkdir()
    path = write(data / "sample.h5")
    old = tmp_path / "app" / "window-source-metadata-whole-window-v1"
    old.mkdir(parents=True)
    source = inspect_sources([path])[0]
    (old / f"{DiscoveredFile.from_path(path).fingerprint}.json").write_text(
        json.dumps(source | {"unit": "old-cache-must-not-be-used"}), encoding="utf8")
    submitted = []
    payload = {"directory": str(data)}

    class Client:
        def claim(self, worker_id):
            return {"id": str(uuid4()), "attempt_count": 1, "kind": "discover",
                    "status": "running", "config": {"root": str(data)}, "payload": payload}

        def heartbeat(self, *args):
            return None

        def submit(self, unit, action, worker, attempt, **body):
            submitted.append((action, body))

    settings = SimpleNamespace(data_root=data, app_data_dir=tmp_path / "app")
    monkeypatch.setattr(api.time, "sleep", lambda seconds: None)
    worker = ModelScanWorker(settings, Client(), "test-cache")
    assert worker.run_once().status == "succeeded"
    page = next(body["payload"] for action, body in submitted if action == "discovery-page")
    assert page["sources"][0]["unit"] == "strain_rate"
    current = settings.app_data_dir / f"window-source-metadata-{ALGORITHM_VERSION}"
    assert (current / f"{source['fingerprint']}.json").is_file()

    def unexpected_read(*args):
        pytest.fail("next discovery unit should reuse the worker's versioned cache")

    monkeypatch.setattr(planning, "inspect_sources", unexpected_read)
    monkeypatch.setattr(api, "inspect_sources", unexpected_read)
    payload = {"context_paths": [str(path)]}
    submitted.clear()
    assert worker.run_once().status == "succeeded"
    assert next(body["payload"] for action, body in submitted if action == "discovery-page") == page
