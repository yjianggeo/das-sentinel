from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import h5py
import numpy as np
import pytest
from das_sentinel.screening_evaluation import (
    matches_reference,
    run_evaluation,
    select_samples,
    snapshot_events,
    summarize_results,
    write_report,
)


def _database(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE events(id TEXT, processing_run_id TEXT, start_sample INT,
            end_sample INT, channel_start INT, channel_stop INT, occurred_at TEXT);
        CREATE TABLE processing_runs(id TEXT, source_file_id TEXT,
            processing_profile_id TEXT, source_fingerprint TEXT);
        CREATE TABLE source_files(id TEXT, site_id TEXT, path TEXT, fingerprint TEXT,
            size_bytes INT, mtime_ns INT);
        CREATE TABLE processing_profiles(id TEXT, payload_json TEXT);
        CREATE TABLE sites(id TEXT, timezone TEXT);
        CREATE TABLE reviews(id INT, event_id TEXT, label TEXT, created_at TEXT);
        CREATE TABLE detection_evidence(event_id TEXT, detector TEXT);
        INSERT INTO events VALUES('e','r',10,20,0,8,'2026-01-01T00:00:00');
        INSERT INTO processing_runs VALUES('r','s','p','abc');
        INSERT INTO source_files VALUES('s','site','wave.h5','abc',123,456);
        INSERT INTO processing_profiles VALUES('p','{}');
        INSERT INTO sites VALUES('site','Asia/Shanghai');
        INSERT INTO reviews VALUES(1,'e','noise','2026-01-02');
        INSERT INTO reviews VALUES(2,'e','earthquake','2026-01-03');
        INSERT INTO detection_evidence VALUES('e','energy');
        """)


def test_snapshot_uses_latest_review_and_preserves_database(tmp_path):
    path = tmp_path / "db.sqlite3"
    _database(path)
    before = path.read_bytes()
    rows = snapshot_events(path)
    assert len(rows) == 1
    assert rows[0]["label"] == "earthquake"
    assert rows[0]["detectors"] == ["energy"]
    assert rows[0]["source_file_id"] == "s"
    assert rows[0]["date_group"] == "2026-01-01"
    assert path.read_bytes() == before


def test_selection_keeps_targets_and_energy_only_with_stratified_controls():
    rows = [
        {"event_id": str(i), "source_file_id": f"source-{i % 3}",
         "date_group": f"2026-01-0{1 + i % 3}", "occurred_at": str(i), "start_sample": i * 100,
         "label": label, "detectors": ["energy"] if i == 8 else ["stalta"]}
        for i, label in enumerate(["earthquake", "traffic", "blast", "noise",
                                   "noise", "noise", None, None, None])
    ]
    selected = select_samples(rows, noise_limit=2, unreviewed_limit=1)
    assert {"0", "1", "2", "8"} <= {row["event_id"] for row in selected}
    assert len([row for row in selected if row["label"] == "noise"]) == 2
    assert selected == select_samples(list(reversed(rows)), noise_limit=2, unreviewed_limit=1)


def test_match_requires_time_and_channel_intersection():
    reference = {"start_sample": 100, "end_sample": 200, "channel_start": 10, "channel_stop": 20}
    assert matches_reference({"start_sample": 90, "end_sample": 160, "channel_start": 12, "channel_stop": 18}, reference)
    assert not matches_reference({"start_sample": 200, "end_sample": 300, "channel_start": 12, "channel_stop": 18}, reference)
    assert not matches_reference({"start_sample": 110, "end_sample": 190, "channel_start": 30, "channel_stop": 40}, reference)


def test_failed_read_is_not_counted_as_missed_event():
    detection = {"start_sample": 10, "end_sample": 20, "channel_start": 1, "channel_stop": 5}
    rows = [
        {"kind": "event", "label": "earthquake", "status": "ok",
         "baseline": {"matched": True, "events": [], "elapsed_s": 1},
         "v2": {"matched": True, "events": [], "elapsed_s": 2}},
        {"kind": "event", "label": "earthquake", "status": "error", "error": "source_changed"},
        {"kind": "background", "label": None, "status": "ok",
         "source_file_id": "source",
         "baseline": {"matched": False, "events": [detection], "elapsed_s": 1},
         "v2": {"matched": False, "events": [detection, {**detection, "start_sample": 11}], "elapsed_s": 2}},
    ]
    result = summarize_results(rows)
    assert result["labels"]["earthquake"] == {"evaluated": 1, "baseline_matched": 1, "v2_matched": 1}
    assert result["errors"] == 1
    assert result["background_candidates"]["v2"] == 2
    assert "false_positive_rate" not in result


def test_same_detection_in_overlapping_windows_is_reported_as_duplicate():
    detection = {"start_sample": 10, "end_sample": 20, "channel_start": 1, "channel_stop": 5}
    row = {"kind": "event", "label": "earthquake", "status": "ok", "source_file_id": "s",
           "baseline": {"matched": True, "events": [detection], "elapsed_s": 1},
           "v2": {"matched": True, "events": [detection], "elapsed_s": 1}}
    result = summarize_results([row, row])
    assert result["window_candidate_counts"]["v2"] == 2
    assert result["distinct_window_detections"]["v2"] == 1
    assert result["exact_duplicate_instances"]["v2"] == 1


def test_html_report_escapes_source_values_and_preserves_json(tmp_path):
    report = {"schema_version": "1", "summary": {"note": "exploratory"},
              "results": [{"event_id": "<script>alert(1)</script>", "status": "error"}]}
    write_report(tmp_path, report)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report


def _waveform_database(tmp_path):
    from das_sentinel.services.discovery import fingerprint
    path = tmp_path / "wave.h5"
    fs = 200
    data = np.random.default_rng(32).normal(0, .05, (6000, 8)).astype(np.float32)
    burst = 4 * np.sin(2 * np.pi * 10 * np.arange(400) / fs) * np.hanning(400)
    data[3000:3400, :4] += burst[:, None]
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition/Raw[0]")
        raw.create_dataset("RawData", data=data)
        raw.create_dataset("RawDataTime", data=1_767_225_600_000_000 + np.arange(6000) * 5000)
        raw.attrs.update(OutputDataRate=fs, SpatialSamplingInterval=4.0, StartLocusIndex=0, RawDataUnit="input")
    database = tmp_path / "db.sqlite3"
    _database(database)
    stat = path.stat()
    with sqlite3.connect(database) as db:
        db.execute("UPDATE source_files SET path=?, fingerprint=?, size_bytes=?, mtime_ns=?",
                   (str(path), fingerprint(path), stat.st_size, stat.st_mtime_ns))
        db.execute("UPDATE processing_runs SET source_fingerprint=?", (fingerprint(path),))
        db.execute("UPDATE events SET start_sample=3000,end_sample=3400,channel_stop=4")
    return database, path


def test_measurement_replay_writes_verified_sidecar_without_changing_inputs(tmp_path):
    database, waveform = _waveform_database(tmp_path)
    database_before, waveform_before = database.read_bytes(), waveform.read_bytes()
    report = run_evaluation(database, tmp_path / "output", measurements_only=True)
    assert report["summary"] == {"measured": 1, "errors": 0}
    sidecar = json.loads((tmp_path / "output/measurements/e.json").read_text(encoding="utf-8"))
    assert sidecar["start_sample"] == 3000
    assert sidecar["measurements"]["schema_version"] == "2"
    assert sidecar["measurements"]["values"]["snr_db"] > 10
    assert database.read_bytes() == database_before
    assert waveform.read_bytes() == waveform_before


def test_changed_waveform_is_reported_as_error_without_sidecar(tmp_path):
    database, waveform = _waveform_database(tmp_path)
    with waveform.open("ab") as stream:
        stream.write(b"changed")
    report = run_evaluation(database, tmp_path / "output", measurements_only=True)
    assert report["summary"] == {"measured": 0, "errors": 1}
    assert "source_changed" in report["results"][0]["error"]
    assert not (tmp_path / "output/measurements/e.json").exists()


def test_paired_replay_uses_both_real_detectors_and_keeps_database_read_only(tmp_path):
    database, waveform = _waveform_database(tmp_path)
    before = database.read_bytes(), waveform.read_bytes()
    report = run_evaluation(database, tmp_path / "output", background_limit=0)
    assert report["summary"]["errors"] == 0
    row = report["results"][0]
    assert row["v2"]["matched"] is True
    assert row["baseline"]["elapsed_s"] > 0
    assert row["v2"]["diagnostics"]["algorithm_version"]
    assert {"without_stalta", "without_energy", "without_coherence", "without_trajectory",
            "without_amplitude", "without_spatial",
            "common_mode_removed"} == set(row["ablations"])
    assert report["common_mode_ablation_event_ids"] == ["e"]
    assert set(report["source_code_sha256"]) == {
        "screening.py", "measurements.py", "screening_evaluation.py", "evaluation_legacy.py",
    }
    assert all(len(digest) == 64 for digest in report["source_code_sha256"].values())
    assert (database.read_bytes(), waveform.read_bytes()) == before


def test_background_failure_does_not_erase_successful_event(tmp_path, monkeypatch):
    import das_sentinel.screening_evaluation as evaluation
    database, _ = _waveform_database(tmp_path)
    monkeypatch.setattr(evaluation, "_background_window", lambda *args: (0, 2000))
    original = evaluation._replay_window

    def fail_background(block, profile, row, *args, **kwargs):
        if row["kind"] == "background":
            raise ValueError("synthetic_background_read_failure")
        return original(block, profile, row, *args, **kwargs)

    monkeypatch.setattr(evaluation, "_replay_window", fail_background)
    report = run_evaluation(database, tmp_path / "output")
    assert report["summary"]["errors"] == 1
    event = next(row for row in report["results"] if row["kind"] == "event")
    assert event["status"] == "ok"
    assert (tmp_path / "output/measurements/e.json").exists()


def test_explicit_selection_is_checked_against_frozen_population(tmp_path):
    database, _ = _waveform_database(tmp_path)
    for event_ids in (["missing"], ["e", "e"]):
        with pytest.raises(ValueError, match="selection"):
            run_evaluation(database, tmp_path / "output", selected_event_ids=event_ids)
    report = run_evaluation(database, tmp_path / "output", selected_event_ids=[])
    assert report["selection_count"] == 0
    assert report["requested_event_ids"] == []
    assert report["full_selection_count"] == 1


def test_background_read_uses_same_memory_budget_as_event(tmp_path, monkeypatch):
    import das_sentinel.screening_evaluation as evaluation
    database, _ = _waveform_database(tmp_path)
    monkeypatch.setattr(evaluation, "_background_window", lambda *args: (0, 2000))
    checks = []

    def budget(start, stop, channel_count):
        checks.append((start, stop, channel_count))
        if len(checks) == 2:
            raise ValueError("window_exceeds_512MiB_read_budget")

    monkeypatch.setattr(evaluation, "_check_read_budget", budget)
    report = run_evaluation(database, tmp_path / "output")
    assert len(checks) == 2
    assert report["summary"]["errors"] == 1
    assert next(r for r in report["results"] if r["kind"] == "event")["status"] == "ok"


def test_paired_detector_branches_receive_identical_full_window(tmp_path, monkeypatch):
    from das_sentinel import evaluation_legacy, screening
    from das_sentinel.config import ProcessingProfile
    from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
    from das_sentinel.screening_evaluation import _replay_window
    database, path = _waveform_database(tmp_path)
    block = ZdHdf5Reader().read_block(path, 500, 5000)
    seen = {}

    def baseline(name, source, profile, backend):
        seen[name] = (source.sample_offset, source.data.shape[0])
        return []

    def current(source, *args):
        seen["v2"] = (source.sample_offset, source.data.shape[0])
        return screening.ScreeningResult((), (), {})

    monkeypatch.setattr(evaluation_legacy, "_run_detector", baseline)
    monkeypatch.setattr(screening, "screen_block", current)
    _replay_window(block, ProcessingProfile(), {**snapshot_events(database)[0], "kind": "event"},
                   2000, 5000, common_mode_ablation=False)
    assert set(seen) == {"stalta", "energy", "coherence", "trajectory", "v2"}
    assert all(window == (500, 4500) for window in seen.values())
