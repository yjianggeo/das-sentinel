from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
from das_sentinel.services.discovery import fingerprint
from das_sentinel.services.waveform_manifest import create_manifest


def _source(
    path: Path,
    *,
    start: datetime,
    seconds: int = 2_000,
    fs: float = 10.0,
    distances: np.ndarray | None = None,
) -> None:
    grid = np.asarray(distances if distances is not None else [8.0, 12.0, 16.0])
    count = round(seconds * fs)
    times = start.timestamp() * 1_000_000 + np.arange(count) * 1_000_000 / fs
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", shape=(count, grid.size), dtype="f4")
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = fs
        raw.attrs["SpatialSamplingInterval"] = grid[1] - grid[0]
        raw.attrs["StartLocusIndex"] = int(grid[0] / (grid[1] - grid[0]))
        raw.attrs["RawDataUnit"] = "strain_rate"


def _database(path: Path, site_root: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sites (id TEXT PRIMARY KEY, name TEXT, data_path TEXT);
        CREATE TABLE source_files (
          id TEXT PRIMARY KEY, site_id TEXT, path TEXT, fingerprint TEXT
        );
        CREATE TABLE processing_runs (
          id TEXT PRIMARY KEY, source_file_id TEXT, source_fingerprint TEXT
        );
        CREATE TABLE events (
          id TEXT PRIMARY KEY, processing_run_id TEXT, start_sample INTEGER,
          end_sample INTEGER, channel_start INTEGER, channel_stop INTEGER,
          occurred_at TEXT, features_json TEXT
        );
        CREATE TABLE reviews (
          id INTEGER PRIMARY KEY, event_id TEXT, label TEXT, created_at TEXT
        );
        """
    )
    connection.execute(
        "INSERT INTO sites VALUES (?, ?, ?)", ("site", "JJK", str(site_root))
    )
    connection.commit()
    connection.close()


def _event(
    database: Path,
    source: Path,
    *,
    event_id: str,
    start_sample: int,
    end_sample: int,
    occurred_at: datetime,
    labels: tuple[tuple[int, str, datetime], ...],
    channel_start: int = 0,
    channel_stop: int = 2,
    run_fingerprint: str | None = None,
) -> None:
    source_id = f"source-{event_id}"
    run_id = f"run-{event_id}"
    current = fingerprint(source)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO source_files VALUES (?, 'site', ?, ?)",
        (source_id, str(source), current),
    )
    connection.execute(
        "INSERT INTO processing_runs VALUES (?, ?, ?)",
        (run_id, source_id, run_fingerprint or current),
    )
    connection.execute(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            run_id,
            start_sample,
            end_sample,
            channel_start,
            channel_stop,
            occurred_at.isoformat().replace("+00:00", "Z"),
            json.dumps({"duration_seconds": 3.0, "detector": "energy"}),
        ),
    )
    connection.executemany(
        "INSERT INTO reviews VALUES (?, ?, ?, ?)",
        [
            (review_id, event_id, label, reviewed.isoformat().replace("+00:00", "Z"))
            for review_id, label, reviewed in labels
        ],
    )
    connection.commit()
    connection.close()


def test_manifest_uses_latest_human_review_and_native_hdf5_coordinates(tmp_path):
    database = tmp_path / "sentinel.db"
    _database(database, tmp_path)
    source = tmp_path / "source.h5"
    source_start = datetime(2026, 9, 13, 23, 40, tzinfo=UTC)
    _source(source, start=source_start)
    _event(
        database,
        source,
        event_id="event-b",
        start_sample=120,
        end_sample=150,
        occurred_at=source_start + timedelta(seconds=12.05),
        labels=(
            (1, "noise", datetime(2026, 9, 14, 1, tzinfo=UTC)),
            (2, "blast", datetime(2026, 9, 14, 2, tzinfo=UTC)),
        ),
    )

    before = database.read_bytes()
    manifest = create_manifest(database, "site")

    assert database.read_bytes() == before
    assert manifest["schema_version"] == 1
    assert manifest["site_id"] == "site"
    assert manifest["site_name"] == "JJK"
    assert manifest["excluded"] == []
    assert len(manifest["targets"]) == 1
    target = manifest["targets"][0]
    assert target == {
        "event_id": "event-b",
        "review_id": 2,
        "label": "blast",
        "reviewed_at": "2026-09-14T02:00:00Z",
        "source_path": str(source),
        "source_fingerprint": fingerprint(source),
        "source_sample_rate_hz": 10.0,
        "source_unit": "strain_rate",
        "source_grid_sha256": __import__("hashlib")
        .sha256(np.asarray([8.0, 12.0, 16.0], dtype="<f8").tobytes())
        .hexdigest(),
        "start_utc": "2026-09-13T23:40:12Z",
        "end_utc": "2026-09-13T23:40:15Z",
        "channel_start": 0,
        "channel_stop": 2,
        "group": "2026-09-13",
        "legacy_features": {"duration_seconds": 3.0, "detector": "energy"},
    }


def test_latest_unknown_review_and_invalid_source_bindings_are_excluded(tmp_path):
    database = tmp_path / "sentinel.db"
    _database(database, tmp_path)
    start = datetime(2026, 9, 14, tzinfo=UTC)
    unknown_source = tmp_path / "unknown.h5"
    mismatch_source = tmp_path / "mismatch.h5"
    bad_roi_source = tmp_path / "bad-roi.h5"
    for source in (unknown_source, mismatch_source, bad_roi_source):
        _source(source, start=start)
    _event(
        database,
        unknown_source,
        event_id="unknown",
        start_sample=10,
        end_sample=20,
        occurred_at=start + timedelta(seconds=1),
        labels=((1, "noise", start), (2, "unknown", start + timedelta(seconds=1))),
    )
    _event(
        database,
        mismatch_source,
        event_id="mismatch",
        start_sample=10,
        end_sample=20,
        occurred_at=start + timedelta(seconds=1),
        labels=((3, "traffic", start),),
        run_fingerprint="a" * 64,
    )
    _event(
        database,
        bad_roi_source,
        event_id="bad-roi",
        start_sample=10,
        end_sample=20,
        occurred_at=start + timedelta(seconds=1),
        labels=((4, "earthquake", start),),
        channel_stop=4,
    )

    manifest = create_manifest(database, "site")

    assert manifest["targets"] == []
    reasons = {item["event_id"]: item["reason"] for item in manifest["excluded"]}
    assert "four supervised labels" in reasons["unknown"]
    assert "fingerprint" in reasons["mismatch"]
    assert "channel" in reasons["bad-roi"]
    assert all("start_utc" not in item for item in manifest["excluded"])


def test_manifest_groups_cross_midnight_overlapping_support_before_selection(tmp_path):
    database = tmp_path / "sentinel.db"
    _database(database, tmp_path)
    source = tmp_path / "long.h5"
    source_start = datetime(2026, 9, 14, 23, 30, tzinfo=UTC)
    _source(source, start=source_start, seconds=100_000, fs=1.0)
    reviewed = datetime(2026, 9, 18, tzinfo=UTC)
    for event_id, offset in (
        ("late", 29 * 60),
        ("early", 31 * 60),
        ("next-day", 25 * 3600),
    ):
        _event(
            database,
            source,
            event_id=event_id,
            start_sample=offset,
            end_sample=offset + 2,
            occurred_at=source_start + timedelta(seconds=offset),
            labels=((offset, "noise", reviewed),),
        )

    targets = create_manifest(database, "site")["targets"]
    groups = {target["event_id"]: target["group"] for target in targets}

    assert groups["late"] == groups["early"]
    assert groups["next-day"] != groups["early"]


def test_manifest_merges_touching_read_support_but_not_support_beyond_neighbors(
    tmp_path,
):
    database = tmp_path / "sentinel.db"
    _database(database, tmp_path)
    source = tmp_path / "two-days.h5"
    source_start = datetime(2026, 9, 14, 23, 30, tzinfo=UTC)
    _source(source, start=source_start, seconds=100_000, fs=5.0)
    reviewed = datetime(2026, 9, 18, tzinfo=UTC)
    centers = {
        "touch-left": datetime(2026, 9, 14, 23, 49, 50, tzinfo=UTC),
        "touch-right": datetime(2026, 9, 15, 0, 10, 10, tzinfo=UTC),
        "apart-left": datetime(2026, 9, 15, 23, 49, 48, tzinfo=UTC),
        "apart-right": datetime(2026, 9, 16, 0, 10, 12, tzinfo=UTC),
    }
    for review_id, (event_id, center) in enumerate(centers.items(), start=10):
        center_sample = round((center - source_start).total_seconds() * 5)
        _event(
            database,
            source,
            event_id=event_id,
            start_sample=center_sample - 1,
            end_sample=center_sample + 1,
            occurred_at=source_start + timedelta(seconds=(center_sample - 1) / 5),
            labels=((review_id, "noise", reviewed),),
        )

    groups = {
        target["event_id"]: target["group"]
        for target in create_manifest(database, "site")["targets"]
    }

    assert groups["touch-left"] == groups["touch-right"]
    assert groups["apart-left"] != groups["apart-right"]


def test_metadata_failure_excludes_only_affected_event(tmp_path):
    database = tmp_path / "sentinel.db"
    _database(database, tmp_path)
    start = datetime(2026, 9, 14, tzinfo=UTC)
    good = tmp_path / "good.h5"
    broken = tmp_path / "broken.h5"
    _source(good, start=start)
    broken.write_bytes(b"not hdf5")
    for event_id, source in (("good", good), ("broken", broken)):
        _event(
            database,
            source,
            event_id=event_id,
            start_sample=10,
            end_sample=20,
            occurred_at=start + timedelta(seconds=1),
            labels=((1 if event_id == "good" else 2, "traffic", start),),
        )

    manifest = create_manifest(database, "site")

    assert [target["event_id"] for target in manifest["targets"]] == ["good"]
    assert manifest["excluded"][0]["event_id"] == "broken"
    assert "metadata" in manifest["excluded"][0]["reason"]
