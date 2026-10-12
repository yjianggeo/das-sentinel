from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import fingerprint

_LABELS = frozenset(("earthquake", "blast", "traffic", "noise"))
_LOW_FREQUENCY_STEP_US = 200_000
_NATIVE_NEIGHBOR_CUSHION = timedelta(seconds=1)


def create_manifest(database: Path, site_id: str) -> dict[str, Any]:
    """Create a deterministic target snapshot without opening SQLite for writes."""

    path = Path(database).resolve(strict=True)
    uri = f"file:{quote(path.as_posix(), safe='/:')}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        site = connection.execute(
            "SELECT id, name, data_path FROM sites WHERE id = ?", (site_id,)
        ).fetchone()
        if site is None:
            raise LookupError(f"site not found: {site_id}")
        rows = connection.execute(
            """
            WITH latest_reviews AS (
              SELECT id, event_id, label, created_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY event_id ORDER BY created_at DESC, id DESC
                     ) AS position
              FROM reviews
            )
            SELECT e.id AS event_id, e.start_sample, e.end_sample,
                   e.channel_start, e.channel_stop, e.occurred_at, e.features_json,
                   r.id AS review_id, r.label, r.created_at AS reviewed_at,
                   s.path AS source_path, p.source_fingerprint
            FROM latest_reviews AS r
            JOIN events AS e ON e.id = r.event_id
            JOIN processing_runs AS p ON p.id = e.processing_run_id
            JOIN source_files AS s ON s.id = p.source_file_id
            WHERE r.position = 1 AND s.site_id = ?
            ORDER BY e.id
            """,
            (site_id,),
        ).fetchall()

        reader = ZdHdf5Reader()
        metadata_cache: dict[Path, tuple[Any, int] | Exception] = {}
        targets: list[dict[str, Any]] = []
        excluded: list[dict[str, str]] = []
        for row in rows:
            event_id = str(row["event_id"])
            label = str(row["label"])
            if label not in _LABELS:
                excluded.append(
                    {
                        "event_id": event_id,
                        "reason": "latest review is not one of the four supervised labels",
                    }
                )
                continue
            try:
                source_path = _source_path(row["source_path"], site["data_path"])
                expected_fingerprint = str(row["source_fingerprint"])
                if fingerprint(source_path) != expected_fingerprint:
                    raise ValueError("source fingerprint does not match processing run")
                cached = metadata_cache.get(source_path)
                if cached is None:
                    try:
                        cached = (
                            reader.read_metadata(source_path),
                            reader.sample_count(source_path),
                        )
                    except (OSError, TypeError, ValueError) as error:
                        cached = error
                    metadata_cache[source_path] = cached
                if isinstance(cached, Exception):
                    raise TypeError("source metadata is unavailable") from cached
                metadata, sample_count = cached
                if fingerprint(source_path) != expected_fingerprint:
                    raise ValueError("source fingerprint changed while reading metadata")
                target = _target(
                    row, source_path, expected_fingerprint, metadata, sample_count
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                excluded.append({"event_id": event_id, "reason": str(error)})
                continue
            targets.append(target)

        _assign_groups(targets)
        created_at = (
            datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        )
        return {
            "schema_version": 1,
            "site_id": str(site["id"]),
            "site_name": str(site["name"]),
            "created_at": created_at,
            "targets": targets,
            "excluded": excluded,
        }
    finally:
        connection.close()


def _source_path(stored: object, site_root: object) -> Path:
    path = Path(str(stored))
    if not path.is_absolute():
        path = Path(str(site_root)) / path
    return path.resolve(strict=True)


def _target(
    row: sqlite3.Row,
    source_path: Path,
    source_fingerprint: str,
    metadata: Any,
    sample_count: int,
) -> dict[str, Any]:
    sample_rate = float(metadata.sample_rate_hz)
    distances = np.asarray(metadata.channel_distance_m)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("source metadata sample rate is invalid")
    if distances.ndim != 1 or distances.size == 0 or not np.isfinite(distances).all():
        raise ValueError("source metadata channel grid is invalid")
    start_sample = int(row["start_sample"])
    end_sample = int(row["end_sample"])
    channel_start = int(row["channel_start"])
    channel_stop = int(row["channel_stop"])
    if start_sample < 0 or end_sample <= start_sample or end_sample > sample_count:
        raise ValueError("event sample bounds are outside source metadata")
    if (
        channel_start < 0
        or channel_stop <= channel_start
        or channel_stop > distances.size
    ):
        raise ValueError("event channel bounds are outside source grid")
    source_start = _utc(metadata.start_time_utc)
    start = source_start + timedelta(seconds=start_sample / sample_rate)
    end = source_start + timedelta(seconds=end_sample / sample_rate)
    occurred = _utc(row["occurred_at"])
    if abs((occurred - start).total_seconds()) > 1.0 / sample_rate + 1e-9:
        raise ValueError("event occurred_at disagrees with native sample bounds")
    try:
        features = json.loads(str(row["features_json"]))
    except json.JSONDecodeError:
        raise ValueError("legacy features are invalid JSON") from None
    if not isinstance(features, dict):
        raise TypeError("legacy features must be an object")
    grid = np.asarray(distances, dtype="<f8")
    return {
        "event_id": str(row["event_id"]),
        "review_id": int(row["review_id"]),
        "label": str(row["label"]),
        "reviewed_at": _stamp(_utc(row["reviewed_at"])),
        "source_path": str(source_path),
        "source_fingerprint": source_fingerprint,
        "source_sample_rate_hz": sample_rate,
        "source_unit": str(metadata.unit),
        "source_grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
        "start_utc": _stamp(start),
        "end_utc": _stamp(end),
        "channel_start": channel_start,
        "channel_stop": channel_stop,
        "group": "",
        "legacy_features": features,
    }


def _assign_groups(targets: list[dict[str, Any]]) -> None:
    days = sorted({_utc(target["start_utc"]).date().isoformat() for target in targets})
    parent = {day: day for day in days}

    def root(day: str) -> str:
        while parent[day] != day:
            parent[day] = parent[parent[day]]
            day = parent[day]
        return day

    def merge(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    intervals = []
    for target in targets:
        start = _utc(target["start_utc"])
        end = _utc(target["end_utc"])
        center = start + (end - start) / 2
        view_start = _floor_utc_grid(center - timedelta(seconds=300))
        intervals.append(
            (
                start.date().isoformat(),
                view_start - timedelta(seconds=310) - _NATIVE_NEIGHBOR_CUSHION,
                view_start + timedelta(seconds=910) + _NATIVE_NEIGHBOR_CUSHION,
            )
        )
    for index, (left_day, left_start, left_end) in enumerate(intervals):
        for right_day, right_start, right_end in intervals[index + 1 :]:
            if (
                left_day != right_day
                and left_start <= right_end
                and right_start <= left_end
            ):
                merge(left_day, right_day)
    for target in targets:
        target["group"] = root(_utc(target["start_utc"]).date().isoformat())


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _floor_utc_grid(value: datetime) -> datetime:
    utc_value = value.astimezone(UTC)
    return utc_value.replace(
        microsecond=utc_value.microsecond
        - utc_value.microsecond % _LOW_FREQUENCY_STEP_US
    )


def _stamp(value: datetime) -> str:
    return (
        value.isoformat(timespec="microseconds")
        .replace(".000000+00:00", "Z")
        .replace("+00:00", "Z")
    )
