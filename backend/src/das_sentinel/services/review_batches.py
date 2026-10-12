from __future__ import annotations

import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from numbers import Real
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from das_sentinel.persistence.models import (
    SUPERVISED_LABELS,
    Classification,
    DetectionEvidence,
    Event,
    ProcessingRun,
    Review,
    Site,
    SourceFile,
)

# Signal measurements only: no clock, path, machine label, or site context.
_FEATURES = (
    "duration_s",
    "rise_time_s",
    "dominant_frequency_hz",
    "bandwidth_hz",
    "band_energy_ratio_low_mid",
    "band_energy_ratio_mid_high",
    "spectral_entropy",
    "continuity",
    "coherence",
    "rms",
    "kurtosis",
    "peak_median_ratio",
    "cross_channel_correlation",
    "repetition_rate_hz",
)
_MIN_COMMON = 2


@dataclass
class _Candidate:
    event: Event
    compatibility: tuple[str, str, int]
    directory: str
    summary: dict[str, Any]
    label: str | None
    vector: np.ndarray
    end: datetime
    job_id: str | None = None
    score: float = 0.0
    preferred: bool = False
    earthquake: bool = False
    similar_id: str | None = None
    catalog_window: dict | None = None
    reasons: list[str] = field(default_factory=list)


def _features(raw: str) -> tuple[np.ndarray, float, bool]:
    try:
        values = json.loads(raw)
    except (ValueError, TypeError):
        values = {}
    if not isinstance(values, dict):
        values = {}
    vector = np.full(len(_FEATURES), np.nan)
    invalid = False
    duration = 0.0
    for index, name in enumerate(_FEATURES):
        value = values.get(name)
        if not isinstance(value, Real) or isinstance(value, bool):
            invalid |= name in values
            continue
        try:
            numeric = float(value)
        except OverflowError:
            invalid = True
            continue
        if not math.isfinite(numeric):
            invalid = True
            continue
        # Log compression limits amplitude/ratio outliers before robust scaling.
        vector[index] = math.copysign(math.log1p(abs(numeric)), numeric)
        if name == "duration_s" and 0 <= numeric <= 86400:
            duration = numeric
    return vector, duration, invalid or np.isfinite(vector).sum() < _MIN_COMMON


def _catalog(app_data_dir: Path, site_id: str) -> tuple[list[dict], list[str]]:
    try:
        payload = json.loads(
            (app_data_dir / "review-catalog-windows.json").read_text(
                encoding="utf-8-sig"
            )
        )
    except FileNotFoundError:
        return [], ["未提供目录到时窗口；继续按信号相似性与抽查安排复核。"]
    except (OSError, ValueError):
        return [], ["目录到时窗口无法读取；继续保留所有候选。"]
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("windows"), list)
    ):
        return [], ["目录到时窗口格式无效；继续保留所有候选。"]
    unique_windows = {}
    invalid = 0
    for item in payload["windows"]:
        if not isinstance(item, dict):
            invalid += 1
            continue
        if item.get("site_id") != site_id:
            continue
        try:
            start = datetime.fromisoformat(item["start_utc"])
            end = datetime.fromisoformat(item["end_utc"])
            if start.utcoffset() is None or end.utcoffset() is None or end <= start:
                raise ValueError("invalid UTC interval")
            if not all(
                isinstance(item.get(key), str) and item[key].strip()
                for key in ("catalog_id", "phase", "source_url")
            ):
                raise ValueError("missing provenance")
            preferred_jobs = item.get("preferred_job_ids", [])
            if not isinstance(preferred_jobs, list) or not all(
                isinstance(job_id, str) and job_id.strip() for job_id in preferred_jobs
            ):
                raise ValueError("invalid preferred job scope")
        except (ValueError, TypeError, KeyError):
            invalid += 1
            continue
        start, end = start.astimezone(UTC), end.astimezone(UTC)
        key = (
            site_id, item["catalog_id"], item["phase"],
            start.isoformat(), end.isoformat(),
        )
        if key in unique_windows:
            window = unique_windows[key]
            window["preferred_job_ids"] |= frozenset(preferred_jobs)
            window["source_url"] = min(window["source_url"], item["source_url"])
            continue
        unique_windows[key] = {
            **item,
            "id": "review-window-" + hashlib.sha256(
                json.dumps(key, ensure_ascii=False).encode()
            ).hexdigest()[:24],
            "start": start,
            "end": end,
            "start_utc": start.isoformat().replace("+00:00", "Z"),
            "end_utc": end.isoformat().replace("+00:00", "Z"),
            "preferred_job_ids": frozenset(preferred_jobs),
        }
    windows = list(unique_windows.values())
    windows.sort(
        key=lambda item: (item["start"], item["end"], item["catalog_id"], item["phase"])
    )
    warnings = (
        [f"忽略了 {invalid} 条无效目录窗口（需含时区及来源）。"] if invalid else []
    )
    if not windows:
        warnings.append("当前站点没有可用目录窗口；继续保留所有候选。")
    return windows, warnings


def _load(session: Session, site: Site) -> tuple[list[_Candidate], int]:
    latest_review = (
        select(Review.label)
        .where(Review.event_id == Event.id)
        .order_by(Review.created_at.desc(), Review.id.desc())
        .limit(1)
        .correlate(Event)
        .scalar_subquery()
    )
    machine_label = (
        select(Classification.label)
        .where(Classification.event_id == Event.id)
        .order_by(Classification.created_at.desc(), Classification.id.desc())
        .limit(1)
        .correlate(Event)
        .scalar_subquery()
    )
    score = (
        select(func.max(DetectionEvidence.score))
        .where(DetectionEvidence.event_id == Event.id)
        .correlate(Event)
        .scalar_subquery()
    )
    statement = (
        select(
            Event, ProcessingRun, SourceFile.path, latest_review, machine_label, score
        )
        .join(ProcessingRun, Event.processing_run_id == ProcessingRun.id)
        .join(SourceFile, ProcessingRun.source_file_id == SourceFile.id)
        .where(SourceFile.site_id == site.id)
        .order_by(Event.occurred_at, Event.id)
    )
    candidates = []
    invalid = 0
    for event, run, path, label, machine, detection_score in session.execute(statement):
        vector, duration, bad = _features(event.features_json)
        invalid += bad
        directory = (
            str(PureWindowsPath(path).parent).casefold()
            if "\\" in path
            else str(PurePosixPath(path).parent)
        )
        summary = {
            "id": event.id,
            "event_uid": event.event_uid,
            "site_id": site.id,
            "site_timezone": site.timezone,
            "occurred_at": event.occurred_at.isoformat().replace("+00:00", "Z"),
            "start_sample": event.start_sample,
            "end_sample": event.end_sample,
            "channel_start": event.channel_start,
            "channel_stop": event.channel_stop,
            "machine_label": machine or "unknown",
            "detection_score": detection_score
            if detection_score is not None and math.isfinite(detection_score)
            else None,
            "latest_review": None,
        }
        candidates.append(
            _Candidate(
                event,
                (run.pipeline_version, run.processing_profile_id, run.profile_version),
                directory,
                summary,
                label,
                vector,
                event.occurred_at + timedelta(seconds=duration),
                job_id=run.job_id,
            )
        )
    return candidates, invalid


def _scale(rows: list[_Candidate]) -> None:
    matrix = np.vstack([row.vector for row in rows])
    for index in range(len(_FEATURES)):
        column = matrix[:, index]
        finite = column[np.isfinite(column)]
        if not finite.size:
            continue
        median = float(np.median(finite))
        low, high = np.percentile(finite, [25, 75])
        scale = max(float(high - low), 0.1)
        matrix[:, index] = np.clip((column - median) / scale, -20, 20)
    for row, vector in zip(rows, matrix, strict=True):
        row.vector = vector


def _distances(vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    common = np.isfinite(vector) & np.isfinite(matrix)
    count = common.sum(axis=1)
    squared = np.where(common, matrix - vector, 0.0) ** 2
    result = np.sqrt(squared.sum(axis=1) / np.maximum(count, 1))
    result[count < _MIN_COMMON] = np.inf
    return result


def _rank(rows: list[_Candidate], windows: list[dict]) -> int:
    references = {}
    for label in ("earthquake", "noise"):
        peers = sorted(
            (row for row in rows if row.label == label), key=lambda row: row.event.id
        )
        references[label] = (
            peers,
            np.vstack([row.vector for row in peers]) if peers else None,
        )
    no_reference = 0
    for row in rows:
        if row.label is not None:
            continue
        nearest = {}
        for label, (peers, matrix) in references.items():
            nearest[label] = math.inf
            if matrix is None:
                continue
            distances = _distances(row.vector, matrix)
            index = int(np.argmin(distances))
            distance = float(distances[index])
            nearest[label] = distance
            if math.isfinite(distance) and label == "earthquake":
                row.similar_id = peers[index].event.id
        eq_distance, noise_distance = nearest["earthquake"], nearest["noise"]
        if math.isfinite(eq_distance):
            row.score += 60 / (1 + eq_distance)
            row.reasons.append(
                f"与同版本人工确认地震的信号距离 {eq_distance:.2f}（越小越相近）。"
            )
            row.earthquake = eq_distance <= 1.5 and eq_distance < noise_distance
        else:
            no_reference += 1
            row.reasons.append("暂无可比较的同版本人工地震参考；保留复核机会。")
        if math.isfinite(noise_distance):
            row.score -= 30 / (1 + noise_distance)
            row.reasons.append(
                f"与同版本人工确认噪声的信号距离 {noise_distance:.2f}；仅影响顺序。"
            )
        if np.isfinite(row.vector).sum() < _MIN_COMMON:
            row.reasons.append("有效信号特征不足；不作相似判断，保留独立抽查。")
        hits = [
            item
            for item in windows
            if row.event.occurred_at <= item["end"] and row.end >= item["start"]
        ]
        if hits:
            # Sorted window order breaks ties after own-job preference and overlap.
            window = min(
                hits,
                key=lambda item: (
                    row.job_id not in item["preferred_job_ids"],
                    -(min(row.end, item["end"]) - max(
                        row.event.occurred_at, item["start"]
                    )).total_seconds(),
                ),
            )
            row.catalog_window = window
            row.score += 40
            row.earthquake = True
            if row.job_id in window["preferred_job_ids"]:
                row.preferred = True
                row.score += 100
                row.reasons.append("本批目录优先：本次新增任务且与地震到时窗口相交，优先于此前数据复核。")
            row.reasons.append(
                f"目录到时窗口相交：{window['catalog_id']} {window['phase']}；仅作线索，来源 {window['source_url']}"
            )
    return no_reference


def _group(rows: list[_Candidate]) -> list[list[_Candidate]]:
    groups: list[list[_Candidate]] = []
    window_groups: dict[str, list[_Candidate]] = {}
    active: dict[tuple, deque] = defaultdict(deque)
    for row in rows:
        if row.label is not None:
            continue
        if row.catalog_window is not None:
            key = row.catalog_window["id"]
            if key not in window_groups:
                window_groups[key] = []
                groups.append(window_groups[key])
            window_groups[key].append(row)
            continue
        key = (*row.compatibility, row.directory, row.preferred)
        pool = active[key]
        while (
            pool
            and (row.event.occurred_at - pool[0][0].event.occurred_at).total_seconds()
            > 120
        ):
            pool.popleft()
        match = None
        for group in pool:
            anchor = group[0]
            if (
                len(group) >= 20
                or (
                    max(row.end, *(member.end for member in group))
                    - anchor.event.occurred_at
                ).total_seconds()
                > 120
            ):
                continue
            if max(
                row.event.channel_start,
                *(member.event.channel_start for member in group),
            ) >= min(
                row.event.channel_stop, *(member.event.channel_stop for member in group)
            ):
                continue
            if _distances(row.vector, anchor.vector.reshape(1, -1))[0] <= 0.6:
                match = group
                break
        if match is None:
            match = []
            groups.append(match)
            pool.append(match)
        match.append(row)
    return groups


def _representatives(
    members: list[_Candidate], representative: _Candidate,
) -> list[dict[str, Any]]:
    selected = [representative]
    if representative.catalog_window is None:
        return [representative.summary]
    start = min(row.event.occurred_at for row in members)
    time_span = max((max(row.end for row in members) - start).total_seconds(), 1)
    channel_start = min(row.event.channel_start for row in members)
    channel_span = max(
        max(row.event.channel_stop for row in members) - channel_start, 1,
    )
    # Coverage endpoints only; this is navigation diversity, not signal similarity.
    coverage = {
        row.event.id: (
            (row.event.occurred_at - start).total_seconds() / time_span,
            (row.end - start).total_seconds() / time_span,
            (row.event.channel_start - channel_start) / channel_span,
            (row.event.channel_stop - channel_start) / channel_span,
        )
        for row in members
    }
    remaining = [row for row in members if row is not representative]
    while remaining and len(selected) < 3:
        def distance(row: _Candidate) -> float:
            return min(
                sum((a - b) ** 2 for a, b in zip(
                    coverage[row.event.id], coverage[chosen.event.id], strict=True,
                ))
                for chosen in selected
            )

        candidate = min(
            remaining,
            key=lambda row: (
                -distance(row), -row.score, row.event.occurred_at, row.event.id,
            ),
        )
        if distance(candidate) == 0:
            break
        selected.append(candidate)
        remaining = [row for row in remaining if row is not candidate]
    return [row.summary for row in selected]


def _ordered(groups: list[list[_Candidate]]) -> list[dict[str, Any]]:
    # Build the complete order before slicing. Hashes are deterministic across processes.
    records = {}
    buckets = defaultdict(deque)
    preferred_ids = set()
    for members in groups:
        representative = min(
            members, key=lambda row: (-row.score, row.event.occurred_at, row.event.id)
        )
        window = representative.catalog_window
        group_id = window["id"] if window is not None else (
            "review-"
            + hashlib.sha256(
                "\n".join(row.event.id for row in members).encode()
            ).hexdigest()[:24]
        )
        if any(row.preferred for row in members):
            preferred_ids.add(group_id)
        bucket = str(
            (
                representative.compatibility,
                tuple(
                    math.floor(representative.vector[index])
                    if math.isfinite(representative.vector[index])
                    else None
                    for index in (0, 2, 8, 9)
                ),
            )
        )
        reasons = list(representative.reasons)
        if window is not None:
            reasons.append(
                f"同一目录到时窗口内 {len(members)} 个片段共用查阅入口；逐条复核，不等同同一已确认地震。"
            )
        elif len(members) > 1:
            reasons.append(
                f"{len(members)} 个近时、空间重叠且信号相近片段；逐条复核，不等同独立地震。"
            )
        record = {
            "id": group_id,
            "representative": representative.summary,
            "representatives": _representatives(members, representative),
            "catalog_window": {
                key: window[key]
                for key in ("catalog_id", "phase", "start_utc", "end_utc", "source_url")
            } if window is not None else None,
            "members": [row.summary for row in members],
            "start_utc": min(row.event.occurred_at for row in members)
            .isoformat()
            .replace("+00:00", "Z"),
            "end_utc": max(row.end for row in members)
            .isoformat()
            .replace("+00:00", "Z"),
            "priority_score": round(representative.score, 6),
            "reasons": reasons,
            "selection_kind": "diverse",
            "similar_earthquake_event_id": representative.similar_id,
        }
        sort_key = (-representative.score, representative.event.occurred_at, group_id)
        records[group_id] = (
            record,
            bucket,
            sort_key,
            any(row.earthquake for row in members),
        )
    ranked = sorted(records, key=lambda key: records[key][2])
    earthquakes = deque(key for key in ranked if records[key][3])
    audit = deque(
        sorted(
            records,
            key=lambda key: hashlib.sha256(
                ("review-audit-v1:" + key).encode()
            ).digest(),
        )
    )
    for key in ranked:
        buckets[records[key][1]].append(key)
    counts: Counter = Counter()
    selected = set()
    ordered = []

    def choose(queue: deque, kind: str) -> bool:
        while queue and queue[0] in selected:
            queue.popleft()
        if not queue:
            return False
        key = queue.popleft()
        selected.add(key)
        record, bucket, _, _ = records[key]
        counts[bucket] += 1
        record["selection_kind"] = kind
        if kind == "audit":
            record["reasons"].append("固定种子抽查；机器标签不影响保留资格。")
        elif kind == "diverse":
            record["reasons"].append("补充不同信号形态，减少连续复核相似片段。")
        ordered.append(record)
        return True

    # Lazy heap refresh accounts for earthquake/audit selections in each bucket.
    heap = [(0, records[queue[0]][2], bucket) for bucket, queue in buckets.items()]
    heapq.heapify(heap)

    def diverse() -> bool:
        while heap:
            count, rank, bucket = heapq.heappop(heap)
            queue = buckets[bucket]
            while queue and queue[0] in selected:
                queue.popleft()
            if not queue:
                continue
            current = (counts[bucket], records[queue[0]][2], bucket)
            if (count, rank, bucket) != current:
                heapq.heappush(heap, current)
                continue
            choose(queue, "diverse")
            if queue:
                heapq.heappush(heap, (counts[bucket], records[queue[0]][2], bucket))
            return True
        return False

    # Complete the explicitly requested campaign first; score alone would still
    # interleave earlier data through diversity and audit quotas.
    preferred = deque(key for key in ranked if key in preferred_ids)
    while choose(preferred, "earthquake"):
        pass

    while len(selected) < len(records):
        target = min(len(records), len(ordered) + 20)
        for _ in range(10):
            if not choose(earthquakes, "earthquake"):
                break
        for _ in range(min(6, target - len(ordered))):
            diverse()
        for _ in range(min(4, target - len(ordered))):
            choose(audit, "audit")
        while len(ordered) < target:
            diverse()
    return ordered


def get_review_batch(
    session: Session,
    app_data_dir: Path,
    *,
    site_id: str,
    offset: int = 0,
    limit: int = 20,
) -> dict[str, Any]:
    """Read-only review ordering; no classifier fitting or raw-file access."""
    if offset < 0 or not 1 <= limit <= 50:
        raise ValueError("invalid review batch pagination")
    site = session.get(Site, site_id)
    if site is None:
        raise LookupError("Site not found")
    rows, invalid = _load(session, site)
    windows, warnings = _catalog(app_data_dir, site_id)
    warnings.insert(
        0,
        "推荐分仅表示查阅顺序，不是地震概率；相似组不等同独立地震，也不自动继承标签。",
    )
    if invalid:
        warnings.append(
            f"{invalid} 条事件的信号特征不足或含无效值；未复核候选全部保留。"
        )
    compatible = defaultdict(list)
    for row in rows:
        compatible[row.compatibility].append(row)
    no_reference = 0
    for peers in compatible.values():
        _scale(peers)
        no_reference += _rank(peers, windows)
    if no_reference:
        warnings.append(f"{no_reference} 条未复核事件没有可比较的同版本人工地震参考。")
    reference_counts = {
        label: sum(row.label == label for row in rows) for label in SUPERVISED_LABELS
    }
    ordered = _ordered(_group(rows))
    return {
        "site_id": site_id,
        "total_unreviewed": sum(row.label is None for row in rows),
        "total_groups": len(ordered),
        "offset": offset,
        "limit": limit,
        "reference_counts": reference_counts,
        "catalog_window_count": len(windows),
        "warnings": warnings,
        "groups": ordered[offset : offset + limit],
    }
