"""Read-only, paired replay of historical DAS windows; never creates jobs or labels."""
from __future__ import annotations

import hashlib
import html
import json
import os
import sqlite3
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def snapshot_events(database_path: Path) -> list[dict[str, Any]]:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute("""
            WITH latest AS (
                SELECT event_id, label, ROW_NUMBER() OVER (
                    PARTITION BY event_id ORDER BY created_at DESC, id DESC
                ) AS position FROM reviews
            )
            SELECT e.id AS event_id, e.start_sample, e.end_sample,
                e.channel_start, e.channel_stop, e.occurred_at,
                r.source_file_id, r.processing_profile_id, r.source_fingerprint,
                s.path, s.size_bytes, s.mtime_ns, s.site_id,
                p.payload_json, site.timezone, latest.label
            FROM events e
            JOIN processing_runs r ON r.id=e.processing_run_id
            JOIN source_files s ON s.id=r.source_file_id
            JOIN processing_profiles p ON p.id=r.processing_profile_id
            JOIN sites site ON site.id=s.site_id
            LEFT JOIN latest ON latest.event_id=e.id AND latest.position=1
            ORDER BY e.occurred_at, e.id
        """).fetchall()
        detectors: dict[str, set[str]] = defaultdict(set)
        for row in connection.execute("SELECT event_id, detector FROM detection_evidence"):
            detectors[row["event_id"]].add(row["detector"])
        result = []
        for row in rows:
            item = dict(row)
            item["profile"] = json.loads(item.pop("payload_json"))
            item["date_group"] = item["occurred_at"][:10]
            item["detectors"] = sorted(detectors[item["event_id"]])
            result.append(item)
        return result


def _stratified(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], deque] = defaultdict(deque)
    for row in sorted(rows, key=lambda r: (r["occurred_at"], r["event_id"])):
        groups[(row["date_group"], row["source_file_id"])].append(row)
    ordered = [groups[key] for key in sorted(groups)]
    selected = []
    while len(selected) < limit and any(ordered):
        for group in ordered:
            if group and len(selected) < limit:
                selected.append(group.popleft())
    return selected


def select_samples(rows: list[dict[str, Any]], noise_limit: int = 16,
                   unreviewed_limit: int = 8) -> list[dict[str, Any]]:
    if noise_limit < 0 or unreviewed_limit < 0:
        raise ValueError("sample limits must be nonnegative")
    targets = [row for row in rows if row["label"] in {"earthquake", "traffic", "blast"}
               or set(row["detectors"]) == {"energy"}]
    targets += _stratified([row for row in rows if row["label"] == "noise"], noise_limit)
    targets += _stratified([row for row in rows if row["label"] is None], unreviewed_limit)
    unique = {row["event_id"]: row for row in targets}
    return sorted(unique.values(), key=lambda row: (row["source_file_id"], row["start_sample"], row["event_id"]))


def matches_reference(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    for low, high in (("start_sample", "end_sample"), ("channel_start", "channel_stop")):
        intersection = min(candidate[high], reference[high]) - max(candidate[low], reference[low])
        shorter = min(candidate[high] - candidate[low], reference[high] - reference[low])
        if shorter <= 0 or intersection <= 0 or intersection / shorter < 0.25:
            return False
    return True


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, dict[str, int]] = {}
    background = {"baseline": 0, "v2": 0}
    elapsed = {"baseline": 0.0, "v2": 0.0}
    counts = {"baseline": 0, "v2": 0}
    unique: dict[str, set[tuple]] = {"baseline": set(), "v2": set()}
    contributions: dict[str, dict[str, int]] = {}
    ablation_losses: dict[str, dict[str, int]] = {}
    errors = 0
    for row in results:
        if row["status"] != "ok":
            errors += 1
            continue
        for version in elapsed:
            elapsed[version] += row[version]["elapsed_s"]
            counts[version] += len(row[version]["events"])
            for event in row[version]["events"]:
                unique[version].add((row["source_file_id"], event["start_sample"], event["end_sample"],
                                     event["channel_start"], event["channel_stop"]))
            if row["kind"] == "background":
                background[version] += len(row[version]["events"])
        if row["kind"] == "event" and row["label"] is not None:
            label = labels.setdefault(row["label"], {"evaluated": 0, "baseline_matched": 0, "v2_matched": 0})
            label["evaluated"] += 1
            label["baseline_matched"] += int(row["baseline"]["matched"])
            label["v2_matched"] += int(row["v2"]["matched"])
            for name, variant in row.get("ablations", {}).items():
                if name.startswith("without_"):
                    losses = ablation_losses.setdefault(name, {})
                    losses.setdefault(row["label"], 0)
                    losses[row["label"]] += int(row["v2"]["matched"] and not variant["matched"])
        detector_counts: dict[str, int] = defaultdict(int)
        for candidate in row["v2"].get("candidates", []):
            detector_counts[candidate["detector"].split(":")[0]] += 1
        for name, count in detector_counts.items():
            contribution = contributions.setdefault(name, {"window_raw_candidates": 0, "windows_with_candidates": 0})
            contribution["window_raw_candidates"] += count
            contribution["windows_with_candidates"] += 1
    return {"labels": labels, "errors": errors, "window_candidate_counts": counts,
            "distinct_window_detections": {key: len(value) for key, value in unique.items()},
            "exact_duplicate_instances": {key: counts[key] - len(unique[key]) for key in counts},
            "background_candidates": background, "elapsed_s": elapsed,
            "v2_detector_contributions": contributions, "reviewed_matches_lost_on_removal": ablation_losses,
            "interpretation": "Exploratory paired windows. Noise/traffic are reviewed classes, not automatically false triggers. Background candidates need blind review; overlapping windows are not independent recording hours."}


def _write_json(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".screening-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    os.replace(temporary, path)


def write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    _write_json(output_dir / "report.json", report)
    rows = []
    label_names = {"earthquake": "地震", "traffic": "车辆", "blast": "爆破", "noise": "噪声"}
    def match_text(version):
        if "matched" not in version:
            return "—"
        return "匹配" if version["matched"] else "未匹配"

    for result in report.get("results", []):
        title = str(result.get("event_id", result.get("kind", "window")))
        if result.get("occurred_at"):
            title = f"{result['occurred_at']} UTC · {title}"
        baseline = result.get("baseline", {})
        current = result.get("v2", {})
        unlabeled = "待复核背景" if result.get("kind") == "background" else "未复核"
        label = result.get("label")
        row = [title, label_names.get(label, label) or unlabeled,
               "完成" if result["status"] == "ok" else "读取或计算失败",
               match_text(baseline), match_text(current),
               str(len(baseline.get("events", []))), str(len(current.get("events", [])))]
        detail = html.escape(json.dumps(result, ensure_ascii=False, indent=2))
        rows.append("<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) +
                    f"</tr><tr><td colspan=7><details><summary>查看证据与测量</summary><pre>{detail}</pre></details></td></tr>")
    summary = html.escape(json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2))
    class_rows = []
    for label, counts in report.get("summary", {}).get("labels", {}).items():
        cells = [label_names.get(label, label), counts["evaluated"], counts["baseline_matched"], counts["v2_matched"]]
        class_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells) + "</tr>")
    comparison = ("<table><thead><tr><th>人工标签</th><th>成功读取</th><th>旧版匹配</th><th>新版匹配</th></tr></thead><tbody>"
                  + "".join(class_rows) + "</tbody></table>") if class_rows else ""
    document = """<!doctype html><html lang=zh-CN><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>DAS 初筛离线比较</title>
<style>body{font:15px/1.65 system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 24px;color:#192a3a;background:#f5f7fa}h1{font-size:26px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde5ed}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{padding:6px}summary{cursor:pointer;color:#286096}.note{border-left:4px solid #5489ae;padding:14px;background:white}</style>
<h1>DAS 初筛与事件特征离线比较</h1><p class=note>本报告使用冻结的人工标签和相同原始窗口进行探索性比较。未复核背景的候选不等同误报；历史已触发样本不能给出全目录漏检率。当前生产初筛与人工标签未改动。</p>
""" + f"<h2>已复核事件的窗口匹配</h2>{comparison}<details><summary>完整统计与限制说明</summary><pre>{summary}</pre></details><h2>逐窗口结果</h2><table><thead><tr><th>时间 / 事件 / 窗口</th><th>最新人工标签</th><th>读取状态</th><th>旧版匹配</th><th>新版匹配</th><th>旧候选数</th><th>新候选数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></html>"
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def _reference(row: dict[str, Any]):
    from das_sentinel.domain import EventRecord
    return EventRecord(event_id=row["event_id"], start_sample=row["start_sample"],
                       end_sample=row["end_sample"], peak_sample=(row["start_sample"] + row["end_sample"]) // 2,
                       channel_start=row["channel_start"], channel_stop=row["channel_stop"], evidence=())


def _verify_source(row: dict[str, Any]) -> None:
    from das_sentinel.services.discovery import fingerprint
    path = Path(row["path"])
    stat = path.stat()
    if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns"] or fingerprint(path) != row["source_fingerprint"]:
        raise ValueError("source_changed_since_processing")


def _sidecar(row: dict[str, Any], measurements: dict[str, Any]) -> dict[str, Any]:
    fields = ("event_id", "source_fingerprint", "start_sample", "end_sample", "channel_start", "channel_stop")
    return {**{name: row[name] for name in fields}, "measurements": measurements}


def _version_result(events, candidates, reference, elapsed_s: float, diagnostics=None):
    import numpy as np

    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"unsupported report value: {type(value).__name__}")

    serialized = [asdict(event) for event in events]
    result = {"events": serialized, "candidates": [asdict(candidate) for candidate in candidates],
            "matched": reference is not None and any(matches_reference(event, reference) for event in serialized),
            "elapsed_s": elapsed_s, "diagnostics": diagnostics or {}}
    # Legacy detector sample coordinates can be NumPy scalars; normalize only at the report boundary.
    return json.loads(json.dumps(result, allow_nan=False, default=scalar))


def _replay_window(block, profile, row, central_start, central_stop, *, common_mode_ablation: bool):
    from das_sentinel.evaluation_legacy import _detect_block
    from das_sentinel.fusion import fuse_candidates
    from das_sentinel.screening import (
        ScreeningOptions,
        fuse_screening_candidates,
        screen_block,
    )
    from das_sentinel.signal.backend import select_backend
    backend = select_backend("cpu")
    central = lambda candidates: tuple(item for item in candidates if central_start <= item.peak_sample < central_stop)
    reference = row if row["kind"] == "event" else None
    timers = {"preprocess": 0.0, "detect": 0.0}
    start = time.perf_counter()
    baseline_candidates = central(_detect_block(block, profile, backend, timers))
    baseline_events = fuse_candidates(baseline_candidates, profile, block.metadata.sample_rate_hz,
                                     row["site_id"], "offline-v1", row["processing_profile_id"],
                                     source_fingerprint=row["source_fingerprint"])
    baseline = _version_result(baseline_events, baseline_candidates, reference, time.perf_counter() - start, timers)
    start = time.perf_counter()
    screened = screen_block(block, profile, backend)
    candidates = central(screened.candidates)
    events = fuse_screening_candidates(candidates, profile)
    current = _version_result(events, candidates, reference, time.perf_counter() - start, screened.diagnostics)
    variants = {}
    removed_families = {"coherence": {"coherence"}, "trajectory": {"trajectory"},
                        "energy": {"energy"}, "stalta": {"stalta"},
                        "amplitude": {"stalta", "energy"}, "spatial": {"coherence", "trajectory"}}
    for family, removed in removed_families.items():
        retained = tuple(c for c in candidates if c.detector.split(":")[0] not in removed)
        accepted = fuse_screening_candidates(retained, profile)
        variants[f"without_{family}"] = {"events": len(accepted), "matched": reference is not None and
                                          any(matches_reference(asdict(event), reference) for event in accepted)}
    if common_mode_ablation:
        start = time.perf_counter()
        with_cmr = screen_block(block, profile, backend, ScreeningOptions(common_mode_removal=True))
        cmr_candidates = central(with_cmr.candidates)
        variants["common_mode_removed"] = _version_result(fuse_screening_candidates(cmr_candidates, profile),
                                                          cmr_candidates, reference, time.perf_counter() - start,
                                                          with_cmr.diagnostics)
    return {"baseline": baseline, "v2": current, "ablations": variants}


def _background_window(known: list[dict[str, Any]], samples: int, fs: float) -> tuple[int, int] | None:
    width = round(20 * fs)
    guard = round(5 * fs)
    for start in range(round(10 * fs), samples - width + 1, width):
        stop = start + width
        if all(stop <= item["start_sample"] - guard or start >= item["end_sample"] + guard for item in known):
            return start, stop
    return None


def _check_read_budget(start: int, stop: int, channel_count: int) -> None:
    if (stop - start) * channel_count * 4 > 512 * 1024**2:
        raise ValueError("window_exceeds_512MiB_read_budget")


def run_evaluation(database_path: Path, output_dir: Path, *, measurement_dir: Path | None = None,
                   event_limit: int | None = None, measurements_only: bool = False,
                   background_limit: int = 4, selected_event_ids: list[str] | None = None) -> dict[str, Any]:
    from das_sentinel.config import ProcessingProfile
    from das_sentinel.measurements import measure_event
    from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
    if event_limit is not None and event_limit <= 0:
        raise ValueError("event_limit must be positive")
    if background_limit < 0:
        raise ValueError("background_limit must be nonnegative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_code_sha256 = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("screening.py", "measurements.py", "screening_evaluation.py", "evaluation_legacy.py")}
    _write_json(output_dir / "run-code.json", source_code_sha256)
    snapshot_path = output_dir / "snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        snapshot = snapshot_events(database_path)
        _write_json(snapshot_path, snapshot)
    population = snapshot if measurements_only else select_samples(snapshot)
    common_mode_ids = set()
    for label, limit in (("earthquake", 2), ("traffic", 1), ("noise", 1)):
        controls = _stratified([row for row in population if row["label"] == label], limit)
        common_mode_ids.update(row["event_id"] for row in controls)
    selected = population
    if selected_event_ids is not None:
        if not isinstance(selected_event_ids, list) or not all(isinstance(value, str) for value in selected_event_ids):
            raise ValueError("selection must be a list of event IDs")
        requested = set(selected_event_ids)
        if len(requested) != len(selected_event_ids) or not requested.issubset({row["event_id"] for row in population}):
            raise ValueError("selection has duplicate or unknown event IDs")
        selected = [row for row in population if row["event_id"] in requested]
    if event_limit is not None:
        selected = selected[:event_limit]
    common_mode_ids.intersection_update(row["event_id"] for row in selected)
    _write_json(output_dir / "selection.json", selected)
    measurements_path = Path(measurement_dir) if measurement_dir is not None else output_dir / "measurements"
    reader = ZdHdf5Reader()
    results = []
    backgrounds = 0
    seen_sources = set()
    known_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in snapshot:
        known_by_source[item["source_file_id"]].append(item)
    for number, item in enumerate(selected, 1):
        row = {**item, "kind": "event"}
        result = {"kind": "event", "event_id": row["event_id"], "label": row["label"],
                  "source_file_id": row["source_file_id"], "date_group": row["date_group"],
                  "occurred_at": row["occurred_at"], "source_path": row["path"]}
        try:
            _verify_source(row)
            profile = ProcessingProfile.model_validate(row["profile"])
            metadata = reader.read_metadata(Path(row["path"]))
            fs = metadata.sample_rate_hz
            total = reader.sample_count(Path(row["path"]))
            central_start = (row["start_sample"] // max(1, round(20 * fs))) * round(20 * fs)
            central_stop = min(total, max(central_start + round(20 * fs), row["end_sample"] + round(2 * fs)))
            history_samples = round(max(10.0, profile.lta_seconds + profile.sta_seconds) * fs)
            read_start = max(0, central_start - history_samples)
            if measurements_only:
                read_start = max(0, row["start_sample"] - round(6 * fs))
                central_stop = min(total, row["end_sample"] + round(2 * fs))
            _check_read_budget(read_start, central_stop, len(metadata.channel_distance_m))
            block = reader.read_block(Path(row["path"]), read_start, central_stop)
            measurements = measure_event(block, _reference(row), profile)
            _verify_source(row)
            result.update(status="ok", window={"read_start": read_start, "central_start": central_start,
                                               "stop": central_stop, "sample_rate_hz": fs}, measurements=measurements)
            if not measurements_only:
                result.update(_replay_window(block, profile, row, central_start, central_stop,
                                             common_mode_ablation=row["event_id"] in common_mode_ids))
                _verify_source(row)
            _write_json(measurements_path / f"{row['event_id']}.json", _sidecar(row, measurements))
            del block
            if not measurements_only and row["source_file_id"] not in seen_sources and backgrounds < background_limit:
                seen_sources.add(row["source_file_id"])
                bounds = _background_window(known_by_source[row["source_file_id"]], total, fs)
                if bounds:
                    background_row = {**row, "kind": "background", "label": None}
                    start, stop = bounds
                    background_result = {"kind": "background", "event_id": f"background-{row['source_file_id']}-{start}",
                                         "source_file_id": row["source_file_id"], "label": None,
                                         "window": {"central_start": start, "stop": stop, "sample_rate_hz": fs}}
                    try:
                        background_read_start = max(0, start - history_samples)
                        _check_read_budget(background_read_start, stop, len(metadata.channel_distance_m))
                        background_block = reader.read_block(Path(row["path"]), background_read_start, stop)
                        background_result.update(_replay_window(background_block, profile, background_row, start, stop,
                                                                common_mode_ablation=False))
                        _verify_source(row)
                        background_result["status"] = "ok"
                        del background_block
                    except (OSError, ValueError, KeyError, RuntimeError) as error:
                        background_result.update(status="error", error=f"{type(error).__name__}: {error}")
                    results.append(background_result)
                    backgrounds += 1
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result = {key: result[key] for key in ("kind", "event_id", "label", "source_file_id", "date_group")}
            result.update(status="error", error=f"{type(error).__name__}: {error}")
        results.append(result)
        _write_json(output_dir / "results-progress.json", results)
        print(json.dumps({"completed": number, "selected": len(selected), "event": row["event_id"],
                          "status": result["status"], "error": result.get("error")}), flush=True)
    summary = ({"measured": sum(row["status"] == "ok" for row in results),
                "errors": sum(row["status"] != "ok" for row in results)} if measurements_only else summarize_results(results))
    report = {"schema_version": "1", "created_at": datetime.now(UTC).isoformat(), "mode": "measurements" if measurements_only else "paired_screening",
              "source_code_sha256": source_code_sha256,
              "full_selection_count": len(population), "requested_event_ids": selected_event_ids,
              "selection_count": len(selected), "summary": summary, "results": results,
              "common_mode_ablation_event_ids": sorted(common_mode_ids) if not measurements_only else [],
              "matching": "At least25% overlap relative to shorter interval on both time and channel axes; selected windows only.",
              "detector_input_policy": "Both versions receive the same complete read window; only central-window peaks are retained. This is a standardized detector comparison, not a byte-for-byte replay of production file chunking.",
              "database_writes": False, "production_detector_changed": False}
    write_report(output_dir, report)
    return report
