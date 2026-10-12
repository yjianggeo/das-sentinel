"""Conservative migration: local positives may imply presence, never absence."""

import json
from datetime import datetime, timedelta

from sqlalchemy import select

from das_sentinel.persistence.models import Event, ProcessingRun, Review, SourceFile
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowLabelImport,
    WindowReview,
)


def _snapshot(session, event):
    if event is None:
        return None
    run = session.get(ProcessingRun, event.processing_run_id)
    source = session.get(SourceFile, run.source_file_id)
    history = list(
        session.scalars(
            select(Review)
            .where(Review.event_id == event.id)
            .order_by(Review.created_at, Review.id)
        )
    )
    if not history:
        return None
    return {
        "event_id": event.id,
        "start_utc": event.occurred_at.isoformat(),
        "start_sample": event.start_sample,
        "end_sample": event.end_sample,
        "channel_start": event.channel_start,
        "channel_stop": event.channel_stop,
        "source_path": source.path,
        "source_fingerprint": run.source_fingerprint,
        "history": [
            {
                "id": r.id,
                "label": r.label,
                "note": r.note,
                "admin_id": r.admin_id,
                "request_id": r.request_id,
                "created_at": r.created_at.isoformat(),
            }
            for r in history
        ],
    }


def _evaluate(window, analysis, payload):
    plan = json.loads(analysis.plan_json)
    result = json.loads(analysis.result_json or "{}")
    labels = dict.fromkeys(("earthquake", "blast", "traffic"), "unknown")
    evidence = []
    for record in payload["records"]:
        review = record["history"][-1]
        reasons = []
        source = next(
            (
                s
                for s in plan["sources"]
                if s["path"].casefold() == record["source_path"].casefold()
                and s["fingerprint"] == record["source_fingerprint"]
            ),
            None,
        )
        if source is None:
            reasons.append("source_snapshot_unmatched")
        else:
            start = datetime.fromisoformat(record["start_utc"])
            end = start + timedelta(
                seconds=(record["end_sample"] - record["start_sample"])
                / source["sample_rate_hz"]
            )
            if not window.start_utc <= start < end <= window.end_utc:
                reasons.append("cross_minute_requires_confirmation")
            distance = source["distance_m"]
            first, stop = record["channel_start"], record["channel_stop"]
            profile = plan["profile_payload"]
            low, high = (
                profile.get("scan_distance_start_m"),
                profile.get("scan_distance_stop_m"),
            )
            mask = [
                (low is None or d >= low) and (high is None or d <= high)
                for d in distance
            ]
            if analysis.status == "succeeded":
                actual = result.get("recognition_channel_mask")
                if not isinstance(actual, list) or len(actual) != len(mask):
                    reasons.append("result_mask_unavailable")
                else:
                    mask = [a and b for a, b in zip(mask, actual, strict=True)]
            if not 0 <= first < stop <= len(mask):
                reasons.append("invalid_channel_range")
            elif not all(mask[first:stop]):
                reasons.append("roi_touches_excluded_channels")
        if review["label"] == "noise":
            reasons.append("local_noise_not_window_negative")
        if not reasons:
            labels[review["label"]] = "present"
        evidence.append(
            {
                "event_id": record["event_id"],
                "review_id": review["id"],
                "label": review["label"],
                "held_reasons": reasons,
            }
        )
    return labels, evidence


def _source_reviews_current(session, saved):
    return all(
        _snapshot(session, session.get(Event, r["event_id"])) == r
        for r in json.loads(saved.payload_json)["records"]
    )


def inherited_review_current(session, review):
    saved = session.get(WindowLabelImport, review.window_id)
    return (
        saved is None
        or saved.applied_review_id != review.id
        or _source_reviews_current(session, saved)
    )


def stale_inherited_review_ids(session):
    # Check only applied imports; pending drafts do not affect review pagination.
    return [
        saved.applied_review_id
        for saved in session.scalars(
            select(WindowLabelImport).where(
                WindowLabelImport.applied_review_id.is_not(None)
            )
        )
        if not _source_reviews_current(session, saved)
    ]


def import_summary(service, window, analysis):
    saved = service.session.get(WindowLabelImport, window.id)
    if saved is None:
        return None
    payload = json.loads(saved.payload_json)
    bound_analysis = service.session.get(WindowAnalysis, saved.analysis_id)
    labels, evidence = _evaluate(window, bound_analysis, payload)
    latest = service.session.scalar(
        select(WindowReview)
        .where(WindowReview.window_id == window.id)
        .order_by(WindowReview.created_at.desc(), WindowReview.id.desc())
        .limit(1)
    )
    if analysis.id != saved.analysis_id:
        state = "analysis_changed"
    elif latest is not None and latest.id != saved.applied_review_id:
        state = "manual_kept"
    elif not _source_reviews_current(service.session, saved):
        state = "source_review_changed"
    elif analysis.status == "succeeded" and not service._current(analysis):
        state = "input_changed"
    elif latest is not None:
        state = "applied"
    elif "present" not in labels.values():
        state = "needs_review"
    elif analysis.status == "failed":
        state = "waiting_data"
    elif analysis.status == "cancelled":
        state = "cancelled"
    elif analysis.status != "succeeded":
        state = "waiting_analysis"
    else:
        state = "ready"
    return {
        "state": state,
        "labels": labels,
        "evidence": evidence,
        "history_count": sum(len(r["history"]) for r in payload["records"]),
        "event_count": len(payload["records"]),
        "applied_review_id": saved.applied_review_id,
    }


def apply_import(service, analysis):
    window = service.session.get(EventWindow, analysis.window_id)
    if window.latest_analysis_id != analysis.id:
        return
    summary = import_summary(service, window, analysis)
    if summary is None or summary["state"] != "ready":
        return
    saved = service.session.get(WindowLabelImport, window.id)
    references = [
        str(e["review_id"]) for e in summary["evidence"] if not e["held_reasons"]
    ]
    review = service.review(
        window.id,
        analysis_id=analysis.id,
        input_sha256=analysis.input_sha256,
        labels=summary["labels"],
        request_id="legacy-positive-v1:" + window.id,
        admin_id=saved.admin_id,
        note="旧局部复核迁移：仅继承有效识别范围内、完整落于本窗口的正标签；"
        "其他类别未确定，未经重新整窗人工确认。来源复核 ID：" + ", ".join(references),
    )
    saved.applied_review_id = review["id"]
    service.session.flush()


def seed_imports(service, *, admin_id=1, window_seconds=30):
    """Explicit operator action only; never implicitly expands to future reviews."""
    session = service.session
    count = 0
    for window in session.scalars(select(EventWindow).where(EventWindow.window_seconds == window_seconds).order_by(EventWindow.start_utc)):
        if session.get(WindowLabelImport, window.id) is None:
            ids = json.loads(window.legacy_event_ids_json)
            records = [
                _snapshot(session, e)
                for e in session.scalars(
                    select(Event)
                    .where(Event.id.in_(ids))
                    .order_by(Event.occurred_at, Event.id)
                )
            ]
            records = [r for r in records if r is not None]
            if not records:
                continue
            saved = WindowLabelImport(
                window_id=window.id,
                analysis_id=window.latest_analysis_id,
                admin_id=admin_id,
                payload_json=json.dumps(
                    {"schema_version": 1, "records": records},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            session.add(saved)
            session.flush()
            count += 1
        apply_import(service, session.get(WindowAnalysis, window.latest_analysis_id))
    return count
