import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from das_sentinel.persistence.models import (
    Admin,
    Event,
    ProcessingRun,
    Review,
    SourceFile,
)
from das_sentinel.persistence.window_models import WindowReview
from das_sentinel.services import event_windows
from sqlalchemy import func, select

from tests import test_event_windows as cases

context = cases.context


def legacy(context, label="earthquake", *, second=10, duration=2, channels=(0, 2)):
    session, site, profile, _service = context
    if session.get(Admin, 1) is None:
        session.add(Admin(id=1, password_hash="test"))
    source = cases.plan()["sources"][0]
    run = session.scalar(select(ProcessingRun))
    if run is None:
        file = SourceFile(
            site_id=site.id,
            path=source["path"],
            fingerprint=source["fingerprint"],
            size_bytes=source["size_bytes"],
            mtime_ns=source["mtime_ns"],
        )
        session.add(file)
        session.flush()
        run = ProcessingRun(
            source_file_id=file.id,
            processing_profile_id=profile.id,
            source_fingerprint=source["fingerprint"],
            profile_version=1,
            pipeline_version="v2",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        session.add(run)
        session.flush()
    event = Event(
        processing_run_id=run.id,
        event_uid=str(uuid4()),
        start_sample=100 * (60 + second),
        end_sample=100 * (60 + second + duration),
        channel_start=channels[0],
        channel_stop=channels[1],
        occurred_at=datetime(2023, 9, 27, 0, 0, second, tzinfo=UTC),
    )
    session.add(event)
    session.flush()
    review = Review(event_id=event.id, admin_id=1, label=label, note="original")
    session.add(review)
    session.flush()
    return event, review


def enqueue(context, events):
    s, site, p, _ = context
    return event_windows.enqueue_windows(
        s, site.id, p.id, [cases.plan()], legacy_event_ids=[e.id for e in events]
    )[0]


def complete(context, analysis):
    _, _, _, service = context
    service.claim("worker")
    result = cases.make_result(service, analysis)
    cases.write_result(service, analysis, result)
    service.result(analysis.id, "worker", 1)


def test_import_preserves_history_merges_positives_and_never_infers_absence(context):
    from das_sentinel.services.window_label_imports import seed_imports

    s, _, _, service = context
    quake, _first = legacy(context)
    # A later correction wins; the earlier earthquake label must not survive.
    correction = Review(event_id=quake.id, admin_id=1, label="noise", note="correction")
    s.add(correction)
    s.flush()
    blast, _ = legacy(context, "blast", second=20)
    traffic, _ = legacy(context, "traffic", second=30)
    a = enqueue(context, [quake, blast, traffic])
    seed_imports(service, window_seconds=60)
    assert s.scalar(select(func.count()).select_from(WindowReview)) == 0
    detail = service.detail(a.window_id)["legacy_label_import"]
    assert detail["state"] == "waiting_analysis"
    assert detail["history_count"] == 4
    complete(context, a)
    r = service.detail(a.window_id)["latest_review"]
    assert r["labels"] == {"earthquake": "present"}
    assert not r["background_only"] and not r["stale"]
    assert "旧局部复核迁移" in r["note"]
    seed_imports(service, window_seconds=60)
    assert s.scalar(select(func.count()).select_from(WindowReview)) == 1
    assert s.scalar(select(func.count()).select_from(Review)) == 4
    assert service.detail(a.window_id)["legacy_label_import"]["state"] == "applied"


@pytest.mark.parametrize(
    "label,second,duration,channels,mask",
    [
        ("noise", 10, 2, (0, 2), [True, True]),
        ("earthquake", 59, 2, (0, 2), [True, True]),
        ("earthquake", 10, 2, (0, 2), [False, True]),
    ],
)
def test_noise_cross_minute_and_excluded_roi_remain_unreviewed(
    context, label, second, duration, channels, mask
):
    from das_sentinel.services.window_label_imports import apply_import, seed_imports

    s, _, _, service = context
    event, _ = legacy(
        context, label, second=second, duration=duration, channels=channels
    )
    a = enqueue(context, [event])
    seed_imports(service, window_seconds=60)
    a.status = "succeeded"
    a.input_sha256 = "a" * 64
    a.result_json = json.dumps({"recognition_channel_mask": mask})
    s.flush()
    apply_import(service, a)
    assert service.detail(a.window_id)["latest_review"] is None
    assert service.detail(a.window_id)["legacy_label_import"]["state"] == "needs_review"


def test_already_reviewed_window_is_never_overwritten(context):
    from das_sentinel.services.window_label_imports import seed_imports

    _s, _, _, service = context
    event, _ = legacy(context)
    a = enqueue(context, [event])
    complete(context, a)
    saved = service.review(
        a.window_id,
        analysis_id=a.id,
        input_sha256=a.input_sha256,
        labels={"earthquake": "absent", "blast": "absent", "traffic": "unknown"},
        note="whole window",
        request_id="manual",
        admin_id=1,
    )
    seed_imports(service, window_seconds=60)
    assert service.detail(a.window_id)["latest_review"]["id"] == saved["id"]
    assert service.detail(a.window_id)["legacy_label_import"]["state"] == "manual_kept"


def test_changed_legacy_review_blocks_deferred_import(context):
    from das_sentinel.services.window_label_imports import seed_imports

    s, _, _, service = context
    event, _ = legacy(context)
    a = enqueue(context, [event])
    seed_imports(service, window_seconds=60)
    s.add(Review(event_id=event.id, admin_id=1, label="noise"))
    s.flush()
    complete(context, a)
    assert service.detail(a.window_id)["latest_review"] is None
    assert (
        service.detail(a.window_id)["legacy_label_import"]["state"]
        == "source_review_changed"
    )


def test_changed_analysis_does_not_rebind_import(context):
    from das_sentinel.services.window_label_imports import apply_import, seed_imports

    s, site, p, service = context
    event, _ = legacy(context)
    old = enqueue(context, [event])
    seed_imports(service, window_seconds=60)
    new = event_windows.enqueue_windows(
        s, site.id, p.id, [cases.plan() | {"algorithm_version": "next"}]
    )[0]
    assert old.id != new.id
    assert (
        service.detail(new.window_id)["legacy_label_import"]["state"]
        == "analysis_changed"
    )
    assert service.detail(new.window_id)["latest_review"] is None
    # Late results for an old analysis cannot promote a stale import.
    old.status = "succeeded"
    old.input_sha256 = "a" * 64
    old.result_json = json.dumps({"recognition_channel_mask": [True, True]})
    s.flush()
    apply_import(service, old)
    assert s.scalar(select(func.count()).select_from(WindowReview)) == 0


def test_training_export_marks_inherited_origin_and_keeps_unknown(context):
    from das_sentinel.services.window_label_imports import seed_imports
    from das_sentinel.window_training import export_rows

    s, site, _, service = context
    event, _ = legacy(context)
    a = enqueue(context, [event])
    seed_imports(service, window_seconds=60)
    complete(context, a)
    row = export_rows(s, service.artifact_root, site.id, window_seconds=60)["rows"][0]
    assert row["label_origin"] == "legacy_positive_import"
    assert row["labels"] == {"earthquake": "present"}
    assert row["source_labels"] == {
        "earthquake": "present",
        "blast": "unknown",
        "traffic": "unknown",
    }


def test_corrected_legacy_label_invalidates_import_but_not_later_manual_review(context):
    from das_sentinel.services.window_label_imports import seed_imports
    from das_sentinel.window_training import export_rows

    s, site, _, service = context
    event, _ = legacy(context)
    a = enqueue(context, [event])
    seed_imports(service, window_seconds=60)
    complete(context, a)
    s.add(Review(event_id=event.id, admin_id=1, label="noise"))
    s.flush()
    detail = service.detail(a.window_id)
    assert detail["latest_review"]["stale"]
    assert detail["legacy_label_import"]["state"] == "source_review_changed"
    assert export_rows(s, service.artifact_root, site.id, window_seconds=60)["rows"] == []
    assert service.list(window_seconds=60, review_status="unreviewed")["total"] == 1
    assert service.list(window_seconds=60, review_status="reviewed")["total"] == 0
    service.review(
        a.window_id,
        analysis_id=a.id,
        input_sha256=a.input_sha256,
        labels={"earthquake": "present", "blast": "unknown", "traffic": "unknown"},
        note="new whole-window confirmation",
        request_id="new-manual",
        admin_id=1,
    )
    assert not service.detail(a.window_id)["latest_review"]["stale"]
    assert service.list(window_seconds=60, review_status="reviewed")["total"] == 1
    assert (
        export_rows(s, service.artifact_root, site.id, window_seconds=60)["rows"][0]["label_origin"]
        == "whole_window_review"
    )
