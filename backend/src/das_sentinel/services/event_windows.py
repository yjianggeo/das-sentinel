from __future__ import annotations

import hashlib
import json
import math
import stat
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import case, exists, func, literal_column, or_, select, update
from sqlalchemy.orm import Session

from das_sentinel.persistence.models import (
    Event,
    ProcessingProfile,
    Review,
    Site,
    utc_now,
)
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowFavorite,
    WindowLabelImport,
    WindowReview,
    WindowSkip,
    WindowSpatialCoverage,
)
from das_sentinel.window_training import binary_labels

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_EVIDENCE = 100000
LEASE_SECONDS = 180
EMPTY_SKIP_POLICY = "empty-regions-v1"
_COVERAGE_UNSET = object()


class WindowError(ValueError):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def enqueue_windows(
    session: Session,
    site_id,
    profile_id,
    plans,
    *,
    priority=0,
    catalog_matches=(),
    legacy_event_ids=(),
    model_selection=False,
):
    profile = session.get(ProcessingProfile, profile_id)
    if profile is None or profile.site_id != site_id:
        raise WindowError("profile_site_mismatch")
    saved = []
    for raw in plans:
        plan = dict(raw)
        start, end = (datetime.fromisoformat(plan[k]) for k in ("start_utc", "end_utc"))
        if start.utcoffset() is None or end.utcoffset() is None:
            raise WindowError("invalid_window", 422)
        from datetime import UTC

        start, end = start.astimezone(UTC), end.astimezone(UTC)
        duration = (end - start).total_seconds()
        if (
            start.tzinfo is None
            or end.tzinfo is None
            or duration not in (30, 60)
            or start.second % duration
            or start.microsecond
        ):
            raise WindowError("invalid_window", 422)
        plan.update(start_utc=start.isoformat(), end_utc=end.isoformat())
        from das_sentinel.whole_window import window_id

        duration = int(duration)
        wid = window_id(site_id, start, duration)
        plan.update(
            profile_payload=json.loads(profile.payload_json),
            profile_id=profile_id,
            algorithm_version=plan.get("algorithm_version", "whole-window-v2" if duration == 30 else "whole-window-v1"),
        )
        signature = digest(plan)
        if not model_selection and session.scalar(select(WindowSkip.analysis_id).where(
            WindowSkip.window_id == wid, WindowSkip.plan_sha256 == signature,
            WindowSkip.policy_version == EMPTY_SKIP_POLICY,
        )) is not None:
            continue
        window = session.get(EventWindow, wid)
        if window is None:
            window = EventWindow(
                id=wid,
                site_id=site_id,
                start_utc=start,
                end_utc=end,
                window_seconds=duration,
                priority=priority,
                catalog_matches_json=canonical(list(catalog_matches)),
                legacy_event_ids_json=canonical(list(legacy_event_ids)),
            )
            session.add(window)
            session.flush()
        else:
            window.priority = max(window.priority, priority)
            for attr, values in [
                ("catalog_matches_json", catalog_matches),
                ("legacy_event_ids_json", legacy_event_ids),
            ]:
                previous = json.loads(getattr(window, attr))
                setattr(
                    window,
                    attr,
                    canonical(previous + [v for v in values if v not in previous]),
                )
        analysis = session.scalar(
            select(WindowAnalysis).where(
                WindowAnalysis.window_id == wid, WindowAnalysis.plan_sha256 == signature
            )
        )
        if analysis is None:
            complete = plan["coverage"] == "complete"
            analysis = WindowAnalysis(
                window_id=wid,
                profile_id=profile_id,
                plan_json=canonical(plan),
                plan_sha256=signature,
                algorithm_version=plan["algorithm_version"],
                status="queued" if complete else "failed",
                error_code=None if complete else "coverage_incomplete",
            )
            session.add(analysis)
            session.flush()
            window.latest_analysis_id = analysis.id
        saved.append(analysis)
    session.flush()
    return saved


class WindowService:
    def __init__(self, session: Session, artifact_root: Path):
        self.session, self.artifact_root = session, Path(artifact_root).resolve()

    def _analysis(self, aid):
        a = self.session.get(WindowAnalysis, aid)
        if a is None:
            raise WindowError("analysis_not_found", 404)
        return a

    def _window(self, wid):
        w = self.session.get(EventWindow, wid)
        if w is None:
            raise WindowError("window_not_found", 404)
        return w

    def _owned(self, aid, worker_id, attempt_count):
        a = self._analysis(aid)
        if (
            a.status != "running"
            or a.worker_id != worker_id
            or a.attempt_count != attempt_count
            or a.heartbeat_at is None
            or a.heartbeat_at < utc_now() - timedelta(seconds=LEASE_SECONDS)
        ):
            raise WindowError("claim_lost")
        return a

    def job(self, a):
        from das_sentinel.persistence.model_scan_models import ModelScanReceipt
        w = self._window(a.window_id)
        site = self.session.get(Site, w.site_id)
        return {
            "model_scan_id": self.session.scalar(select(ModelScanReceipt.scan_id).where(
                ModelScanReceipt.analysis_id == a.id).limit(1)),
            "site_id": w.site_id,
            "site_name": site.name,
            "site_timezone": site.timezone,
            "start_utc": w.start_utc,
            "end_utc": w.end_utc,
            "window_seconds": w.window_seconds,
            "profile_id": a.profile_id,
        } | {
            k: getattr(a, k)
            for k in (
                "id",
                "window_id",
                "status",
                "worker_id",
                "attempt_count",
                "heartbeat_at",
                "error_code",
                "error_message",
                "created_at",
                "input_sha256",
            )
        }

    def _can_skip_empty(self, a, w):
        if (w.window_seconds != 30 or w.latest_analysis_id != a.id
                or a.result_json is not None or a.input_sha256 is not None):
            return False
        if self.session.scalar(select(func.count()).select_from(WindowAnalysis).where(
            WindowAnalysis.window_id == w.id,
        )) != 1:
            return False
        if self.session.scalar(select(WindowReview.id).where(WindowReview.window_id == w.id).limit(1)):
            return False
        if self.session.scalar(select(WindowFavorite.window_id).where(WindowFavorite.window_id == w.id).limit(1)):
            return False
        # Existing attempts may already have generated files before publishing.
        if (self.artifact_root / a.id).exists():
            return False
        legacy_ids = set(json.loads(w.legacy_event_ids_json))
        draft = self.session.get(WindowLabelImport, w.id)
        if draft is not None:
            if draft.applied_review_id is not None or draft.analysis_id != a.id:
                return False
            for record in json.loads(draft.payload_json)["records"]:
                if record["history"][-1]["label"] != "noise":
                    return False
                legacy_ids.add(record["event_id"])
        # A human may have changed a legacy label after the import snapshot.
        for event_id in legacy_ids:
            label = self.session.scalar(select(Review.label).where(Review.event_id == event_id)
                .order_by(Review.created_at.desc(), Review.id.desc()).limit(1))
            if label is not None and label != "noise":
                return False
        return True

    def claim(self, worker_id, supports_empty_skip=False):
        from das_sentinel.persistence.model_scan_models import ModelScanReceipt
        cutoff = utc_now() - timedelta(seconds=LEASE_SECONDS)
        available = or_(
            WindowAnalysis.status == "queued",
            (WindowAnalysis.status == "running")
            & (WindowAnalysis.heartbeat_at < cutoff),
        )
        ids = self.session.scalars(
            select(WindowAnalysis.id)
            .join(EventWindow, EventWindow.id == WindowAnalysis.window_id)
            .where(available, EventWindow.latest_analysis_id == WindowAnalysis.id,
                   ~exists(select(ModelScanReceipt.id).where(ModelScanReceipt.analysis_id == WindowAnalysis.id)))
            .order_by(EventWindow.priority.desc(), WindowAnalysis.created_at)
            .limit(20)
        ).all()
        for aid in ids:
            previous = self._analysis(aid)
            policy = None
            if (supports_empty_skip and
                    (previous.status == "queued" or previous.skip_policy_version == EMPTY_SKIP_POLICY)
                    and (previous.attempt_count == 0 or previous.skip_policy_version == EMPTY_SKIP_POLICY)
                    and self._can_skip_empty(previous, self._window(previous.window_id))):
                policy = EMPTY_SKIP_POLICY
            result = self.session.execute(
                update(WindowAnalysis)
                .where(WindowAnalysis.id == aid, available)
                .values(
                    status="running",
                    worker_id=worker_id,
                    attempt_count=WindowAnalysis.attempt_count + 1,
                    heartbeat_at=utc_now(),
                    skip_policy_version=policy,
                )
            )
            if result.rowcount != 1:
                continue
            self.session.flush()
            self.session.expire_all()
            a = self._analysis(aid)
            w = self._window(a.window_id)
            site = self.session.get(Site, w.site_id)
            plan = json.loads(a.plan_json)
            return self.job(a) | {
                "site_id": w.site_id,
                "site_data_root": site.data_path,
                "start_utc": w.start_utc.isoformat(),
                "end_utc": w.end_utc.isoformat(),
                "profile_payload": plan["profile_payload"],
                "sources": plan["sources"],
                "skip_empty": a.skip_policy_version == EMPTY_SKIP_POLICY,
                "skip_policy_version": a.skip_policy_version,
                "plan_sha256": a.plan_sha256,
            }
        return None

    def skip(self, aid, worker_id, attempt_count, plan_sha256, reason="no_qualified_regions"):
        receipt = self.session.get(WindowSkip, aid)
        if receipt is not None:
            if (receipt.worker_id != worker_id or receipt.attempt_count != attempt_count
                    or receipt.plan_sha256 != plan_sha256 or receipt.reason != reason):
                raise WindowError("claim_lost")
            return {"id": aid, "status": "skipped"}
        a = self._owned(aid, worker_id, attempt_count)
        w = self._window(a.window_id)
        if (a.skip_policy_version != EMPTY_SKIP_POLICY or a.plan_sha256 != plan_sha256
                or reason != "no_qualified_regions" or not self._can_skip_empty(a, w)):
            raise WindowError("skip_not_allowed")
        profile = self.session.get(ProcessingProfile, a.profile_id)
        if profile is None or not profile.is_current or not self._current(a):
            raise WindowError("source_profile_changed")
        self.session.add(WindowSkip(analysis_id=a.id, window_id=w.id, site_id=w.site_id,
            start_utc=w.start_utc, window_seconds=w.window_seconds, plan_sha256=a.plan_sha256,
            policy_version=a.skip_policy_version, worker_id=worker_id, attempt_count=attempt_count,
            reason=reason))
        draft = self.session.get(WindowLabelImport, w.id)
        if draft is not None:
            self.session.delete(draft)
            self.session.flush()
        self.session.delete(a)
        self.session.flush()
        self.session.delete(w)
        self.session.flush()
        return {"id": aid, "status": "skipped"}

    def heartbeat(self, aid, worker_id, attempt_count):
        a = self._owned(aid, worker_id, attempt_count)
        a.heartbeat_at = utc_now()
        self.session.flush()
        return self.job(a)

    def fail(self, aid, worker_id, error_code, error_message, attempt_count):
        a = self._owned(aid, worker_id, attempt_count)
        a.status = "failed"
        a.error_code = error_code
        a.error_message = error_message
        self.session.flush()
        return self.job(a)

    def cancel(self, aid):
        a = self._analysis(aid)
        if self.job(a)["model_scan_id"]:
            raise WindowError("use_model_scan_control")
        if a.status not in ("succeeded", "cancelled"):
            a.status = "cancelled"
        self.session.flush()
        return self.job(a)

    def retry(self, aid):
        a = self._analysis(aid)
        if self.job(a)["model_scan_id"]:
            raise WindowError("use_model_scan_control")
        if a.status != "failed" or json.loads(a.plan_json)["coverage"] != "complete":
            raise WindowError("retry_requires_complete_failed_analysis")
        if self._window(a.window_id).latest_analysis_id != a.id:
            raise WindowError("retry_requires_latest_analysis")
        a.status = "queued"
        a.worker_id = None
        a.heartbeat_at = None
        a.error_code = None
        a.error_message = None
        self.session.flush()
        return self.job(a)

    def _current(self, a):
        plan = json.loads(a.plan_json)
        p = self.session.get(ProcessingProfile, a.profile_id)
        if p is None or json.loads(p.payload_json) != plan["profile_payload"]:
            return False
        newer = self.session.scalar(
            select(ProcessingProfile).where(
                ProcessingProfile.site_id == p.site_id,
                ProcessingProfile.is_current == True,
            )
        )
        if newer is not None and newer.id != p.id:
            return False
        for source in plan["sources"]:
            path = Path(source["path"])
            try:
                info = path.stat()
            except OSError:
                return False
            if (
                info.st_size != source["size_bytes"]
                or info.st_mtime_ns != source["mtime_ns"]
            ):
                return False
            from das_sentinel.services.discovery import fingerprint

            if fingerprint(path) != source["fingerprint"]:
                return False
        return True

    def _review_dict(self, r, a):
        from das_sentinel.services.window_label_imports import inherited_review_current

        labels = binary_labels(json.loads(r.labels_json))
        return {
            "id": r.id,
            "analysis_id": r.analysis_id,
            "input_sha256": r.input_sha256,
            "labels": labels,
            "note": r.note,
            "request_id": r.request_id,
            "created_at": r.created_at,
            "background_only": all(v == "absent" for v in labels.values()),
            "stale": r.analysis_id != a.id
            or r.input_sha256 != a.input_sha256
            or not self._current(a)
            or not inherited_review_current(self.session, r),
        }

    def review(
        self, wid, *, analysis_id, input_sha256, labels, note, request_id, admin_id
    ):
        if set(labels) not in ({"earthquake"}, {"earthquake", "blast", "traffic"}) or any(
            v not in ("present", "absent", "unknown") for v in labels.values()
        ):
            raise WindowError("invalid_labels", 422)
        w = self._window(wid)
        a = self._analysis(w.latest_analysis_id)
        existing = self.session.scalar(
            select(WindowReview).where(
                WindowReview.window_id == wid, WindowReview.request_id == request_id
            )
        )
        if existing:
            if (
                existing.analysis_id,
                existing.input_sha256,
                canonical(binary_labels(json.loads(existing.labels_json))),
                existing.note,
                existing.admin_id,
            ) != (analysis_id, input_sha256, canonical(binary_labels(labels)), note, admin_id):
                raise WindowError("request_id_conflict")
            return self._review_dict(existing, a)
        if (
            a.id != analysis_id
            or a.status != "succeeded"
            or a.input_sha256 != input_sha256
            or not self._current(a)
        ):
            raise WindowError("stale_analysis")
        r = WindowReview(
            window_id=wid,
            analysis_id=analysis_id,
            input_sha256=input_sha256,
            labels_json=canonical(labels),
            note=note,
            request_id=request_id,
            admin_id=admin_id,
        )
        self.session.add(r)
        self.session.flush()
        return self._review_dict(r, a)

    def _model_recommendation(self, w, model_scan_id=None):
        from das_sentinel.persistence.model_scan_models import (
            ModelScan,
            ModelScanReceipt,
        )
        a = self._analysis(w.latest_analysis_id)
        query = select(ModelScanReceipt).where(ModelScanReceipt.window_id == w.id,
            ModelScanReceipt.status == "ready", ModelScanReceipt.analysis_id == a.id,
            ModelScanReceipt.input_sha256 == a.input_sha256)
        if model_scan_id:
            query = query.where(ModelScanReceipt.scan_id == model_scan_id)
        row = self.session.scalar(query.order_by(ModelScanReceipt.created_at.desc(), ModelScanReceipt.id.desc()).limit(1))
        if row is None or a.status != "succeeded":
            return None
        model = json.loads(self.session.get(ModelScan, row.scan_id).config_json)["model"]
        return {"scan_id": row.scan_id, "model_id": model["id"], "model_name": model["name"],
                "score": row.final_score, "threshold": model["threshold"],
                "analysis_id": a.id, "input_sha256": a.input_sha256}

    def set_favorite(self, wid, favorite, *, admin_id):
        w = self._window(wid)
        saved = self.session.get(WindowFavorite, (wid, admin_id))
        if favorite and saved is None:
            a = self._analysis(w.latest_analysis_id)
            artifacts = json.loads(a.result_json or "{}").get("artifacts", [])
            has_waterfall = any(
                entry.get("kind") in ("waterfall", "waterfall_filtered")
                and entry.get("path") and entry.get("media_type") == "image/png"
                for entry in artifacts
            )
            if w.window_seconds != 30 or a.status != "succeeded" or not has_waterfall:
                raise WindowError("favorite_requires_completed_window")
            saved = WindowFavorite(window_id=wid, admin_id=admin_id)
            self.session.add(saved)
            self.session.flush()
        elif not favorite and saved is not None:
            self.session.delete(saved)
            self.session.flush()
            saved = None
        return {"window_id": wid, "favorite": saved is not None,
                "created_at": saved.created_at if saved is not None else None}

    def _favorite_ids(self, window_ids, admin_id):
        if admin_id is None or not window_ids:
            return set()
        return set(self.session.scalars(select(WindowFavorite.window_id).where(
            WindowFavorite.admin_id == admin_id, WindowFavorite.window_id.in_(window_ids),
        )))

    def _item(self, w, model_scan_id=None, *, favorite=False, spatial_coverage_row=_COVERAGE_UNSET,
              review_protected=_COVERAGE_UNSET):
        a = self._analysis(w.latest_analysis_id)
        site = self.session.get(Site, w.site_id)
        result = json.loads(a.result_json or "{}")
        review = self.session.scalar(
            select(WindowReview)
            .where(WindowReview.window_id == w.id)
            .order_by(WindowReview.created_at.desc(), WindowReview.id.desc())
            .limit(1)
        )
        latest_review = None if review is None else self._review_dict(review, a)
        if review_protected is _COVERAGE_UNSET:
            review_protected = (latest_review is not None and not latest_review["stale"]
                and latest_review["labels"]["earthquake"] == "present")
        return {
            "id": w.id,
            "favorite": favorite,
            "window_seconds": w.window_seconds,
            "site_id": w.site_id,
            "site_name": site.name,
            "site_timezone": site.timezone,
            "start_utc": w.start_utc,
            "end_utc": w.end_utc,
            "status": a.status,
            "score": result.get("score"),
            "candidate": result.get("candidate"),
            "priority": w.priority,
            "legacy_count": len(json.loads(w.legacy_event_ids_json)),
            "catalog_matches": json.loads(w.catalog_matches_json),
            "model_review_attention": result.get("model_review_attention"),
            "model_recommendation": self._model_recommendation(w, model_scan_id),
            "spatial_coverage": self._spatial_coverage(a, spatial_coverage_row,
                review_protected=review_protected),
            "analysis_id": a.id,
            "input_sha256": a.input_sha256,
            "latest_review": latest_review,
        }

    def list(
        self,
        *,
        site_id=None,
        review_status="unreviewed",
        recommended=False,
        candidate_only=False,
        model_recommended=False,
        model_scan_id=None,
        limit=30,
        offset=0,
        window_seconds=30,
        favorites_only=False,
        admin_id=None,
        sort="default",
    ):
        from das_sentinel.services.window_label_imports import (
            stale_inherited_review_ids,
        )

        q = select(EventWindow).join(
            WindowAnalysis, EventWindow.latest_analysis_id == WindowAnalysis.id
        )
        if (model_recommended and (recommended or candidate_only)) or (model_scan_id and not model_recommended):
            raise WindowError("incompatible_window_filters", 422)
        q = q.where(EventWindow.window_seconds == window_seconds)
        if site_id:
            q = q.where(EventWindow.site_id == site_id)
        if favorites_only:
            q = q.where(exists().where(
                WindowFavorite.window_id == EventWindow.id,
                WindowFavorite.admin_id == admin_id,
            ))
        current_profile = exists().where(
            ProcessingProfile.id == WindowAnalysis.profile_id,
            ProcessingProfile.is_current == True,
        )
        latest_review_id = (
            select(WindowReview.id)
            .where(WindowReview.window_id == EventWindow.id)
            .order_by(WindowReview.created_at.desc(), WindowReview.id.desc())
            .limit(1)
            .correlate(EventWindow)
            .scalar_subquery()
        )
        earthquake = func.json_extract(WindowReview.labels_json, "$.earthquake")
        blast = func.json_extract(WindowReview.labels_json, "$.blast")
        # Match binary_labels, including reviews saved before binary conversion.
        positive = or_(earthquake == "present", blast == "present")
        negative = (earthquake == "absent") & or_(blast == "absent", blast.is_(None))
        determined = or_(positive, negative)
        if review_status == "reviewed_earthquake":
            determined = positive
        elif review_status == "reviewed_noise":
            determined = negative
        stale_review_ids = stale_inherited_review_ids(self.session)
        bound_review = (
            WindowReview.id == latest_review_id,
            WindowReview.window_id == EventWindow.id,
            WindowReview.analysis_id == WindowAnalysis.id,
            WindowReview.input_sha256 == WindowAnalysis.input_sha256,
            WindowReview.id.not_in(stale_review_ids),
        )
        reviewed = exists().where(*bound_review, determined) & current_profile
        review_protected = exists().where(*bound_review, positive) & current_profile
        if review_status in ("reviewed", "reviewed_earthquake", "reviewed_noise"):
            q = q.where(reviewed)
        elif review_status == "unreviewed":
            q = q.where(~reviewed)
        if recommended:
            q = q.where(
                or_(
                    EventWindow.catalog_matches_json != "[]",
                    func.json_extract(WindowAnalysis.result_json, literal_column("'$.candidate'")) == 1,
                )
            )
        if candidate_only:
            q = q.where(
                WindowAnalysis.status == "succeeded",
                func.json_extract(WindowAnalysis.result_json, literal_column("'$.candidate'")) == 1,
            )
        # Fixed paths match the compact expression index even in parameterized SQL.
        order = (EventWindow.priority.desc(), func.json_extract(WindowAnalysis.result_json, literal_column("'$.score'")).desc(),
                 EventWindow.start_utc, EventWindow.id)
        if model_recommended:
            from das_sentinel.persistence.model_scan_models import ModelScanReceipt
            latest = select(ModelScanReceipt.final_score).where(ModelScanReceipt.window_id == EventWindow.id,
                ModelScanReceipt.status == "ready", ModelScanReceipt.analysis_id == WindowAnalysis.id,
                ModelScanReceipt.input_sha256 == WindowAnalysis.input_sha256)
            if model_scan_id:
                latest = latest.where(ModelScanReceipt.scan_id == model_scan_id)
            score = latest.order_by(ModelScanReceipt.created_at.desc(), ModelScanReceipt.id.desc()).limit(1).correlate(
                EventWindow, WindowAnalysis).scalar_subquery()
            q = q.where(WindowAnalysis.status == "succeeded", score.is_not(None), current_profile)
            order = (score.desc(), EventWindow.start_utc, EventWindow.id)
        count_query = q
        q = q.add_columns(review_protected.label("review_protected"))
        if sort == "borehole_coverage":
            from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_READ_VERSIONS

            q = q.outerjoin(WindowSpatialCoverage,
                (WindowSpatialCoverage.analysis_id == WindowAnalysis.id)
                & (WindowSpatialCoverage.input_sha256 == WindowAnalysis.input_sha256)
                & (WindowSpatialCoverage.algorithm_version.in_(BOREHOLE_COVERAGE_READ_VERSIONS))
                & (WindowAnalysis.status == "succeeded") & current_profile)
            q = q.add_columns(WindowSpatialCoverage)
            coverage_score = func.json_extract(WindowSpatialCoverage.payload_json, "$.coverage_score")
            effective_score = case((review_protected,
                case((func.json_type(WindowSpatialCoverage.payload_json, "$.coverage_score").is_(None),
                    WindowSpatialCoverage.score), else_=coverage_score)), else_=WindowSpatialCoverage.score)
            order = (effective_score.desc(), *order)
        elif sort != "default":
            raise WindowError("invalid_window_sort", 422)
        total = self.session.scalar(select(func.count()).select_from(count_query.subquery()))
        page = q.order_by(*order).offset(offset).limit(limit)
        if sort == "borehole_coverage":
            rows = self.session.execute(page.execution_options(populate_existing=True)).all()
        else:
            rows = [(w, protected, _COVERAGE_UNSET) for w, protected in self.session.execute(page)]
        favorite_ids = self._favorite_ids([w.id for w, _protected, _coverage in rows], admin_id)
        return {
            "items": [self._item(w, model_scan_id, favorite=w.id in favorite_ids,
                spatial_coverage_row=coverage, review_protected=protected) for w, protected, coverage in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def _spatial_coverage(self, analysis, row=_COVERAGE_UNSET, *, review_protected=_COVERAGE_UNSET):
        from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_READ_VERSIONS

        if row is _COVERAGE_UNSET:
            row = self.session.get(WindowSpatialCoverage, analysis.id)
        profile = self.session.get(ProcessingProfile, analysis.profile_id)
        if (row is None or analysis.status != "succeeded"
                or row.input_sha256 != analysis.input_sha256
                or row.algorithm_version not in BOREHOLE_COVERAGE_READ_VERSIONS or not profile.is_current):
            return None
        if review_protected is _COVERAGE_UNSET:
            review = self.session.scalar(select(WindowReview).where(WindowReview.window_id == analysis.window_id)
                .order_by(WindowReview.created_at.desc(), WindowReview.id.desc()).limit(1))
            review_protected = (review is not None
                and binary_labels(json.loads(review.labels_json))["earthquake"] == "present"
                and not self._review_dict(review, analysis)["stale"])
        payload = json.loads(row.payload_json) | {
            "analysis_id": analysis.id, "input_sha256": analysis.input_sha256,
            "review_protected": bool(review_protected),
        }
        if review_protected:
            payload["score"] = payload.get("coverage_score", payload.get("score"))
            payload["common_mode_penalty"] = 0
        return payload

    def _coverage_analysis(self, wid, analysis_id, input_sha256):
        w, a = self._window(wid), self._analysis(analysis_id)
        profile = self.session.get(ProcessingProfile, a.profile_id)
        if (w.latest_analysis_id != a.id or a.window_id != wid or a.status != "succeeded"
                or a.input_sha256 != input_sha256 or not profile.is_current):
            raise WindowError("stale_analysis")
        return a

    def prepare_spatial_coverage(self, wid, analysis_id, input_sha256):
        """Measure authenticated local caches outside the API write lock."""
        import numpy as np

        from das_sentinel.borehole_coverage import (
            BOREHOLE_COVERAGE_VERSION,
            measure_borehole_coverage,
        )

        a = self._coverage_analysis(wid, analysis_id, input_sha256)
        existing = self._spatial_coverage(a, review_protected=False)
        if existing is not None and existing["algorithm_version"] == BOREHOLE_COVERAGE_VERSION:
            return existing
        result = json.loads(a.result_json)
        raw_path, _ = self.artifact(wid, "waveform", a.id)
        filtered_path, _ = self.artifact(wid, "filtered_waveform", a.id)
        data = self._filtered_array(filtered_path, result["shape"]).T
        with np.load(raw_path, allow_pickle=False) as arrays:
            measured = measure_borehole_coverage(data, arrays["distance_m"],
                arrays["valid_channel_mask"], result["sample_rate_hz"])
        return measured | {"analysis_id": a.id, "input_sha256": a.input_sha256}

    def save_spatial_coverage(self, wid, measured):
        from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_VERSION

        a = self._coverage_analysis(wid, measured["analysis_id"], measured["input_sha256"])
        if measured["algorithm_version"] != BOREHOLE_COVERAGE_VERSION:
            raise WindowError("coverage_version_changed")
        payload = {k: v for k, v in measured.items() if k not in ("analysis_id", "input_sha256", "review_protected")}
        row = self.session.get(WindowSpatialCoverage, a.id)
        if row is None:
            row = WindowSpatialCoverage(analysis_id=a.id)
            self.session.add(row)
        row.input_sha256, row.algorithm_version = a.input_sha256, BOREHOLE_COVERAGE_VERSION
        row.score, row.payload_json = measured["score"], canonical(payload)
        row.created_at = utc_now()
        self.session.flush()
        return self._spatial_coverage(a)

    def jobs(self, *, site_id=None, status=None, limit=30, offset=0, window_seconds=30):
        q = select(WindowAnalysis).join(
            EventWindow, EventWindow.id == WindowAnalysis.window_id
        ).where(EventWindow.window_seconds == window_seconds)
        if site_id:
            q = q.where(EventWindow.site_id == site_id)
        if status:
            q = q.where(WindowAnalysis.status == status)
        total = self.session.scalar(select(func.count()).select_from(q.subquery()))
        rows = self.session.scalars(
            q.order_by(EventWindow.priority.desc(), WindowAnalysis.created_at)
            .offset(offset)
            .limit(limit)
        ).all()
        return {
            "items": [self.job(a) for a in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def detail(self, wid, model_scan_id=None, *, admin_id=None):
        from das_sentinel.services.window_label_imports import import_summary

        w = self._window(wid)
        a = self._analysis(w.latest_analysis_id)
        r = json.loads(a.result_json or "{}")
        artifacts = [
            {
                "kind": v["kind"],
                "url": f"/api/v1/event-windows/{wid}/artifacts/{v['kind']}?analysis_id={a.id}",
            }
            for v in r.get("artifacts", [])
        ]
        return self._item(w, model_scan_id, favorite=wid in self._favorite_ids([wid], admin_id)) | {
            "analysis_error_code": a.error_code,
            "analysis_error_message": a.error_message,
            "features": r.get("features", {}),
            "recognition_range_m": r.get("recognition_range_m"),
            "shape": r.get("shape"),
            "unit": r.get("unit"),
            "machine_labels": [],
            "artifacts": artifacts,
            "visualization_version": r.get("visualization_version"),
            "waterfall_regions_baked": r.get("waterfall_regions_baked", True),
            "detection_regions": r.get("detection_regions", []),
            "plot_geometry": r.get("plot_geometry"),
            "legacy_events": self._legacy(w),
            "legacy_total": len(json.loads(w.legacy_event_ids_json)),
            "legacy_label_import": import_summary(self, w, a),
        }

    def _legacy(self, w):
        ids = json.loads(w.legacy_event_ids_json)[:20]
        rows = self.session.scalars(
            select(Event).where(Event.id.in_(ids)).order_by(Event.occurred_at, Event.id)
        ).all()
        result = []
        for event in rows:
            review = self.session.scalar(
                select(Review)
                .where(Review.event_id == event.id)
                .order_by(Review.created_at.desc(), Review.id.desc())
                .limit(1)
            )
            result.append(
                {
                    "id": event.id,
                    "start_utc": event.occurred_at,
                    "channel_start": event.channel_start,
                    "channel_stop": event.channel_stop,
                    "label": None if review is None else review.label,
                    "note": None if review is None else review.note,
                }
            )
        return result

    def _path(self, a, name):
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or ":" in name
            or name in (".", "..")
        ):
            raise WindowError("invalid_artifact_path", 422)
        folder = self.artifact_root / a.id / f"attempt-{a.attempt_count}"
        path = folder / name
        if (
            folder.is_symlink()
            or folder.parent.is_symlink()
            or path.is_symlink()
            or not path.is_file()
            or folder.resolve() != folder.absolute()
            or path.resolve().parent != folder.resolve()
            or not path.resolve().is_relative_to(self.artifact_root)
        ):
            raise WindowError("invalid_artifact_file", 422)
        if (
            not stat.S_ISREG(path.stat().st_mode)
            or path.stat().st_nlink != 1
            or path.stat().st_size > MAX_ARTIFACT_BYTES
        ):
            raise WindowError("invalid_artifact_file", 422)
        return path

    def _finite_json(self, value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite")
        if isinstance(value, dict):
            for item in value.values():
                self._finite_json(item)
        if isinstance(value, list):
            for item in value:
                self._finite_json(item)
        return value

    def _json(self, path):
        bound = 32 * 1024 * 1024 if path.name == "evidence.json" else MAX_JSON_BYTES
        if path.stat().st_size > bound:
            raise WindowError("json_too_large", 422)
        try:
            return self._finite_json(
                json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)),
                )
            )
        except (ValueError, UnicodeError, RecursionError):
            raise WindowError("invalid_json", 422)

    def result(self, aid, worker_id, attempt_count, *, validate_only=False):
        a = self._owned(aid, worker_id, attempt_count)
        self._window(a.window_id)
        plan = json.loads(a.plan_json)
        r = self._json(self._path(a, "result.json"))
        if not isinstance(r, dict):
            raise WindowError("invalid_result", 422)
        if "waterfall_regions_baked" in r and type(r["waterfall_regions_baked"]) is not bool:
            raise WindowError("invalid_waterfall_regions_baked", 422)
        if (
            not isinstance(r.get("features"), dict)
            or any(
                v is not None and (type(v) not in (int, float))
                for v in r["features"].values()
            )
            or type(r.get("candidate")) is not bool
            or type(r.get("score")) not in (int, float)
        ):
            raise WindowError("invalid_result_features", 422)
        if any(
            r.get(k) != plan.get(k)
            for k in ("start_utc", "end_utc", "sources", "algorithm_version")
        ):
            raise WindowError("result_identity_mismatch")
        if not self._current(a):
            raise WindowError("source_profile_changed")
        manifests = r.get("artifacts", [])
        if not isinstance(manifests, list) or len(manifests) > 10:
            raise WindowError("invalid_manifest", 422)
        expected = {
            "waveform.npz": ("waveform", "application/octet-stream"),
            "metadata.json": ("metadata", "application/json"),
            "evidence.json": ("evidence", "application/json"),
            "waterfall.png": ("waterfall", "image/png"),
            "waveform.png": ("waveform_plot", "image/png"),
        }
        if r.get("visualization_version") == 2:
            expected = {
                "waveform.npz": ("waveform", "application/octet-stream"),
                "metadata_visual_v2.json": ("metadata", "application/json"),
                "evidence.json": ("evidence", "application/json"),
                "waterfall_raw_v2.png": ("waterfall", "image/png"),
                "waterfall_filtered_v2.png": ("waterfall_filtered", "image/png"),
                "filtered_waveform_v2.npz": ("filtered_waveform", "application/octet-stream"),
            }
        elif r.get("visualization_version") is not None:
            raise WindowError("invalid_visualization_version", 422)
        if any(
            not isinstance(m, dict)
            or not isinstance(m.get("path"), str)
            or m.get("path") not in expected
            or (m.get("kind"), m.get("media_type")) != expected[m["path"]]
            for m in manifests
        ):
            raise WindowError("invalid_manifest", 422)
        required = set(expected)
        if {m.get("path") for m in manifests} != required or len(manifests) != len(
            required
        ):
            raise WindowError("invalid_manifest", 422)
        if len({m.get("kind") for m in manifests}) != len(manifests):
            raise WindowError("duplicate_artifact_kind", 422)
        total = 0
        for m in manifests:
            path = self._path(a, m["path"])
            total += path.stat().st_size
            if total > MAX_ARTIFACT_BYTES:
                raise WindowError("artifacts_too_large", 422)
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha != m.get("sha256"):
                raise WindowError("artifact_hash_mismatch", 422)
        evidence = self._json(self._path(a, "evidence.json"))
        if (
            not isinstance(evidence, list)
            or len(evidence) > MAX_EVIDENCE
            or len(evidence) != r.get("evidence_count")
        ):
            raise WindowError("invalid_evidence_count", 422)
        if (
            r.get("schema_version") != 1
            or r.get("sample_rate_hz") != 100
            or not isinstance(r.get("shape"), list)
            or len(r["shape"]) != 2
            or r["shape"][1] != round((datetime.fromisoformat(plan["end_utc"]) - datetime.fromisoformat(plan["start_utc"])).total_seconds() * 100)
        ):
            raise WindowError("invalid_array_contract", 422)
        if not isinstance(r.get("input_sha256"), str) or len(r["input_sha256"]) != 64:
            raise WindowError("invalid_input_signature", 422)
        from das_sentinel.whole_window import compute_input_sha256

        try:
            expected_signature = compute_input_sha256(r)
        except (KeyError, TypeError, ValueError) as error:
            raise WindowError("invalid_input_signature", 422) from error
        if expected_signature != r["input_sha256"]:
            raise WindowError("input_signature_mismatch")
        from das_sentinel.config import ProcessingProfile as ProfileConfig

        if r.get("profile") != ProfileConfig.model_validate(
            plan["profile_payload"]
        ).model_dump(mode="json"):
            raise WindowError("result_profile_mismatch")
        metadata_name = "metadata_visual_v2.json" if r.get("visualization_version") == 2 else "metadata.json"
        metadata = self._json(self._path(a, metadata_name))
        if metadata != {k: v for k, v in r.items() if k != "artifacts"}:
            raise WindowError("metadata_mismatch", 422)
        self._validate_read_provenance(r, plan)
        self._validate_array(self._path(a, "waveform.npz"), r)
        if r.get("visualization_version") == 2:
            self._validate_visualization(a, r, evidence)
        if validate_only:
            return r
        a.result_json = canonical(r)
        a.input_sha256 = r["input_sha256"]
        a.status = "succeeded"
        self.session.flush()
        from das_sentinel.services.window_label_imports import apply_import

        apply_import(self, a)
        return self.job(a)

    def _validate_read_provenance(self, result, plan):
        """Derive the reader's native union slices from the immutable source plan."""
        try:
            start = datetime.fromisoformat(plan["start_utc"])
            end = datetime.fromisoformat(plan["end_utc"])
            conditioning_start = start - timedelta(seconds=30)
            conditioning_end = end + timedelta(seconds=30)
            native_start = conditioning_start - timedelta(seconds=10)
            native_end = conditioning_end + timedelta(seconds=10)
            sources = sorted(
                plan["sources"],
                key=lambda source: (source["start_utc"], source["path"]),
            )
            if (
                not sources
                or datetime.fromisoformat(sources[0]["start_utc"]) > native_start
                or datetime.fromisoformat(sources[-1]["end_utc"]) < native_end
            ):
                raise ValueError("missing native context")
            expected = []
            for index, source in enumerate(sources):
                begin = datetime.fromisoformat(source["start_utc"])
                stop = datetime.fromisoformat(source["end_utc"])
                fs = source["sample_rate_hz"]
                count = round((stop - begin).total_seconds() * fs)
                first = max(
                    0, math.floor((native_start - begin).total_seconds() * fs) - 1
                )
                last = min(
                    count, math.ceil((native_end - begin).total_seconds() * fs) + 1
                )
                if index + 1 < len(sources):
                    next_start = datetime.fromisoformat(sources[index + 1]["start_utc"])
                    last = min(
                        last,
                        math.ceil((next_start - begin).total_seconds() * fs - 1e-7),
                    )
                if last <= first:
                    continue
                previous = sources[index - 1]["end_utc"] if index else None
                seam = (
                    (begin - datetime.fromisoformat(previous)).total_seconds()
                    if previous
                    else None
                )
                if seam is not None and abs(seam) > min(1 / fs, 0.05) + 1e-9:
                    raise ValueError("source gap")
                expected.append(
                    {
                        key: source[key]
                        for key in (
                            "path",
                            "fingerprint",
                            "size_bytes",
                            "mtime_ns",
                            "sample_rate_hz",
                            "unit",
                        )
                    }
                    | {
                        "sample_start": first,
                        "sample_stop": last,
                        "channel_start": 0,
                        "channel_stop": len(source["distance_m"]),
                        "source_start_utc": source["start_utc"],
                        "source_stop_utc": source["end_utc"],
                        "read_start_utc": (
                            begin + timedelta(seconds=first / fs)
                        ).isoformat(),
                        "read_stop_utc": (
                            begin + timedelta(seconds=last / fs)
                        ).isoformat(),
                        "source_verified_unchanged": True,
                        "boundary_utc": source["start_utc"],
                        "previous_source_end_utc": previous,
                        "previous_seam_seconds": seam,
                    }
                )
            if (
                result["conditioning_start_utc"] != conditioning_start.isoformat()
                or result["conditioning_end_utc"] != conditioning_end.isoformat()
                or result["read_sources"] != expected
                or result["read_start_utc"]
                != min(source["read_start_utc"] for source in expected)
                or result["read_end_utc"]
                != max(source["read_stop_utc"] for source in expected)
            ):
                raise ValueError("read bounds or source slices differ")
        except (
            ValueError,
            KeyError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
        ) as error:
            raise WindowError("read_provenance_mismatch") from error

    def _validate_array(self, path, r):
        import zipfile

        import numpy as np

        from das_sentinel.config import ProcessingProfile as ProfileConfig
        from das_sentinel.signal.scan_distance import scan_channel_mask

        keys = {
            "waveform",
            "distance_m",
            "relative_seconds",
            "valid_channel_mask",
            "recognition_channel_mask",
            "region_id",
        }
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if (
                    len(entries) != 6
                    or sum(v.file_size for v in entries) > MAX_ARTIFACT_BYTES
                ):
                    raise ValueError("array archive limit")
            with np.load(path, allow_pickle=False) as array:
                if set(array.files) != keys:
                    raise ValueError("array keys")
                waveform = array["waveform"]
                distance = array["distance_m"]
                seconds = array["relative_seconds"]
                valid = array["valid_channel_mask"]
                recognition = array["recognition_channel_mask"]
                region = array["region_id"]
                n = r["shape"][0]
                if (
                    waveform.shape != tuple(r["shape"])
                    or distance.shape != (n,)
                    or seconds.shape != (r["shape"][1],)
                    or valid.shape != (n,)
                    or recognition.shape != (n,)
                    or region.shape != (n,)
                ):
                    raise ValueError("array shape")
                if not all(
                    np.isfinite(v).all() for v in (waveform, distance, seconds, region)
                ):
                    raise ValueError("array nonfinite")
                if (
                    valid.dtype != np.bool_
                    or recognition.dtype != np.bool_
                    or (recognition & ~valid).any()
                ):
                    raise ValueError("array masks")
                if (
                    valid.tolist() != r["valid_channel_mask"]
                    or recognition.tolist() != r["recognition_channel_mask"]
                ):
                    raise ValueError("array metadata masks")
                profile = ProfileConfig.model_validate(r["profile"])
                if not np.array_equal(valid, np.ptp(waveform, axis=1) > 0):
                    raise ValueError("array validity mask")
                if not np.array_equal(
                    recognition, valid & scan_channel_mask(distance, profile)
                ):
                    raise ValueError("array recognition mask")
                if r["recognition_range_m"] != {
                    "start": profile.scan_distance_start_m,
                    "stop": profile.scan_distance_stop_m,
                }:
                    raise ValueError("array recognition range")
                if region.dtype.kind not in "iu" or not np.array_equal(
                    region, np.where(distance < 299, 0, np.where(distance < 499, 1, 2))
                ):
                    raise ValueError("array region map")
                if not np.array_equal(seconds, np.arange(r["shape"][1]) / 100):
                    raise ValueError("array time axis")
                if (
                    hashlib.sha256(
                        np.asarray(distance, dtype="<f8").tobytes()
                    ).hexdigest()
                    != r["grid_sha256"]
                ):
                    raise ValueError("array grid hash")
                if r["sources"] and any(
                    source["distance_m"] != distance.tolist()
                    or source["unit"] != r["unit"]
                    for source in r["sources"]
                ):
                    raise ValueError("array source grid or unit")
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as error:
            raise WindowError("invalid_array: " + str(error), 422) from error

    def _filtered_array(self, path, shape):
        import zipfile

        import numpy as np

        try:
            with zipfile.ZipFile(path) as archive:
                if len(archive.infolist()) != 1 or sum(v.file_size for v in archive.infolist()) > MAX_ARTIFACT_BYTES:
                    raise ValueError("filtered archive limit")
            with np.load(path, allow_pickle=False) as arrays:
                if arrays.files != ["waveform"]:
                    raise ValueError("filtered array keys")
                data = arrays["waveform"]
                if data.shape != tuple(shape) or data.dtype != np.float32 or not np.isfinite(data).all():
                    raise ValueError("filtered array shape or values")
                return data
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
            raise WindowError("invalid_filtered_array", 422) from error

    def _validate_visualization(self, a, result, evidence):
        import numpy as np

        from das_sentinel.whole_window_plots import PLOT_AXES
        from das_sentinel.window_regions import distance_edges, merge_regions

        try:
            with np.load(self._path(a, "waveform.npz"), allow_pickle=False) as arrays:
                distance = arrays["distance_m"]
                recognition = arrays["recognition_channel_mask"]
            expected = merge_regions(evidence, distance, recognition, result["shape"][1],
                                     round(result["profile"]["merge_gap_seconds"] * 100))
            edges = distance_edges(distance)
            geometry = {"axes": PLOT_AXES, "distance_min_m": float(edges[0]), "distance_max_m": float(edges[-1])}
            if result.get("detection_regions") != expected or result.get("plot_geometry") != geometry:
                raise ValueError("regions or geometry do not match evidence")
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            raise WindowError("invalid_visualization", 422) from error
        self._filtered_array(self._path(a, "filtered_waveform_v2.npz"), result["shape"])

    def _preview_samples(self, wid, analysis_id, filtered):
        import numpy as np

        # artifact() verifies analysis ownership, status, path and content hash.
        raw_path, _ = self.artifact(wid, "waveform", analysis_id)
        a = self._analysis(analysis_id)
        result = json.loads(a.result_json)
        with np.load(raw_path, allow_pickle=False) as arrays:
            distance = arrays["distance_m"]
            recognition = arrays["recognition_channel_mask"]
            data = arrays["waveform"].T
        if filtered:
            path, _ = self.artifact(wid, "filtered_waveform", analysis_id)
            data = self._filtered_array(path, result["shape"]).T
        return result, data, distance, recognition

    def waterfall_preview(self, wid, analysis_id, filtered=False):
        from das_sentinel.whole_window_plots import render_waterfall_preview

        # Preserve waveform verification even when the matching plain PNG can be reused.
        self.artifact(wid, "waveform", analysis_id)
        result = json.loads(self._analysis(analysis_id).result_json)
        if result.get("visualization_version") == 2 and result.get("waterfall_regions_baked") is False:
            if filtered:
                self.artifact(wid, "filtered_waveform", analysis_id)
            path, _ = self.artifact(wid, "waterfall_filtered" if filtered else "waterfall", analysis_id)
            return path.read_bytes()
        result, data, distance, recognition = self._preview_samples(wid, analysis_id, filtered)
        return render_waterfall_preview(data, distance, datetime.fromisoformat(result["start_utc"]),
                                        recognition, result["unit"], filtered=filtered,
                                        site_timezone=result.get("site_timezone", "Asia/Shanghai"))

    def region_preview(self, wid, analysis_id, region_id, filtered=False, show_regions=True):
        from das_sentinel.whole_window_plots import render_region_preview

        result, data, distance, recognition = self._preview_samples(wid, analysis_id, filtered)
        region = next((v for v in result.get("detection_regions", []) if v["id"] == region_id), None)
        if region is None:
            raise WindowError("region_not_found", 404)
        return render_region_preview(data, distance, datetime.fromisoformat(result["start_utc"]),
                                     recognition, result["unit"], region, filtered=filtered,
                                     site_timezone=result.get("site_timezone", "Asia/Shanghai"),
                                     show_regions=show_regions)

    def artifact(self, wid, kind, analysis_id=None):
        w = self._window(wid)
        a = self._analysis(analysis_id or w.latest_analysis_id)
        if a.window_id != wid or a.status != "succeeded":
            raise WindowError("analysis_window_mismatch")
        r = json.loads(a.result_json)
        m = next((v for v in r["artifacts"] if v["kind"] == kind), None)
        if m is None:
            raise WindowError("artifact_not_found", 404)
        path = self._path(a, m["path"])
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if sha != m["sha256"]:
            raise WindowError("artifact_hash_mismatch", 422)
        return path, m["media_type"]

    def evidence(self, wid, *, analysis_id=None, limit=50, offset=0):
        if limit < 1 or limit > 100 or offset < 0:
            raise WindowError("invalid_pagination", 422)
        path, _ = self.artifact(wid, "evidence", analysis_id)
        evidence = self._json(path)
        if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE:
            raise WindowError("invalid_evidence_count", 422)
        return {
            "items": evidence[offset : offset + limit],
            "total": len(evidence),
            "limit": limit,
            "offset": offset,
        }
