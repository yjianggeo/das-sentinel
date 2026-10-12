"""Short, caller-owned transactions for durable model-first scanning."""

import json
import math
from datetime import timedelta
from pathlib import Path

from sqlalchemy import case, delete, func, literal, select, update

from das_sentinel.model_scan_contract import (
    PROTOCOL,
    canonical,
    digest,
    excluded_two_metre_source,
    inference_key,
    utc,
)
from das_sentinel.persistence.model_scan_models import (
    ModelScan,
    ModelScanReceipt,
    ModelScanSource,
    ModelScanUnit,
)
from das_sentinel.persistence.models import (
    Job,
    ProcessingProfile,
    Site,
    SourceFile,
    utc_now,
)
from das_sentinel.persistence.window_models import (
    EventWindow,
    WindowAnalysis,
    WindowReview,
)
from das_sentinel.services.event_windows import WindowError
from das_sentinel.whole_window import window_id, window_start

LEASE_SECONDS = 180


class ModelScanService:
    def __init__(self, session, app_data_dir):
        self.session, self.root = session, Path(app_data_dir)

    def _scan(self, scan_id):
        row = self.session.get(ModelScan, scan_id)
        if row is None:
            raise WindowError("model_scan_not_found", 404)
        return row

    def create_checked(self, site_id, profile_id, *args, **kwargs):
        profile = self.session.get(ProcessingProfile, profile_id)
        if profile is None or not profile.is_current:
            raise WindowError("source_profile_changed")
        return self.create(site_id, profile_id, *args, **kwargs)

    def list(self, *, site_id=None, limit=20, offset=0):
        query = select(ModelScan)
        if site_id:
            query = query.where(ModelScan.site_id == site_id)
        total = self.session.scalar(select(func.count()).select_from(query.subquery()))
        rows = self.session.scalars(query.order_by(ModelScan.created_at.desc(), ModelScan.id).offset(offset).limit(limit))
        return {"items": [self.get(r.id) for r in rows], "total": total, "limit": limit, "offset": offset}

    def create(self, site_id, profile_id, request_id, directories, model, exclusions, *, time_ranges=None):
        site, profile = self.session.get(Site, site_id), self.session.get(ProcessingProfile, profile_id)
        if site is None or profile is None or profile.site_id != site_id:
            raise WindowError("profile_site_mismatch", 422)
        config = {"protocol": PROTOCOL, "directories": sorted(set(directories)), "model": model,
                  "exclusions": exclusions, "time_ranges": time_ranges, "root": site.data_path,
                  "profile_id": profile_id, "profile_payload": json.loads(profile.payload_json),
                  "site_timezone": site.timezone}
        signature = digest(config)
        previous = self.session.scalar(select(ModelScan).where(
            ModelScan.site_id == site_id, ModelScan.request_id == request_id))
        if previous is not None:
            if previous.config_sha256 != signature:
                raise WindowError("request_conflict")
            return self.get(previous.id)
        row = ModelScan(site_id=site_id, profile_id=profile_id, request_id=request_id,
                        config_json=canonical(config), config_sha256=signature)
        self.session.add(row)
        self.session.flush()
        for directory in config["directories"]:
            self._add_unit(row.id, "discover", digest(directory), {"directory": directory})
        # Only registered files are context; never recursively discover the separate eq archive.
        paths = list(dict.fromkeys(self.session.scalars(select(SourceFile.path).where(
            SourceFile.site_id == site_id).order_by(SourceFile.discovered_at.desc()))))
        for offset in range(0, len(paths), 256):
            group = paths[offset:offset + 256]
            self._add_unit(row.id, "discover", digest(group), {"context_paths": group})
        return self.get(row.id)

    def _add_unit(self, scan_id, kind, key, payload):
        row = self.session.scalar(select(ModelScanUnit).where(
            ModelScanUnit.scan_id == scan_id, ModelScanUnit.kind == kind, ModelScanUnit.unit_key == key))
        if row is None:
            row = ModelScanUnit(scan_id=scan_id, kind=kind, unit_key=key, payload_json=canonical(payload),
                                window_count=len(payload.get("windows", [])))
            self.session.add(row)
            self.session.flush()
        elif row.payload_json != canonical(payload):
            raise WindowError("unit_conflict")
        return row

    def _job(self, row):
        scan = self._scan(row.scan_id)
        payload = json.loads(row.payload_json)
        if row.kind == "plan":
            payload = self._planning_payload(row)
        elif row.kind == "infer":
            completed = set(self.session.scalars(select(ModelScanReceipt.start_utc).where(
                ModelScanReceipt.infer_unit_id == row.id, ModelScanReceipt.status != "pending")))
            payload["windows"] = [p for p in payload["windows"] if utc(p["start_utc"]) not in completed]
        job = {"id": row.id, "scan_id": row.scan_id, "site_id": scan.site_id,
                "kind": row.kind, "status": row.status, "attempt_count": row.attempt_count,
                "payload": payload, "config": json.loads(scan.config_json)}
        if row.kind == "materialize" and row.status == "running":
            receipt = self.session.get(ModelScanReceipt, payload["receipt_id"])
            if receipt.analysis_id:
                from das_sentinel.services.event_windows import WindowService
                analysis = self.session.get(WindowAnalysis, receipt.analysis_id)
                job["analysis"] = WindowService(self.session, self.root / "window-artifacts").job(analysis) | {
                    "skip_empty": False, "reuse": analysis.status == "succeeded"}
                job["payload"]["model_input_sha256"] = receipt.model_input_sha256
                job["payload"]["screening_score"] = receipt.score
        return job

    def _owned(self, unit_id, worker_id, attempt_count, *, replay=False):
        row = self.session.get(ModelScanUnit, unit_id)
        if (row is None or row.worker_id != worker_id or row.attempt_count != attempt_count
                or row.status not in (("running", "succeeded") if replay else ("running",))
                or self._scan(row.scan_id).status == "cancelled"):
            raise WindowError("claim_lost")
        if row.status == "running" and (row.heartbeat_at is None or
                row.heartbeat_at < utc_now() - timedelta(seconds=LEASE_SECONDS)):
            raise WindowError("claim_lost")
        return row

    def claim(self, worker_id, kind=None):
        if not worker_id.strip() or len(worker_id) > 200:
            raise WindowError("invalid_worker", 422)
        now = utc_now()
        scans = select(ModelScan.id).where(ModelScan.status.in_(("queued", "running")))
        stale = self.session.scalars(select(ModelScanUnit).where(
            ModelScanUnit.scan_id.in_(scans), ModelScanUnit.status == "running",
            ModelScanUnit.heartbeat_at < now - timedelta(seconds=LEASE_SECONDS))).all()
        for row in stale:
            row.status = "failed" if row.attempt_count >= 3 else "queued"
            row.error_code = "lease_expired"
        self.session.flush()
        query = select(ModelScanUnit).where(ModelScanUnit.scan_id.in_(scans), ModelScanUnit.status == "queued")
        if kind:
            query = query.where(ModelScanUnit.kind == kind)
        priority = case({"materialize": 0, "infer": 1, "plan": 2, "discover": 3}, value=ModelScanUnit.kind)
        row = self.session.scalar(query.order_by(priority, ModelScanUnit.created_at, ModelScanUnit.id).limit(1))
        if row is None:
            for scan_id in self.session.scalars(scans):
                self._refresh(scan_id)
            return None
        changed = self.session.execute(update(ModelScanUnit).where(
            ModelScanUnit.id == row.id, ModelScanUnit.status == "queued").values(
                status="running", worker_id=worker_id, attempt_count=ModelScanUnit.attempt_count + 1,
                heartbeat_at=now, started_at=now, finished_at=None, error_code=None, error_message=None))
        if changed.rowcount != 1:
            raise WindowError("claim_lost")
        self.session.refresh(row)
        if row.kind == "discover" and row.attempt_count > 1:
            # Planning starts only after every directory has a complete manifest.
            # A crashed enumeration is rebuilt atomically, never mixed with old pages.
            self.session.execute(delete(ModelScanSource).where(ModelScanSource.discovery_unit_id == row.id))
            row.pages_json = "{}"
        self._scan(row.scan_id).status = "running"
        if row.kind == "materialize":
            self._claim_materialization(row)
        return self._job(row)

    def heartbeat(self, unit_id, worker_id, attempt_count):
        row = self._owned(unit_id, worker_id, attempt_count)
        row.heartbeat_at = utc_now()
        if row.kind == "materialize":
            receipt = self.session.get(ModelScanReceipt, json.loads(row.payload_json)["receipt_id"])
            if receipt.analysis_id:
                analysis = self.session.get(WindowAnalysis, receipt.analysis_id)
                if (analysis.status == "running" and analysis.worker_id == worker_id
                        and analysis.attempt_count == receipt.analysis_attempt_count):
                    analysis.heartbeat_at = row.heartbeat_at
        self.session.flush()
        return {"status": "running"}

    def control(self, scan_id, action):
        scan = self._scan(scan_id)
        if scan.status == "cancelled":
            if action == "cancel":
                return self.get(scan_id)
            raise WindowError("model_scan_cancelled")
        if action == "pause":
            if scan.status in ("queued", "running"):
                scan.status = "paused"
        elif action == "resume":
            if scan.status != "paused":
                raise WindowError("model_scan_not_paused")
            scan.status = "queued"
        elif action == "cancel":
            scan.status = "cancelled"
            for unit in self.session.scalars(select(ModelScanUnit).where(
                    ModelScanUnit.scan_id == scan_id, ModelScanUnit.kind == "materialize")):
                receipt = self.session.get(ModelScanReceipt, json.loads(unit.payload_json)["receipt_id"])
                if receipt.analysis_id:
                    self.session.execute(update(WindowAnalysis).where(
                        WindowAnalysis.id == receipt.analysis_id,
                        WindowAnalysis.worker_id == unit.worker_id,
                        WindowAnalysis.attempt_count == receipt.analysis_attempt_count,
                        WindowAnalysis.status.in_(("queued", "running", "failed"))).values(status="cancelled"))
            self.session.execute(update(ModelScanUnit).where(ModelScanUnit.scan_id == scan_id,
                ModelScanUnit.status.in_(("queued", "running", "failed"))).values(status="cancelled"))
            self.session.execute(update(ModelScanReceipt).where(ModelScanReceipt.scan_id == scan_id,
                ModelScanReceipt.status.in_(("selected", "pending_artifacts"))).values(status="cancelled"))
        elif action == "retry":
            for receipt in self.session.scalars(select(ModelScanReceipt).where(
                    ModelScanReceipt.scan_id == scan_id, ModelScanReceipt.status == "unjudgeable")):
                unit = self.session.get(ModelScanUnit, receipt.infer_unit_id)
                # Coverage failures require new source discovery; reader failures can retry the frozen plan.
                if any(utc(p["start_utc"]) == receipt.start_utc for p in json.loads(unit.payload_json)["windows"]):
                    receipt.status, receipt.result_json, receipt.result_sha256 = "pending", "{}", ""
                    receipt.finished_at = receipt.error_code = receipt.error_message = None
                    unit.status = "failed"
            self.session.flush()
            self.session.execute(update(ModelScanUnit).where(ModelScanUnit.scan_id == scan_id,
                ModelScanUnit.status == "failed").values(status="queued",
                                                       worker_id=None, error_code=None, error_message=None))
            scan.status = "queued"
        else:
            raise WindowError("invalid_scan_action", 422)
        scan.updated_at = utc_now()
        self.session.flush()
        return self.get(scan_id)

    def fail(self, unit_id, worker_id, attempt_count, error_code, error_message):
        row = self._owned(unit_id, worker_id, attempt_count)
        row.status, row.error_code, row.error_message = "failed", error_code[:100], error_message[:2000]
        row.finished_at = utc_now()
        if row.kind == "materialize":
            receipt = self.session.get(ModelScanReceipt, json.loads(row.payload_json)["receipt_id"])
            receipt.error_code, receipt.error_message = row.error_code, row.error_message
            if receipt.analysis_id:
                analysis = self.session.get(WindowAnalysis, receipt.analysis_id)
                if (analysis.status == "running" and analysis.worker_id == worker_id
                        and analysis.attempt_count == receipt.analysis_attempt_count):
                    analysis.status, analysis.error_code, analysis.error_message = "failed", row.error_code, row.error_message
        self.session.flush()
        self._refresh(row.scan_id)
        return self._job(row)

    @staticmethod
    def _receipt(row):
        return {"id": row.id, "status": row.status, "start_utc": row.start_utc.isoformat(),
                "score": row.score, "analysis_id": row.analysis_id, "window_id": row.window_id}

    def record_prediction(self, unit_id, worker_id, attempt_count, result):
        unit = self._owned(unit_id, worker_id, attempt_count, replay=True)
        if unit.kind != "infer":
            raise WindowError("wrong_scan_phase")
        scan = self._scan(unit.scan_id)
        config = json.loads(scan.config_json)
        model = config["model"]
        plans = json.loads(unit.payload_json)["windows"]
        try:
            start = utc(result["start_utc"])
            plan = next(p for p in plans if utc(p["start_utc"]) == start)
            if result["status"] == "succeeded":
                score = result["score"]
                sha = result["model_input_sha256"]
                if (isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 1
                        or result["model_id"] != model["id"]
                        or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
                    raise ValueError()
                status = "selected" if score >= model["threshold"] else "negative"
            elif result["status"] == "failed" and result.get("error_code"):
                if "score" in result or "label" in result:
                    raise ValueError()
                score, sha, status = None, None, "unjudgeable"
            else:
                raise ValueError()
            signature = digest(result)
        except (ValueError, TypeError, KeyError, StopIteration):
            raise WindowError("invalid_prediction", 422) from None
        previous = self.session.scalar(select(ModelScanReceipt).where(
            ModelScanReceipt.scan_id == scan.id, ModelScanReceipt.start_utc == start))
        if previous is not None and previous.status != "pending":
            if previous.result_sha256 != signature or previous.infer_unit_id != unit.id:
                raise WindowError("result_conflict")
            return self._receipt(previous)
        if unit.status != "running":
            raise WindowError("claim_lost")
        receipt = previous or ModelScanReceipt(scan_id=scan.id, infer_unit_id=unit.id, start_utc=start,
            inference_key=inference_key(scan.site_id, start, plan["sources"], model))
        if receipt.infer_unit_id != unit.id:
            raise WindowError("result_conflict")
        receipt.status, receipt.score, receipt.model_input_sha256 = status, score, sha
        receipt.result_json, receipt.result_sha256 = canonical(result), signature
        receipt.error_code, receipt.error_message = result.get("error_code"), result.get("error_message")
        receipt.finished_at = utc_now() if status != "selected" else None
        if previous is None:
            self.session.add(receipt)
        self.session.flush()
        if status == "selected":
            self._add_unit(scan.id, "materialize", receipt.id, {"receipt_id": receipt.id, "plan": plan})
        count = self.session.scalar(select(func.count()).select_from(ModelScanReceipt).where(
            ModelScanReceipt.infer_unit_id == unit.id, ModelScanReceipt.status != "pending",
            ModelScanReceipt.start_utc.in_([utc(p["start_utc"]) for p in plans])))
        if count == len(plans):
            unit.status, unit.finished_at = "succeeded", utc_now()
        self.session.flush()
        self._refresh(scan.id)
        return self._receipt(receipt)

    def _claim_materialization(self, unit):
        from das_sentinel.services.event_windows import enqueue_windows
        payload = json.loads(unit.payload_json)
        receipt = self.session.get(ModelScanReceipt, payload["receipt_id"])
        scan = self._scan(unit.scan_id)
        wid = window_id(scan.site_id, receipt.start_utc, 30)
        if self.session.scalar(select(WindowReview.id).where(WindowReview.window_id == wid).limit(1)):
            receipt.status, receipt.finished_at = "protected", utc_now()
            unit.status, unit.finished_at = "succeeded", utc_now()
            unit.payload_json = canonical(payload | {"protected": True})
            self.session.flush()
            return
        current = self.session.get(EventWindow, wid)
        previous = self.session.get(WindowAnalysis, current.latest_analysis_id) if current and current.latest_analysis_id else None
        if previous and previous.status == "cancelled":
            receipt.status, receipt.finished_at = "excluded", utc_now()
            unit.status, unit.finished_at = "succeeded", utc_now()
            unit.payload_json = canonical(payload | {"protected": True})
            self.session.flush()
            return
        analysis = self.session.get(WindowAnalysis, receipt.analysis_id) if receipt.analysis_id else None
        if analysis is None:
            analysis = enqueue_windows(self.session, scan.site_id, scan.profile_id, [payload["plan"]],
                                       model_selection=True)[0]
            receipt.window_id, receipt.analysis_id = wid, analysis.id
        if analysis.status == "cancelled":
            receipt.status, receipt.finished_at = "excluded", utc_now()
            unit.status, unit.finished_at = "succeeded", utc_now()
            unit.payload_json = canonical(payload | {"protected": True})
            self.session.flush()
            return
        current = self.session.get(EventWindow, wid)
        if current.latest_analysis_id != analysis.id:
            unit.status, unit.error_code, unit.finished_at = "failed", "analysis_binding_changed", utc_now()
            unit.error_message = "Window analysis changed; existing binding retained"
            receipt.error_code, receipt.error_message = unit.error_code, unit.error_message
            self.session.flush()
            return
        if (analysis.status == "running" and analysis.heartbeat_at is not None and
                analysis.heartbeat_at >= utc_now() - timedelta(seconds=LEASE_SECONDS)):
            unit.status, unit.error_code, unit.finished_at = "failed", "analysis_busy", utc_now()
            unit.error_message = "Another task owns this window analysis; retry after it finishes"
            receipt.error_code, receipt.error_message = unit.error_code, unit.error_message
            self.session.flush()
            return
        if analysis.status != "succeeded":
            analysis.status, analysis.worker_id = "running", unit.worker_id
            analysis.attempt_count += 1
            analysis.heartbeat_at = utc_now()
            analysis.skip_policy_version = None
            receipt.analysis_attempt_count = analysis.attempt_count
        receipt.status = "pending_artifacts"
        self.session.flush()

    def validate_materialization(self, unit_id, worker_id, attempt_count, binding):
        """Read/validate local artifacts outside the API write lock; does not mutate rows."""
        import numpy as np

        from das_sentinel.model_scan_input import arrays_sha256
        from das_sentinel.services.event_windows import WindowService
        from das_sentinel.window_inference import (
            checked_metadata,
            deployment_folder,
            normalize_prediction_row,
        )
        unit = self._owned(unit_id, worker_id, attempt_count, replay=True)
        receipt = self.session.get(ModelScanReceipt, json.loads(unit.payload_json)["receipt_id"])
        if unit.status == "succeeded" and receipt.status == "ready":
            return {"replay": True, "binding": binding}
        scan = self._scan(unit.scan_id)
        config = json.loads(scan.config_json)
        analysis = self.session.get(WindowAnalysis, receipt.analysis_id)
        if binding.get("analysis_id") != analysis.id:
            raise WindowError("input_mismatch")
        windows = WindowService(self.session, self.root / "window-artifacts")
        if analysis.status == "succeeded":
            result = json.loads(analysis.result_json)
        else:
            result = windows.result(analysis.id, worker_id, analysis.attempt_count, validate_only=True)
        if not windows._current(analysis):
            raise WindowError("source_profile_changed")
        model = config["model"]
        meta = checked_metadata(deployment_folder(self.root / "window-models", scan.site_id, model["id"]), model)
        artifact = next(a for a in result["artifacts"] if a["kind"] == "waveform")
        path = windows._path(analysis, artifact["path"])
        normalized = normalize_prediction_row({"window_id": analysis.window_id, "window_seconds": 30,
            "input_sha256": result["input_sha256"], "sources": result["read_sources"], "unit": result["unit"],
            "grid_sha256": result["grid_sha256"], "waveform": str(path), "waveform_sha256": artifact["sha256"]},
            meta, self.root / "msc" / model["id"])
        with np.load(normalized["waveform"], allow_pickle=False) as arrays:
            signature = arrays_sha256(arrays, receipt.start_utc, result["unit"])
        score = binding.get("score")
        if (signature != receipt.model_input_sha256 or binding.get("model_input_sha256") != signature
                or binding.get("input_sha256") != result["input_sha256"]
                or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                or not model["threshold"] <= score <= 1 or abs(score - receipt.score) > 1e-6):
            raise WindowError("input_mismatch")
        return {"result": result, "binding": binding, "model_input_sha256": signature}

    def finish_materialization(self, unit_id, worker_id, attempt_count, validated):
        from das_sentinel.services.event_windows import WindowService
        unit = self._owned(unit_id, worker_id, attempt_count, replay=True)
        receipt = self.session.get(ModelScanReceipt, json.loads(unit.payload_json)["receipt_id"])
        analysis = self.session.get(WindowAnalysis, receipt.analysis_id)
        binding = validated["binding"]
        if receipt.status == "ready" and unit.status == "succeeded":
            if (binding.get("input_sha256") != receipt.input_sha256 or binding.get("score") != receipt.final_score
                    or binding.get("analysis_id") != receipt.analysis_id):
                raise WindowError("result_conflict")
            return self._receipt(receipt)
        if self.session.scalar(select(WindowReview.id).where(WindowReview.window_id == analysis.window_id).limit(1)):
            receipt.status, receipt.finished_at = "protected", utc_now()
        else:
            windows = WindowService(self.session, self.root / "window-artifacts")
            if (self.session.get(EventWindow, analysis.window_id).latest_analysis_id != analysis.id
                    or not windows._current(analysis)
                    or (analysis.status != "succeeded" and (analysis.status != "running"
                        or analysis.worker_id != worker_id or analysis.attempt_count != receipt.analysis_attempt_count))):
                raise WindowError("claim_lost")
            result = validated["result"]
            analysis.result_json, analysis.input_sha256, analysis.status = canonical(result), result["input_sha256"], "succeeded"
            receipt.input_sha256, receipt.final_score = analysis.input_sha256, binding["score"]
            receipt.status, receipt.finished_at = "ready", utc_now()
            receipt.error_code, receipt.error_message = None, None
        unit.status, unit.finished_at = "succeeded", utc_now()
        self.session.flush()
        self._refresh(unit.scan_id)
        return self._receipt(receipt)

    def commit_discovery_page(self, unit_id, worker_id, attempt_count, page_index, payload):
        unit = self._owned(unit_id, worker_id, attempt_count, replay=True)
        if unit.kind != "discover" or page_index < 0 or sum(len(v) for v in payload.values()) > 256:
            raise WindowError("invalid_discovery_page", 422)
        pages, signature = json.loads(unit.pages_json), digest(payload)
        key = str(page_index)
        if key in pages:
            if pages[key]["sha256"] != signature:
                raise WindowError("page_conflict")
            return {"sha256": signature}
        if unit.status != "running":
            raise WindowError("claim_lost")
        discovery = json.loads(unit.payload_json)
        context_paths = discovery.get("context_paths")
        directory = Path(discovery["directory"]) if not context_paths else None
        for child in payload.get("directories", []):
            if context_paths or Path(child).parent != directory:
                raise WindowError("discovery_path_outside_directory", 422)
            self._add_unit(unit.scan_id, "discover", digest(child), {"directory": child})
        for source in payload.get("sources", []):
            path = Path(source["path"])
            if (str(path) not in context_paths if context_paths else path.parent != directory) or path.suffix.lower() != ".h5":
                raise WindowError("discovery_path_outside_directory", 422)
            previous = self.session.scalar(select(ModelScanSource).where(
                ModelScanSource.scan_id == unit.scan_id, ModelScanSource.path == str(path)))
            if previous is not None:
                if previous.metadata_json != canonical(source):
                    raise WindowError("source_changed")
                if not context_paths:
                    previous.context_only = False
                    previous.discovery_unit_id = unit.id
                continue
            start, end = utc(source["start_utc"]), utc(source["end_utc"])
            if end <= start:
                raise WindowError("invalid_source_time", 422)
            self.session.add(ModelScanSource(scan_id=unit.scan_id, discovery_unit_id=unit.id,
                path=str(path), fingerprint=source["fingerprint"], metadata_json=canonical(source),
                start_utc=start, end_utc=end, context_only=bool(context_paths)))
        pages[key] = {"sha256": signature, "errors": payload.get("errors", [])}
        unit.pages_json = canonical(pages)
        self.session.flush()
        return {"sha256": signature}

    def finish_discovery(self, unit_id, worker_id, attempt_count, page_count, manifest_sha256):
        unit = self._owned(unit_id, worker_id, attempt_count, replay=True)
        pages = json.loads(unit.pages_json)
        if (unit.kind != "discover" or set(pages) != {str(i) for i in range(page_count)}
                or digest([pages[str(i)]["sha256"] for i in range(page_count)]) != manifest_sha256):
            raise WindowError("incomplete_discovery")
        if unit.status == "succeeded":
            return {"status": "succeeded"}
        unit.status, unit.finished_at = "succeeded", utc_now()
        self.session.flush()
        pending = self.session.scalar(select(func.count()).select_from(ModelScanUnit).where(
            ModelScanUnit.scan_id == unit.scan_id, ModelScanUnit.kind == "discover",
            ModelScanUnit.status != "succeeded"))
        if not pending:
            # One cursor plans at most five minutes per claim, using the durable time index.
            planner = self._add_unit(unit.scan_id, "plan", "coverage", {"cursor": None})
            planner.pages_json = canonical({"total": self._total_windows(unit.scan_id)})
        return {"status": "succeeded"}

    def _total_windows(self, scan_id):
        """Count the union of aligned source intervals, without allocating each window."""
        config = json.loads(self._scan(scan_id).config_json)
        ranges = config.get("time_ranges")
        intervals = []
        for first, last in self.session.execute(select(ModelScanSource.start_utc, ModelScanSource.end_utc)
                .where(ModelScanSource.scan_id == scan_id, ModelScanSource.context_only == False).order_by(ModelScanSource.start_utc)):
            lo, hi = math.floor(first.timestamp() / 30), math.ceil(last.timestamp() / 30)
            bounds = [(lo, hi)] if not ranges else [(max(lo, math.ceil(utc(r["start_utc"]).timestamp() / 30)),
                min(hi, math.floor(utc(r["end_utc"]).timestamp() / 30))) for r in ranges]
            for left, right in bounds:
                if left < right:
                    intervals.append((left, right))
        total, stop = 0, None
        for left, right in sorted(intervals):
            total += max(0, right - max(left, stop if stop is not None else left))
            stop = max(right, stop if stop is not None else right)
        return total

    def _planning_payload(self, unit):
        payload = json.loads(unit.payload_json)
        config = json.loads(self._scan(unit.scan_id).config_json)
        query = select(ModelScanSource).where(ModelScanSource.scan_id == unit.scan_id, ModelScanSource.context_only == False)
        cursor = utc(payload["cursor"]) if payload.get("cursor") else None
        if cursor is not None:
            query = query.where(ModelScanSource.end_utc > cursor)
        ranges = config.get("time_ranges")
        bounds = sorted((utc(r["start_utc"]), utc(r["end_utc"])) for r in ranges) if ranges else [(None, None)]
        start = end = None
        for lower, upper in bounds:
            lower = max(lower, cursor) if lower and cursor else lower or cursor
            if lower:
                aligned = window_start(lower, 30)
                lower = aligned + timedelta(seconds=30) if aligned < lower else aligned
            if upper and lower and lower + timedelta(seconds=30) > upper:
                continue
            candidate = query
            if lower:
                candidate = candidate.where(ModelScanSource.end_utc > lower)
            if upper:
                candidate = candidate.where(ModelScanSource.start_utc < window_start(upper, 30))
            first = self.session.scalar(candidate.order_by(ModelScanSource.start_utc, ModelScanSource.path).limit(1))
            if first is not None:
                start = max(window_start(first.start_utc, 30), lower) if lower else window_start(first.start_utc, 30)
                end = min(start + timedelta(seconds=300), window_start(upper, 30)) if upper else start + timedelta(seconds=300)
                break
        if start is None:
            return {"sources": [], "start_utc": None, "end_utc": None}
        rows = self.session.scalars(select(ModelScanSource).where(
            ModelScanSource.scan_id.in_(select(ModelScan.id).where(ModelScan.site_id == self._scan(unit.scan_id).site_id)),
            ModelScanSource.start_utc < end + timedelta(seconds=40),
            ModelScanSource.end_utc > start - timedelta(seconds=40))
            .order_by((ModelScanSource.scan_id == unit.scan_id).desc(), ModelScanSource.start_utc, ModelScanSource.path).limit(1001)).all()
        if len(rows) > 1000:
            raise WindowError("too_many_overlapping_sources")
        sources = {}
        for row in rows:
            sources.setdefault(row.path, json.loads(row.metadata_json))
        requested = [json.loads(r.metadata_json) for r in rows if r.scan_id == unit.scan_id and not r.context_only]
        return {"sources": list(sources.values()), "requested_sources": requested,
                "start_utc": start.isoformat(), "end_utc": end.isoformat()}

    def finish_plan(self, unit_id, worker_id, attempt_count, blocks):
        unit = self._owned(unit_id, worker_id, attempt_count)
        if unit.kind != "plan":
            raise WindowError("wrong_scan_phase")
        scope = self._planning_payload(unit)
        scan = self._scan(unit.scan_id)
        config = json.loads(scan.config_json)
        if not scope["sources"]:
            if blocks:
                raise WindowError("invalid_window_plan", 422)
            unit.status, unit.finished_at = "succeeded", utc_now()
        else:
            expected = set()
            for source in scope["requested_sources"]:
                start = max(window_start(utc(source["start_utc"]), 30), utc(scope["start_utc"]))
                while start < utc(source["end_utc"]) and start + timedelta(seconds=30) <= utc(scope["end_utc"]):
                    if not config.get("time_ranges") or any(utc(r["start_utc"]) <= start and
                            start + timedelta(seconds=30) <= utc(r["end_utc"]) for r in config["time_ranges"]):
                        expected.add(start)
                    start += timedelta(seconds=30)
            provided = [utc(p["start_utc"]) for b in blocks for p in b["windows"]]
            if set(provided) != expected or len(provided) != len(expected):
                raise WindowError("incomplete_window_plan", 422)
            for block in blocks:
                plans = block["windows"]
                if len(plans) > 10:
                    raise WindowError("invalid_window_plan", 422)
                selected = []
                for plan in plans:
                    start, end = utc(plan["start_utc"]), utc(plan["end_utc"])
                    if (start != window_start(start, 30) or end - start != timedelta(seconds=30)
                            or start < utc(scope["start_utc"]) or end > utc(scope["end_utc"])):
                        raise WindowError("invalid_window_plan", 422)
                    if config.get("time_ranges") and not any(utc(r["start_utc"]) <= start and
                            end <= utc(r["end_utc"]) for r in config["time_ranges"]):
                        continue
                    if self.session.scalar(select(ModelScanReceipt.id).where(
                        ModelScanReceipt.scan_id == scan.id, ModelScanReceipt.start_utc == start)):
                        continue
                    if any(s not in scope["sources"] for s in plan["sources"]):
                        raise WindowError("invalid_plan_source", 422)
                    selected.append(plan)
                if not selected:
                    continue
                infer = self._add_unit(scan.id, "infer", digest([p["start_utc"] for p in selected]), {"windows": selected})
                work = []
                for plan in selected:
                    start = utc(plan["start_utc"])
                    wid = window_id(scan.site_id, start, 30)
                    status = "pending"
                    if any(utc(e["start_utc"]) == start and e.get("window_seconds", 30) == 30
                           for e in config["exclusions"]):
                        status = "excluded"
                    window = self.session.get(EventWindow, wid)
                    if window is not None:
                        analysis = self.session.get(WindowAnalysis, window.latest_analysis_id) if window.latest_analysis_id else None
                        reviewed = self.session.scalar(select(WindowReview.id).where(WindowReview.window_id == wid))
                        if reviewed:
                            status = "protected"
                        elif analysis is not None and analysis.status == "cancelled":
                            status = "excluded"
                    geometry_excluded = bool(plan["sources"]) and all(
                        excluded_two_metre_source(s) for s in plan["sources"])
                    if status == "pending" and geometry_excluded:
                        status = "excluded"
                    if status == "pending" and plan["coverage"] != "complete":
                        status = "unjudgeable"
                    if status == "pending" and self.session.scalar(select(Job.id).join(SourceFile).where(
                            SourceFile.site_id == scan.site_id, SourceFile.path.in_([s["path"] for s in plan["sources"]]),
                            Job.status == "cancelled").limit(1)):
                        status = "excluded"
                    row = ModelScanReceipt(scan_id=scan.id, infer_unit_id=infer.id, start_utc=start,
                        inference_key=inference_key(scan.site_id, start, plan["sources"], config["model"]),
                        status=status, result_json="{}", result_sha256="", window_id=wid,
                        error_code=("excluded_2023_two_metre" if geometry_excluded and status == "excluded"
                                    else "coverage_incomplete" if status == "unjudgeable" else None),
                        finished_at=None if status == "pending" else utc_now())
                    if status == "pending":
                        previous = self.session.scalar(select(ModelScanReceipt).where(
                            ModelScanReceipt.inference_key == row.inference_key,
                            ModelScanReceipt.status.in_(("negative", "ready")), ModelScanReceipt.scan_id != scan.id)
                            .order_by(ModelScanReceipt.finished_at.desc()).limit(1))
                        if previous and (previous.status == "negative" or (window is not None and
                                previous.analysis_id == window.latest_analysis_id and
                                self.session.get(WindowAnalysis, previous.analysis_id).input_sha256 == previous.input_sha256)):
                            for attr in ("status", "score", "final_score", "model_input_sha256", "result_json", "result_sha256",
                                         "analysis_id", "input_sha256"):
                                setattr(row, attr, getattr(previous, attr))
                            row.reused_from, row.finished_at = previous.id, utc_now()
                    self.session.add(row)
                    if row.status == "pending":
                        work.append(plan)
                infer.payload_json = canonical({"windows": work})
                infer.window_count = len(work)
                if not work:
                    infer.status, infer.finished_at = "succeeded", utc_now()
            unit.payload_json = canonical({"cursor": scope["end_utc"]})
            unit.status = "queued"
        self.session.flush()
        self._refresh(scan.id)
        return {"status": unit.status}

    def _refresh(self, scan_id):
        scan = self._scan(scan_id)
        if scan.status in ("paused", "cancelled"):
            return
        active = self.session.scalar(select(literal(1)).select_from(ModelScanUnit).where(
            ModelScanUnit.scan_id == scan_id, ModelScanUnit.status.in_(("queued", "running"))).limit(1))
        if active is not None:
            return
        counts = dict(self.session.execute(select(ModelScanUnit.status, func.count()).where(
            ModelScanUnit.scan_id == scan_id).group_by(ModelScanUnit.status)).all())
        if counts.get("failed"):
            scan.status = "failed"
        else:
            errors = self.session.scalar(select(func.count()).select_from(ModelScanReceipt).where(
                ModelScanReceipt.scan_id == scan_id, ModelScanReceipt.status == "unjudgeable"))
            errors += sum(len(page.get("errors", [])) for pages in self.session.scalars(
                select(ModelScanUnit.pages_json).where(ModelScanUnit.scan_id == scan_id, ModelScanUnit.kind == "discover"))
                for page in json.loads(pages).values())
            scan.status = "completed_with_errors" if errors else "succeeded"
        scan.updated_at = utc_now()

    def get(self, scan_id):
        scan = self._scan(scan_id)
        config = json.loads(scan.config_json)
        counts = dict(self.session.execute(select(ModelScanReceipt.status, func.count()).where(
            ModelScanReceipt.scan_id == scan_id).group_by(ModelScanReceipt.status)).all())
        ready = counts.get("ready", 0)
        pending = counts.get("selected", 0) + counts.get("pending_artifacts", 0)
        counts = {key: counts.get(key, 0) for key in
                  ("negative", "unjudgeable", "protected", "excluded", "cancelled", "ready")} | {
            "selected": ready + pending, "pending_artifacts": pending,
            "judged": counts.get("negative", 0) + ready + pending,
            "sources": self.session.scalar(select(func.count()).select_from(ModelScanSource).where(
                ModelScanSource.scan_id == scan_id)),
            "planned": self.session.scalar(select(func.count()).select_from(ModelScanReceipt).where(
                ModelScanReceipt.scan_id == scan_id))}
        unfinished = self.session.scalar(select(func.count()).select_from(ModelScanUnit).where(
            ModelScanUnit.scan_id == scan_id, ModelScanUnit.kind == "discover",
            ModelScanUnit.status != "succeeded"))
        source_errors = [error for pages in self.session.scalars(
            select(ModelScanUnit.pages_json).where(ModelScanUnit.scan_id == scan_id, ModelScanUnit.kind == "discover"))
            for page in json.loads(pages).values() for error in page.get("errors", [])]
        counts["source_errors"] = len(source_errors)
        counts["reused"] = self.session.scalar(select(func.count()).select_from(ModelScanReceipt).where(
            ModelScanReceipt.scan_id == scan_id, ModelScanReceipt.reused_from.is_not(None)))
        failed_units = self.session.execute(select(ModelScanUnit.kind, ModelScanUnit.error_code,
            ModelScanUnit.error_message).where(ModelScanUnit.scan_id == scan_id,
            ModelScanUnit.status == "failed").order_by(ModelScanUnit.finished_at.desc()).limit(5)).all()
        failed_windows = self.session.execute(select(ModelScanReceipt.start_utc, ModelScanReceipt.error_code,
            ModelScanReceipt.error_message).where(ModelScanReceipt.scan_id == scan_id,
            ModelScanReceipt.status == "unjudgeable").order_by(ModelScanReceipt.start_utc).limit(5)).all()
        estimate = self._estimate(scan, counts, unfinished == 0)
        return {"id": scan.id, "site_id": scan.site_id, "status": scan.status, "model": config["model"],
                "directories": config["directories"], "time_ranges": config["time_ranges"], "counts": counts,
                "discovery_complete": unfinished == 0, **estimate,
                "retryable": bool(failed_units) or bool(self.session.scalar(select(ModelScanReceipt.id).where(
                    ModelScanReceipt.scan_id == scan_id, ModelScanReceipt.status == "unjudgeable",
                    ModelScanReceipt.error_code != "coverage_incomplete").limit(1))),
                "errors": ([{"phase": u.kind, "code": u.error_code, "message": u.error_message} for u in failed_units] +
                           [{"phase": "discover", "code": "source_unavailable", "message": f"{e['path']}: {e['error']}"}
                            for e in source_errors[:5]] + [{"phase": "infer", "code": r.error_code,
                            "message": f"{r.start_utc.isoformat()}: {r.error_message or r.error_code}"} for r in failed_windows]),
                "created_at": scan.created_at.isoformat(), "updated_at": scan.updated_at.isoformat()}

    def _estimate(self, scan, counts, discovery_complete):
        now = utc_now()
        planner_pages = self.session.scalar(select(ModelScanUnit.pages_json).where(
            ModelScanUnit.scan_id == scan.id, ModelScanUnit.kind == "plan"))
        total = json.loads(planner_pages).get("total") if planner_pages is not None and discovery_complete else None
        result = {"total": total, "remaining_seconds": None, "estimated_finish_at": None, "estimate_status": "estimating"}
        if scan.status in ("paused", "cancelled", "failed"):
            return result | {"estimate_status": scan.status}
        if not discovery_complete:
            return result | {"estimate_status": "discovering"}
        if scan.status in ("succeeded", "completed_with_errors"):
            return result | {"estimate_status": "complete", "remaining_seconds": 0}
        heartbeat = self.session.scalar(select(func.max(ModelScanUnit.heartbeat_at)).where(
            ModelScanUnit.scan_id == scan.id, ModelScanUnit.status == "running"))
        if heartbeat is None or heartbeat < now - timedelta(seconds=LEASE_SECONDS):
            return result | {"estimate_status": "waiting_worker"}
        remaining = max(0, (total or 0) - sum(counts[k] for k in
            ("judged", "unjudgeable", "protected", "excluded", "cancelled")))
        rates = {}
        for kind in ("infer", "materialize"):
            samples = self.session.execute(select(ModelScanUnit.started_at, ModelScanUnit.finished_at,
                ModelScanUnit.window_count).where(
                ModelScanUnit.scan_id == scan.id, ModelScanUnit.kind == kind,
                ModelScanUnit.status == "succeeded", ModelScanUnit.started_at.is_not(None),
                ModelScanUnit.finished_at >= now - timedelta(hours=2))
                .order_by(ModelScanUnit.finished_at.desc()).limit(30)).all()
            samples = [u for u in samples if u.finished_at > u.started_at and (kind != "infer" or u.window_count)]
            if len(samples) >= 4:
                rates[kind] = sum((u.finished_at - u.started_at).total_seconds() for u in samples) / sum(
                    u.window_count if kind == "infer" else 1 for u in samples)
        if total is None or (remaining and "infer" not in rates):
            return result
        # Future positive windows also require artifacts; estimate using this batch's observed selection fraction.
        expected_artifacts = counts["pending_artifacts"] + (remaining * counts["selected"] / counts["judged"] if counts["judged"] else 0)
        if expected_artifacts and "materialize" not in rates:
            return result
        seconds = math.ceil(remaining * rates.get("infer", 0) + expected_artifacts * rates.get("materialize", 0))
        return result | {"estimate_status": "running", "remaining_seconds": seconds,
                         "estimated_finish_at": (now + timedelta(seconds=seconds)).isoformat()}
