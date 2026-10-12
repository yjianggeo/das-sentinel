"""Model-first worker: durable HTTP work, one NAS reader and one resident model."""

import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import numpy as np

from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.model_scan_contract import digest, utc
from das_sentinel.model_scan_input import arrays_sha256, read_scan_input
from das_sentinel.services.discovery import _open_anchored_source
from das_sentinel.services.model_scan_discovery import (
    discover_context,
    discover_directory,
    plan_source_windows,
)
from das_sentinel.window_inference import (
    checked_metadata,
    deployment_folder,
    file_sha,
    normalize_prediction_row,
    save_json,
)
from das_sentinel.worker.main import (
    ResultSubmissionError,
    WorkerClaimLostError,
    WorkerFatalProtocolError,
    WorkerOutcome,
    WorkerTransientError,
)

logger = logging.getLogger(__name__)


class HttpxModelScanClient:
    def __init__(self, api_url, worker_token, *, transport=None):
        local = urlsplit(api_url).hostname in {"localhost", "127.0.0.1", "::1"}
        self.client = httpx.Client(base_url=api_url.rstrip("/"), transport=transport,
                                  headers={"X-Worker-Token": worker_token}, timeout=180,
                                  trust_env=not local)

    def close(self):
        self.client.close()

    def _post(self, suffix, payload):
        try:
            response = self.client.post(f"/api/v1/internal/model-scans/{suffix}", json=payload)
        except httpx.TransportError as error:
            raise WorkerTransientError("model scan API transport failure") from error
        if response.status_code >= 500 or response.status_code == 429:
            raise WorkerTransientError(f"model scan API HTTP {response.status_code}")
        if response.status_code in (404, 409):
            detail = response.json().get("detail", "")
            if detail in ("input_mismatch", "source_profile_changed", "result_conflict"):
                raise ResultSubmissionError(str(detail))
            raise WorkerClaimLostError("model scan claim lost")
        if response.status_code == 422:
            raise ResultSubmissionError(f"model scan result rejected: {response.text[:500]}")
        if response.status_code >= 400:
            raise WorkerFatalProtocolError(f"model scan API HTTP {response.status_code}")
        return None if response.status_code == 204 else response.json()

    def claim(self, worker_id):
        return self._post("claim", {"worker_id": worker_id})

    def heartbeat(self, unit_id, worker_id, attempt_count):
        return self.submit(unit_id, "heartbeat", worker_id, attempt_count)

    def submit(self, unit_id, action, worker_id, attempt_count, **payload):
        body = {"worker_id": worker_id, "attempt_count": attempt_count} | payload
        try:
            return self._post(f"{unit_id}/{action}", body)
        except WorkerTransientError:
            # Replay only operations with a fixed request identity/result payload.
            if action not in ("prediction", "discovery-page", "discovery-done", "materialized"):
                raise
            return self._post(f"{unit_id}/{action}", body)


class InferenceProcess:
    def __init__(self, folder, snapshot, spool, python):
        self.spool = Path(spool).resolve()
        self.spool.mkdir(parents=True, exist_ok=True)
        self.snapshot = snapshot
        self.request_root = self.spool / uuid4().hex
        self.request_root.mkdir()
        snapshot_path = self.request_root / "model.json"
        save_json(snapshot_path, snapshot)
        self.errors = (self.request_root / "inference.log").open("a", encoding="utf8")
        self.process = subprocess.Popen([python, "-m", "das_sentinel.model_scan_inference", "serve",
            "--model-folder", str(folder), "--snapshot", str(snapshot_path), "--spool-root", str(self.request_root)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors, text=True, encoding="utf8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), bufsize=1)
        self.responses = queue.Queue(maxsize=2)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def predict(self, arrays):
        request_id = uuid4().hex
        path = self.request_root / f"{request_id}.npz"
        try:
            with path.open("wb") as stream:
                np.savez(stream, **arrays)
            request = {"request_id": request_id, "path": str(path), "sha256": file_sha(path)}
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.responses.get(timeout=120)
            if line is None:
                raise RuntimeError("inference process exited")
            response = json.loads(line)
            if response.get("request_id") != request_id:
                raise RuntimeError("inference response ID mismatch")
            if response.get("status") != "succeeded":
                if response.get("error_kind") == "runtime":
                    raise RuntimeError(response.get("error_message", "inference failed"))
                raise ValueError(response.get("error_message", "inference failed"))
            return response
        except (queue.Empty, BrokenPipeError) as error:
            raise RuntimeError("inference process unavailable or timed out") from error
        finally:
            # This exact file was created above under the private inference spool.
            path.unlink(missing_ok=True)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    # venv python.exe can be a launcher with the interpreter as its child.
                    subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=10, check=False)
                else:
                    self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        self.process.stdout.close()
        self.errors.close()


class ModelScanWorker:
    def __init__(self, settings, client, worker_id, *, predictor=None, heartbeat_seconds=15, read_mode="single"):
        self.settings, self.client, self.worker_id = settings, client, worker_id
        self.injected_predictor = predictor
        self.predictor = predictor
        self.model_key = None
        self.heartbeat_seconds = heartbeat_seconds
        if read_mode not in ("single", "block"):
            raise ValueError("unsupported read mode")
        self.read_mode = read_mode

    def close(self):
        if self.predictor is not None and self.injected_predictor is None:
            self.predictor.close()
        self.predictor = None

    def _model(self, job):
        model = job["config"]["model"]
        folder = deployment_folder(self.settings.app_data_dir / "window-models", job["site_id"], model["id"])
        metadata = checked_metadata(folder, model)
        if self.injected_predictor is None and self.model_key != digest(model):
            self.close()
            python = os.environ.get("DAS_INFERENCE_PYTHON") or os.environ.get("DAS_TRAINING_PYTHON") or sys.executable
            self.predictor = InferenceProcess(folder, model, self.settings.app_data_dir / "model-scan-spool", python)
            self.model_key = digest(model)
        return metadata

    def run_once(self):
        job = self.client.claim(self.worker_id)
        if job is None:
            return WorkerOutcome(status="idle")
        try:
            UUID(job["id"])
            root = Path(job["config"]["root"])
            if not root.resolve().is_relative_to(self.settings.data_root.resolve()):
                raise ValueError("root outside worker data root")
        except (ValueError, KeyError, TypeError) as error:
            raise WorkerFatalProtocolError("invalid model scan claim") from error
        unit_id, attempt = job["id"], job["attempt_count"]
        if job["status"] != "running" or job["payload"].get("protected"):
            return WorkerOutcome(status=job["status"], job_id=unit_id)
        stop, heartbeat_errors = threading.Event(), []
        def heartbeat():
            while not stop.wait(self.heartbeat_seconds):
                try:
                    self.client.heartbeat(unit_id, self.worker_id, attempt)
                except Exception as error:  # noqa: BLE001 - Transfer thread errors to submit().
                    heartbeat_errors.append(error)
                    return
        def submit(action, **payload):
            if heartbeat_errors:
                raise heartbeat_errors[0]
            return self.client.submit(unit_id, action, self.worker_id, attempt, **payload)
        thread = threading.Thread(target=heartbeat, daemon=True)
        try:
            self.client.heartbeat(unit_id, self.worker_id, attempt)
            thread.start()
            cache = self.settings.app_data_dir / "model-scan-cache"
            if job["kind"] == "discover":
                from das_sentinel.whole_window import ALGORITHM_VERSION
                metadata_cache = self.settings.app_data_dir / f"window-source-metadata-{ALGORITHM_VERSION}"
                signatures = []
                pages = (discover_context(root, job["payload"]["context_paths"], cache_root=metadata_cache)
                         if "context_paths" in job["payload"] else
                         discover_directory(root, Path(job["payload"]["directory"]), cache_root=metadata_cache))
                for page_index, payload in enumerate(pages):
                    submit("discovery-page", page_index=page_index, payload=payload)
                    signatures.append(digest(payload))
                submit("discovery-done", page_count=len(signatures), manifest_sha256=digest(signatures))
            elif job["kind"] == "plan":
                scope = job["payload"]
                blocks = plan_source_windows(scope["sources"], cache_root=cache,
                    start_utc=scope["start_utc"], end_utc=scope["end_utc"],
                    requested_sources=scope.get("requested_sources")) if scope["sources"] else []
                submit("plan-done", blocks=blocks)
            elif job["kind"] == "infer":
                metadata = self._model(job)
                from das_sentinel.readers.continuous_low_frequency import CoreCache
                core_cache = CoreCache() if self.read_mode == "block" else None
                for plan in job["payload"]["windows"]:
                    try:
                        with ExitStack() as stack:
                            for source in plan["sources"]:
                                stack.enter_context(_open_anchored_source(root, Path(source["path"])))
                            arrays, provenance = read_scan_input(plan, metadata, cache, core_cache=core_cache)
                            prediction = self.predictor.predict(arrays)
                        result = {"start_utc": plan["start_utc"], "status": "succeeded",
                                  "model_id": prediction["model_id"], "score": prediction["score"],
                                  "model_input_sha256": provenance["model_input_sha256"]}
                        del arrays
                    except (OSError, ValueError, RuntimeError) as error:
                        result = {"start_utc": plan["start_utc"], "status": "failed",
                                  "error_code": "input_or_inference_failed", "error_message": str(error)[:2000]}
                        if isinstance(error, RuntimeError) and self.injected_predictor is None:
                            self.close()
                            self.model_key = None
                            metadata = self._model(job)
                    submit("prediction", result=result)
            elif job["kind"] == "materialize":
                metadata = self._model(job)
                binding = self._materialize(job, metadata, root, cache)
                submit("materialized", binding=binding)
            else:
                raise WorkerFatalProtocolError("unsupported model scan phase")
            return WorkerOutcome(status="succeeded", job_id=unit_id)
        except WorkerClaimLostError:
            return WorkerOutcome(status="claim_lost", job_id=unit_id)
        except (WorkerTransientError, WorkerFatalProtocolError):
            raise
        except Exception as error:
            logger.exception("Model scan unit failed: %s", unit_id)
            try:
                submit("fail", error_code="model_scan_failed", error_message=str(error)[:2000])
            except WorkerClaimLostError:
                return WorkerOutcome(status="claim_lost", job_id=unit_id)
            return WorkerOutcome(status="failed", job_id=unit_id)
        finally:
            stop.set()
            if thread.ident is not None:
                thread.join()

    def _materialize(self, job, metadata, root, cache):
        from das_sentinel.whole_window import analyze_window
        analysis, plan = job["analysis"], job["payload"]["plan"]
        output = self.settings.app_data_dir / "window-artifacts" / analysis["id"] / f"attempt-{analysis['attempt_count']}"
        if analysis["reuse"]:
            result = json.loads((output / "result.json").read_text(encoding="utf8"))
        else:
            with ExitStack() as stack:
                paths = [Path(s["path"]) for s in plan["sources"]]
                for path in paths:
                    stack.enter_context(_open_anchored_source(root, path))
                result = analyze_window(paths, utc(plan["start_utc"]),
                    ProcessingProfile.model_validate(job["config"]["profile_payload"]), output,
                    site_timezone=job["config"]["site_timezone"], window_seconds=30, skip_empty=False)
        waveform = next(a for a in result["artifacts"] if a["kind"] == "waveform")
        row = normalize_prediction_row({"window_id": analysis["window_id"], "window_seconds": 30,
            "input_sha256": result["input_sha256"], "sources": result["read_sources"], "unit": result["unit"],
            "grid_sha256": result["grid_sha256"], "waveform": str(output / waveform["path"]),
            "waveform_sha256": waveform["sha256"]}, metadata,
            self.settings.app_data_dir / "msc" / job["config"]["model"]["id"])
        with np.load(row["waveform"], allow_pickle=False) as source_arrays:
            arrays = {key: source_arrays[key] for key in source_arrays.files}
            signature = arrays_sha256(arrays, plan["start_utc"], result["unit"])
            if signature != job["payload"]["model_input_sha256"]:
                raise ValueError("input_mismatch")
            prediction = self.predictor.predict(arrays)
        return {"analysis_id": analysis["id"], "input_sha256": result["input_sha256"],
                "model_input_sha256": signature, "score": prediction["score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", default="model-scan-1")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--read-mode", choices=("single", "block"), default="single")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    client = HttpxModelScanClient(settings.api_url, settings.worker_token)
    worker = ModelScanWorker(settings, client, args.worker_id, read_mode=args.read_mode)
    try:
        while True:
            try:
                outcome = worker.run_once()
                logger.info("model_scan %s %s", outcome.status, outcome.job_id)
            except WorkerTransientError:
                logger.exception("Temporary API failure; uncommitted work remains resumable")
                time.sleep(10)
                continue
            if args.once:
                break
            if outcome.status == "idle":
                time.sleep(5)
    finally:
        worker.close()
        client.close()


if __name__ == "__main__":
    main()
