"""Bounded, durable background whole-window training jobs."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from das_sentinel.persistence.models import Site
from das_sentinel.window_training import export_rows


class WindowTrainingError(ValueError):
    def __init__(self, message, status_code=409):
        super().__init__(message)
        self.status_code = status_code


class WindowTrainingService:
    def __init__(self, session_factory, app_data_dir):
        self.session_factory = session_factory
        self.root = Path(app_data_dir) / "window-training"
        self.artifacts = Path(app_data_dir) / "window-artifacts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="window-training"
        )
        self.active = None
        self.process = None
        self.closed = False
        for path in self.root.glob("*/job.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if job["status"] in ("queued", "running"):
                job.update(status="failed", error="服务重启中断训练，请重新创建任务。")
                self._save(path, job)

    @staticmethod
    def _save(path, value):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)

    def manifest(self, site_id):
        with self.session_factory() as session:
            if session.get(Site, site_id) is None:
                raise WindowTrainingError("站点不存在", 404)
            return export_rows(session, self.artifacts, site_id,
                               normalization_root=self.root.parent / "window-training-grids" / site_id)

    def readiness(self, site_id):
        return self.manifest(site_id)["readiness"]

    def jobs(self, site_id):
        with self.session_factory() as session:
            if session.get(Site, site_id) is None:
                raise WindowTrainingError("站点不存在", 404)
        jobs = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in self.root.glob("*/job.json")
        ]
        return sorted(
            [j for j in jobs if j["site_id"] == site_id],
            key=lambda j: j["created_at"],
            reverse=True,
        )[:20]

    def create(self, site_id):
        with self.lock:
            if self.closed:
                raise WindowTrainingError("训练服务正在关闭。", 503)
            if self.active is not None:
                raise WindowTrainingError("已有整窗训练任务正在运行。")
            manifest = self.manifest(site_id)
            if not manifest["readiness"]["ready"]:
                raise WindowTrainingError("; ".join(manifest["readiness"]["reasons"]))
            job = {
                "id": str(uuid4()),
                "site_id": site_id,
                "status": "queued",
                "created_at": datetime.now(UTC).isoformat(),
                "metrics": None,
                "error": None,
            }
            folder = self.root / job["id"]
            folder.mkdir()
            self._save(folder / "manifest.json", manifest)
            self._save(folder / "job.json", job)
            self.active = job["id"]
            self.executor.submit(self._run, folder, job.copy())
            return job

    def _run(self, folder, job):
        try:
            job["status"] = "running"
            self._save(folder / "job.json", job)
            with (folder / "training.log").open("w", encoding="utf-8") as log:
                with self.lock:
                    if self.closed:
                        raise RuntimeError("训练服务已关闭。")
                    self.process = subprocess.Popen(
                        [
                            os.environ.get("DAS_TRAINING_PYTHON", sys.executable),
                            "-m",
                            "das_sentinel.window_training",
                            "train",
                            "--manifest",
                            str(folder / "manifest.json"),
                            "--output",
                            str(folder / "model"),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                try:
                    returncode = self.process.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                    raise RuntimeError("整窗训练超过两小时限制。")
            if returncode:
                raise RuntimeError("训练失败；请检查隔离环境 Torch 和 training.log。")
            job.update(
                status="succeeded",
                metrics=json.loads(
                    (folder / "model" / "metrics.json").read_text(encoding="utf-8")
                ),
            )
        except Exception as error:  # noqa: BLE001 - persist failure at the background job boundary
            job.update(status="failed", error=str(error))
        finally:
            self._save(folder / "job.json", job)
            with self.lock:
                self.active = None
                self.process = None

    def close(self):
        with self.lock:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.executor.shutdown(wait=False, cancel_futures=True)
