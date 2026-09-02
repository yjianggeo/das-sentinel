"""Seed and serve a local DAS Sentinel stack for the Playwright e2e suite.

Creates an isolated app-data/raw pair, writes two synthetic HDF5 files containing
strong multi-channel STA/LTA events, seeds admin + site + processing profile,
runs migrations, serves the API on 127.0.0.1:8000, scans (real 10 s stability
window), drains the queue with an in-process worker, then keeps serving until
Ctrl+C.

Usage:
    python e2e/seed_stack.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.worker.main import HttpxWorkerClient, Worker

PASSWORD = "test-admin-password"
WORKER_TOKEN = "e2e-worker-token-" + "x" * 32
BASE = "http://127.0.0.1:8000"
STACK_DIR = Path(
    os.environ.get(
        "E2E_STACK_DIR",
        str(Path(tempfile.gettempdir()) / "das-sentinel-e2e"),
    )
).resolve()


def write_event_file(path: Path, start_time_utc: datetime) -> None:
    sample_rate_hz = 200.0
    sample_count = round(20.0 * sample_rate_hz)
    start = start_time_utc.astimezone(UTC)
    rng = np.random.default_rng(20260730)
    raw_data = rng.normal(0.0, 0.02, (sample_count, 8)).astype(np.float32)
    event_start = round(15.0 * sample_rate_hz)
    event_samples = round(0.3 * sample_rate_hz)
    burst = (
        np.hanning(event_samples)
        * np.sin(2 * np.pi * 10.0 * np.arange(event_samples) / sample_rate_hz)
        * 20.0
    ).astype(np.float32)
    raw_data[event_start : event_start + event_samples, :4] += burst[:, None]
    times = start.timestamp() * 1_000_000 + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=raw_data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


def main() -> None:
    if STACK_DIR.exists():
        shutil.rmtree(STACK_DIR)
    raw = (STACK_DIR / "raw").resolve()
    app_data = (STACK_DIR / "app-data").resolve()
    raw.mkdir(parents=True)
    app_data.mkdir()
    write_event_file(
        raw / "synthetic-earthquake-00.h5",
        datetime(2026, 7, 26, tzinfo=UTC),
    )
    write_event_file(
        raw / "synthetic-earthquake-01.h5",
        datetime(2026, 7, 26, 1, tzinfo=UTC),
    )

    settings = Settings(
        data_root=raw,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url=BASE,
        session_secret="e2e-" + "s" * 32,
        worker_token=WORKER_TOKEN,
        secure_cookies=False,
        pipeline_version="e2e-v1",
    )
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site = repositories.sites.add(
            name="e2e-site",
            data_path=str(raw),
            timezone="UTC",
            cable_type="downhole",
        )
        profile = ProcessingProfile(
            device="cpu",
            stalta_freq_low_hz=2.0,
            stalta_freq_high_hz=40.0,
            coherence_threshold=0.5,
            energy_ratio_threshold=101.0,
            trajectory_threshold=0.8,
        )
        repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=profile.model_dump(mode="json"),
            is_current=True,
        )
        site_id = site.id

    app = create_app(settings)
    import threading

    import httpx

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)

    client = httpx.Client(base_url=BASE, timeout=30.0)
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    csrf = login.headers["x-csrf-token"]
    first = client.post(f"/api/v1/sites/{site_id}/scan", json={}, headers={"X-CSRF-Token": csrf})
    assert first.status_code == 201, first.text
    print("waiting out the 10 s file-stability window ...", flush=True)
    time.sleep(11.0)
    enqueued = client.post(f"/api/v1/sites/{site_id}/scan", json={}, headers={"X-CSRF-Token": csrf})
    assert enqueued.status_code == 201, enqueued.text
    print(f"enqueued {len(enqueued.json())} job(s)", flush=True)
    jobs_listed = client.get("/api/v1/jobs").json()
    print(f"jobs via API: {len(jobs_listed)}", flush=True)
    assert len(jobs_listed) == 2, jobs_listed

    worker = Worker(
        settings=settings,
        client=HttpxWorkerClient(BASE, WORKER_TOKEN),
        worker_id="e2e-worker",
        heartbeat_interval_seconds=1.0,
    )
    for _ in range(len(jobs_listed) + 1):
        outcome = worker.run_once()
        print(f"worker: {outcome.status}", flush=True)
        if outcome.status == "idle":
            break

    events = client.get("/api/v1/events").json()
    print(f"events visible: {len(events)}", flush=True)
    if len(events) < 2:
        raise RuntimeError("seed produced fewer than two events; aborting e2e setup")
    print(f"serving seeded stack at {BASE} — press Ctrl+C to stop", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
