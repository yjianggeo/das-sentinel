from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from das_sentinel import pipeline
from das_sentinel.api.routes.internal import claim_response
from das_sentinel.api.schemas import WorkerResultRequest
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.domain import Candidate
from das_sentinel.features import FEATURE_NAMES
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.persistence.models import Classification, Event, ProcessingRun
from das_sentinel.screening import ScreeningResult
from das_sentinel.services.discovery import fingerprint
from das_sentinel.services.jobs import JobService
from das_sentinel.services.results import ResultService
from das_sentinel.services.training import TrainingService
from das_sentinel.worker.main import Worker, WorkerClaim
from sqlalchemy import select

LABELS = ("earthquake", "blast", "traffic", "noise")
CONTEXT_FEATURES = {"site_id", "cable_type", "device_config"}


def _training_features(site_id: str, label_index: int, index: int):
    values = {
        name: 0.0
        for name in FEATURE_NAMES
        if name not in CONTEXT_FEATURES
    }
    values.update(
        {
            "duration_s": label_index * 100.0 + index,
            "rms": label_index + index / 100.0,
            "site_id": site_id,
            "cable_type": "urban",
            "device_config": "cpu",
        }
    )
    return values


def _write_inference_hdf5(path: Path) -> None:
    sample_rate_hz = 200.0
    sample_count = 400
    start = datetime(2026, 8, 1, tzinfo=UTC)
    rng = np.random.default_rng(20260806)
    raw_data = rng.normal(0.0, 0.05, (sample_count, 8)).astype(np.float32)
    raw_data[100:160, :4] += (
        np.hanning(60).astype(np.float32)[:, None] * 4.0
    )
    times = (
        start.timestamp() * 1_000_000
        + np.arange(sample_count) * 1_000_000 / sample_rate_hz
    )
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition").create_group("Raw[0]")
        raw.create_dataset("RawData", data=raw_data)
        raw.create_dataset("RawDataTime", data=times.astype(np.int64))
        raw.attrs["OutputDataRate"] = sample_rate_hz
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 2
        raw.attrs["RawDataUnit"] = "strain_rate"


def _fixed_model_event(block, profile, backend):
    del profile, backend
    if not (
        block.sample_offset
        <= 130
        < block.sample_offset + block.data.shape[0]
    ):
        return ScreeningResult(candidates=(), events=(), diagnostics={})
    return ScreeningResult(
        candidates=(
            Candidate(
                detector="stalta:short",
                start_sample=100,
                end_sample=160,
                peak_sample=130,
                channel_start=0,
                channel_stop=4,
                score=0.95,
                metrics={"sample_rate_hz": block.metadata.sample_rate_hz},
            ),
        ),
        events=(),
        diagnostics={},
    )


def _seed_published_model(sessions, artifact_root: Path, inference_path: Path):
    training_path = inference_path.parent / "training.h5"
    training_path.write_bytes(b"training-source")
    with sessions.begin() as session:
        repositories = Repositories(session)
        admin = repositories.admins.add(password_hash="not-used")
        site = repositories.sites.add(
            name="urban-utc",
            data_path=str(inference_path.parent.resolve()),
            timezone="UTC",
            cable_type="urban",
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        training_source = repositories.source_files.add(
            site_id=site.id,
            path=str(training_path.resolve()),
            fingerprint=fingerprint(training_path),
            size_bytes=training_path.stat().st_size,
            mtime_ns=training_path.stat().st_mtime_ns,
        )
        training_run = repositories.runs.add(
            source_file_id=training_source.id,
            processing_profile_id=profile.id,
            pipeline_version="training-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        base = datetime(2026, 7, 1, tzinfo=UTC)
        for label_index, label in enumerate(LABELS):
            for index in range(25):
                event = repositories.events.add(
                    run_id=training_run.id,
                    event_uid=f"training-{label}-{index}",
                    start_sample=index * 10,
                    end_sample=index * 10 + 5,
                    channel_start=0,
                    channel_stop=4,
                    occurred_at=base
                    + timedelta(days=index % 5, minutes=index),
                    features=_training_features(site.id, label_index, index),
                )
                repositories.reviews.add(
                    event.id,
                    label=label,
                    note="seeded",
                    admin_id=admin.id,
                )

        training = TrainingService(session, artifact_root)
        candidate = training.train(site_id=site.id)
        published = training.publish(candidate.id)
        inference_source = repositories.source_files.add(
            site_id=site.id,
            path=str(inference_path.resolve()),
            fingerprint=fingerprint(inference_path),
            size_bytes=inference_path.stat().st_size,
            mtime_ns=inference_path.stat().st_mtime_ns,
        )
        job = JobService(session, artifact_root).enqueue(
            inference_source.id,
            profile.id,
            "v2",
        )
        return published, job.id


class CapturingProtocol:
    def __init__(self, claim: WorkerClaim) -> None:
        self.pending_claim: WorkerClaim | None = claim
        self.job_id = claim.id
        self.submissions: list[dict[str, Any]] = []
        self.failures: list[dict[str, Any]] = []
        self.heartbeats: list[tuple[str, str, int, int]] = []

    def claim(self, worker_id: str) -> WorkerClaim | None:
        assert worker_id == "worker-a"
        claim, self.pending_claim = self.pending_claim, None
        return claim

    def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        numerator: int,
        denominator: int,
    ) -> None:
        self.heartbeats.append(
            (job_id, worker_id, numerator, denominator)
        )

    def submit_result(self, job_id: str, payload: dict[str, Any]) -> None:
        assert job_id == self.job_id
        self.submissions.append(payload)

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error_code: str,
        error_message: str,
        retry_eligible: bool,
    ) -> None:
        self.failures.append(
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "error_code": error_code,
                "error_message": error_message,
                "retry_eligible": retry_eligible,
            }
        )


@pytest.fixture
def model_flow_env(tmp_path: Path):
    data_root = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    data_root.mkdir()
    app_data.mkdir()
    database_url = f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}"
    upgrade_database(database_url)
    engine = create_local_engine(database_url)
    sessions = session_factory(engine)
    settings = Settings(
        data_root=data_root,
        app_data_dir=app_data,
        database_url=database_url,
        api_url="http://api.invalid",
        session_secret="s" * 32,
        worker_token="worker-token-" + "x" * 32,
        worker_device="cpu",
        secure_cookies=False,
    )
    try:
        yield sessions, settings, app_data / "artifacts"
    finally:
        engine.dispose()


def test_published_model_executes_fuses_and_persists_one_identity(
    model_flow_env,
    monkeypatch: pytest.MonkeyPatch,
):
    sessions, settings, artifact_root = model_flow_env
    inference_path = settings.data_root / "inference.h5"
    _write_inference_hdf5(inference_path)
    published, job_id = _seed_published_model(
        sessions,
        artifact_root,
        inference_path,
    )
    with sessions.begin() as session:
        claim_snapshot = JobService(session, artifact_root).claim("worker-a")
        assert claim_snapshot is not None
        claim = WorkerClaim.from_dict(
            claim_response(claim_snapshot).model_dump(mode="json")
        )

    monkeypatch.setattr(pipeline, "screen_block", _fixed_model_event)
    protocol = CapturingProtocol(claim)
    outcome = Worker(
        settings=settings,
        client=protocol,
        worker_id="worker-a",
    ).run_once()

    assert outcome.status == "succeeded"
    assert protocol.failures == []
    assert len(protocol.submissions) == 1
    payload = WorkerResultRequest.model_validate(protocol.submissions[0])
    with sessions.begin() as session:
        ResultService(session, artifact_root).persist(job_id, payload)

    with sessions() as session:
        persisted_run = session.scalar(
            select(ProcessingRun).where(ProcessingRun.job_id == job_id)
        )
        assert persisted_run is not None
        persisted_event = session.scalar(
            select(Event).where(
                Event.processing_run_id == persisted_run.id
            )
        )
        assert persisted_event is not None
        persisted_classification = session.scalar(
            select(Classification).where(
                Classification.event_id == persisted_event.id
            )
        )
        assert persisted_classification is not None

    evidence_path = next(
        Path(artifact.path)
        for artifact in payload.artifacts
        if artifact.kind == "evidence"
    )
    evidence_json = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert claim.model_version_id == published.id
    assert persisted_run.model_version_id == published.id
    assert persisted_classification.model_version_id == published.id
    assert persisted_classification.method == "fusion"
    assert evidence_json["model_version"] == published.version
    assert set(json.loads(persisted_event.features_json)) == set(FEATURE_NAMES)
    assert evidence_json["measurements"]["schema_version"] == "2"
    assert evidence_json["source_fingerprint"] == claim.source_fingerprint
    assert evidence_json["model_output"]["model_version_id"] == published.id
    assert set(evidence_json["model_output"]["probabilities"]) == {
        "earthquake",
        "blast",
        "traffic",
        "noise",
    }
