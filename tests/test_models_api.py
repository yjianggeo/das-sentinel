from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from das_sentinel.api import create_app
from das_sentinel.config import ProcessingProfile, Settings
from das_sentinel.persistence import (
    Repositories,
    create_local_engine,
    session_factory,
    upgrade_database,
)
from das_sentinel.services.training import LABEL_GATES
from fastapi.testclient import TestClient

PASSWORD = "correct horse"
WORKER_TOKEN = "worker-token-" + "x" * 32
LABELS = ("earthquake", "blast", "traffic", "noise")


def _settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "raw"
    app_data = tmp_path / "app-data"
    raw.mkdir()
    app_data.mkdir()
    return Settings(
        data_root=raw,
        app_data_dir=app_data,
        database_url=f"sqlite+pysqlite:///{app_data / 'das.sqlite3'}",
        api_url="http://testserver",
        session_secret="s" * 32,
        worker_token=WORKER_TOKEN,
        secure_cookies=False,
        pipeline_version="v2",
    )


def _seed_reviewed_events(repositories, site_id: str, run_id: str, per_label: int = 25):
    base = datetime(2026, 7, 1, tzinfo=UTC)
    for label_index, label in enumerate(LABELS):
        for index in range(per_label):
            event = repositories.events.add(
                run_id=run_id,
                event_uid=f"{label}-{index}",
                start_sample=0,
                end_sample=10,
                channel_start=0,
                channel_stop=2,
                occurred_at=base + timedelta(days=index % 5, minutes=index),
                features={
                    "duration_s": label_index * 100.0 + index,
                    "rms": label_index + index / 100.0,
                },
            )
            repositories.reviews.add(event.id, label=label, admin_id=1)


@pytest.fixture
def model_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    upgrade_database(settings.database_url)
    engine = create_local_engine(settings.database_url)
    sessions = session_factory(engine)
    with sessions.begin() as session:
        repositories = Repositories(session)
        repositories.admins.add(password_hash=PasswordHasher().hash(PASSWORD))
        site = repositories.sites.add(
            name="site-a",
            data_path=str(settings.data_root),
            timezone="UTC",
            cable_type="borehole",
        )
        profile = repositories.profiles.add(
            site_id=site.id,
            version=1,
            payload=ProcessingProfile(device="cpu").model_dump(mode="json"),
            is_current=True,
        )
        source = repositories.source_files.add(
            site_id=site.id,
            path=str((settings.data_root / "a.h5").resolve()),
            fingerprint="a" * 64,
            size_bytes=1,
            mtime_ns=1,
        )
        run = repositories.runs.add(
            source_file_id=source.id,
            processing_profile_id=profile.id,
            pipeline_version="pipeline-v1",
            requested_backend="cpu",
            actual_backend="cpu",
            status="succeeded",
        )
        _seed_reviewed_events(repositories, site.id, run.id)
        site_id = site.id
    monkeypatch.setattr("das_sentinel.api.app.check_database_startup", lambda _: None)
    with TestClient(create_app(settings)) as client:
        login = client.post("/api/v1/session", json={"password": PASSWORD})
        yield client, sessions, login.headers["x-csrf-token"], site_id
    engine.dispose()


def _train(client, token, site_id):
    response = client.post(
        f"/api/v1/sites/{site_id}/models",
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_model_routes_require_admin_and_csrf(model_api):
    client, _, _, site_id = model_api
    client.cookies.clear()
    assert client.get(f"/api/v1/sites/{site_id}/models").status_code == 401
    assert client.post(f"/api/v1/sites/{site_id}/models").status_code == 401
    login = client.post("/api/v1/session", json={"password": PASSWORD})
    assert login.status_code == 204
    assert client.post(f"/api/v1/sites/{site_id}/models").status_code == 403


def test_train_list_publish_and_rollback_roundtrip(model_api):
    client, _, token, site_id = model_api

    trained = _train(client, token, site_id)
    assert trained["status"] == "candidate"
    assert trained["metrics"]["reviewed_count_by_label"] == {
        label: 25 for label in LABELS
    }
    assert trained["metrics"]["macro_f1"] >= LABEL_GATES["macro_f1"]
    assert trained["dataset_sha256"] is not None

    listed = client.get(f"/api/v1/sites/{site_id}/models").json()
    assert [item["id"] for item in listed] == [trained["id"]]

    published = client.post(
        f"/api/v1/models/{trained['id']}/publish",
        headers={"X-CSRF-Token": token},
    )
    assert published.status_code == 200, published.text
    assert published.json()["status"] == "active"

    second = _train(client, token, site_id)
    assert client.post(
        f"/api/v1/models/{second['id']}/publish",
        headers={"X-CSRF-Token": token},
    ).status_code == 200

    rolled_back = client.post(
        f"/api/v1/sites/{site_id}/models/rollback",
        headers={"X-CSRF-Token": token},
    )
    assert rolled_back.status_code == 200, rolled_back.text
    assert rolled_back.json()["id"] == trained["id"]
    assert rolled_back.json()["status"] == "active"

    detail = client.get(f"/api/v1/models/{second['id']}").json()
    assert detail["status"] == "retired"


def test_rollback_without_previous_returns_409(model_api):
    client, _, token, site_id = model_api
    trained = _train(client, token, site_id)
    assert client.post(
        f"/api/v1/models/{trained['id']}/publish",
        headers={"X-CSRF-Token": token},
    ).status_code == 200
    assert (
        client.post(
            f"/api/v1/sites/{site_id}/models/rollback",
            headers={"X-CSRF-Token": token},
        ).status_code
        == 409
    )


def test_model_below_gate_cannot_publish(model_api):
    client, sessions, token, site_id = model_api
    trained = _train(client, token, site_id)
    with sessions.begin() as session:
        stored = Repositories(session).models.get(trained["id"])
        metrics = dict(json.loads(stored.metrics_json))
        metrics["macro_f1"] = 0.40
        stored.metrics_json = json.dumps(metrics)

    response = client.post(
        f"/api/v1/models/{trained['id']}/publish",
        headers={"X-CSRF-Token": token},
    )
    assert response.status_code == 409


def test_unknown_site_or_model_returns_404(model_api):
    client, _, token, _ = model_api
    assert (
        client.post(
            "/api/v1/sites/does-not-exist/models",
            headers={"X-CSRF-Token": token},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/models/does-not-exist/publish",
            headers={"X-CSRF-Token": token},
        ).status_code
        == 404
    )
    assert client.get("/api/v1/models/does-not-exist").status_code == 404
