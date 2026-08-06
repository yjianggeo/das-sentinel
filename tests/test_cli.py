from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from das_sentinel.cli import app
from typer.testing import CliRunner

from tests.fixtures import write_zd_h5

runner = CliRunner()


@pytest.fixture(autouse=True)
def safe_sqlite_runtime(monkeypatch):
    monkeypatch.setattr(
        "das_sentinel.cli.sqlite3", SimpleNamespace(sqlite_version="3.53.3")
    )
    monkeypatch.setattr(
        "das_sentinel.persistence.db.validate_selected_sqlite_runtime",
        lambda: (3, 53, 3),
    )


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeApi:
    """Records requests and returns canned responses for CLI tests."""

    def __init__(self, payloads=None):
        self.requests = []
        self.payloads = payloads or {}

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        if (method, path) in self.payloads:
            payload = self.payloads[(method, path)]
        elif method in self.payloads:
            payload = self.payloads[method]
        elif method == "GET" or path.endswith("/scan"):
            payload = []
        else:
            payload = {"id": "fake-1", "version": "1", "status": "active",
                       "metrics": {"macro_f1": 0.9}}
        return FakeResponse(payload, 201 if method == "POST" else 200)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)

    @property
    def last_request(self):
        method, path, kwargs = self.requests[-1]

        class _Recorded:
            pass

        recorded = _Recorded()
        recorded.method = method
        recorded.path = path
        recorded.kwargs = kwargs
        return recorded


@pytest.fixture
def mock_api(monkeypatch):
    api = FakeApi()
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)
    return api


def test_doctor_reports_cpu_and_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("DAS_APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv("DAS_DATA_ROOT", str(tmp_path / "raw"))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "CPU backend: available" in result.stdout
    assert "Application storage: writable" in result.stdout


def test_doctor_warns_about_missing_secrets_without_failing(tmp_path, monkeypatch):
    monkeypatch.setenv("DAS_APP_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv("DAS_DATA_ROOT", str(tmp_path / "raw"))
    monkeypatch.delenv("DAS_SESSION_SECRET", raising=False)
    monkeypatch.delenv("DAS_WORKER_TOKEN", raising=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.stdout
    assert "Session secret: missing" in result.stdout
    assert "Worker token: missing" in result.stdout


def test_scan_calls_api(mock_api):
    result = runner.invoke(app, ["scan", "site-1", "/data/day", "--settle-seconds", "0"])

    assert result.exit_code == 0, result.stdout
    assert mock_api.last_request.method == "POST"
    assert mock_api.last_request.path == "/api/v1/sites/site-1/scan"
    assert mock_api.last_request.kwargs["json"] == {"path": "/data/day"}


def test_site_list_calls_api(mock_api):
    result = runner.invoke(app, ["site", "list"])

    assert result.exit_code == 0, result.stdout
    assert mock_api.last_request.method == "GET"
    assert mock_api.last_request.path == "/api/v1/sites"


def test_jobs_retry_and_cancel_call_api(mock_api):
    assert runner.invoke(app, ["jobs", "retry", "job-9"]).exit_code == 0
    assert mock_api.last_request.path == "/api/v1/jobs/job-9/retry"
    assert runner.invoke(app, ["jobs", "cancel", "job-9"]).exit_code == 0
    assert mock_api.last_request.path == "/api/v1/jobs/job-9/cancel"


def test_events_export_writes_csv(tmp_path, monkeypatch):
    api = FakeApi(
        payloads={
            ("GET", "/api/v1/events"): [
                {
                    "id": "event-1",
                    "event_uid": "uid-1",
                    "site_id": "site-1",
                    "occurred_at": "2026-07-30T00:00:10Z",
                    "start_sample": 10,
                    "end_sample": 20,
                    "channel_start": 1,
                    "channel_stop": 3,
                    "machine_label": "earthquake",
                    "latest_review": None,
                }
            ]
        }
    )
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)
    out = tmp_path / "events.csv"

    result = runner.invoke(app, ["events", "export", "--out", str(out)])

    assert result.exit_code == 0, result.stdout
    text = out.read_text(encoding="utf-8")
    assert "event_uid" in text.splitlines()[0]
    assert "uid-1" in text


def test_train_and_models_commands_call_api(mock_api):
    assert runner.invoke(app, ["train", "site-1"]).exit_code == 0
    assert mock_api.last_request.path == "/api/v1/sites/site-1/models"
    assert runner.invoke(app, ["models", "publish", "model-1"]).exit_code == 0
    assert mock_api.last_request.path == "/api/v1/models/model-1/publish"
    assert runner.invoke(app, ["models", "rollback", "site-1"]).exit_code == 0
    assert mock_api.last_request.path == "/api/v1/sites/site-1/models/rollback"


def test_benchmark_runs_local_file(tmp_path, monkeypatch):
    source = tmp_path / "sample.h5"
    write_zd_h5(source, fs=1000.0)
    monkeypatch.setenv("DAS_APP_DATA_DIR", str(tmp_path / "app-data"))

    result = runner.invoke(app, ["benchmark", str(source)])

    assert result.exit_code == 0, result.stdout
    assert "backend: cpu" in result.stdout
    assert "events:" in result.stdout


def test_admin_init_seeds_single_admin(tmp_path, monkeypatch):
    monkeypatch.setenv("DAS_APP_DATA_DIR", str(tmp_path / "app-data"))

    result = runner.invoke(app, ["admin", "init", "--password", "correct horse"])

    assert result.exit_code == 0, result.stdout
    database = tmp_path / "app-data" / "das-sentinel.sqlite3"
    with sqlite3.connect(database) as connection:
        count = connection.execute("SELECT COUNT(*) FROM admins").fetchone()[0]
    assert count == 1

    second = runner.invoke(app, ["admin", "init", "--password", "correct horse"])
    assert second.exit_code != 0
