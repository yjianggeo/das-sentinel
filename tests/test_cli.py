from __future__ import annotations

import sqlite3
from collections import deque
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
        if isinstance(payload, deque):
            payload = payload.popleft()
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
    mock_api.payloads[("GET", "/api/v1/sites/site-1")] = {
        "id": "site-1", "data_path": "C:/data"
    }
    mock_api.payloads[("POST", "/api/v1/sites/site-1/scan")] = scan_snapshot(
        "scan-1", "completed"
    )

    result = runner.invoke(app, ["scan", "site-1", "jjk/day"])

    assert result.exit_code == 0, result.stdout
    assert mock_api.last_request.method == "POST"
    assert mock_api.last_request.path == "/api/v1/sites/site-1/scan"
    assert mock_api.last_request.kwargs["json"] == {"path": "jjk/day"}


def scan_snapshot(
    scan_id: str,
    phase: str,
    *,
    submitted_files: int = 0,
    error_message: str | None = None,
) -> dict[str, object]:
    return {
        "id": scan_id,
        "site_id": "site-1",
        "path": "jjk/day",
        "phase": phase,
        "initial_checked_files": 0,
        "initial_found_files": 0,
        "verification_checked_files": 0,
        "verification_found_files": 0,
        "stable_files": 0,
        "submitted_files": submitted_files,
        "skipped_files": 0,
        "stability_seconds_remaining": 0,
        "error_message": error_message,
    }


def test_scan_posts_once_and_polls_until_automatic_recheck_finishes(monkeypatch):
    api = FakeApi(payloads={
        ("GET", "/api/v1/sites/site-1"): {"id": "site-1", "data_path": "C:/data"},
        ("POST", "/api/v1/sites/site-1/scan"): scan_snapshot(
            "scan-1", "stability_wait"
        ),
        ("GET", "/api/v1/scans/scan-1"): deque([
            scan_snapshot("scan-1", "verification_scan"),
            scan_snapshot("scan-1", "completed", submitted_files=2),
        ]),
    })
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)
    monkeypatch.setattr("das_sentinel.cli.time.sleep", lambda _: None)

    result = runner.invoke(app, ["scan", "site-1", "jjk/day"])

    assert result.exit_code == 0, result.stdout
    assert sum(path.endswith("/scan") for _, path, _ in api.requests) == 1
    assert "扫描完成" in result.stdout


@pytest.mark.parametrize("command", ["scan", "watch"])
def test_scan_commands_require_path(command):
    result = runner.invoke(app, [command, "site-1"])

    assert result.exit_code == 2
    assert "Missing argument" in result.stderr


@pytest.mark.parametrize("path", [".", "..", "C:/data", "/outside", r"\outside"])
def test_scan_rejects_root_parent_or_rooted_path_before_post(monkeypatch, path):
    api = FakeApi(payloads={
        ("GET", "/api/v1/sites/site-1"): {"id": "site-1", "data_path": "C:/data"},
    })
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)

    result = runner.invoke(app, ["scan", "site-1", path])

    assert result.exit_code == 2
    assert "PATH 必须是站点根目录下的具体数据目录" in result.stderr
    assert not any(method == "POST" for method, _, _ in api.requests)


def test_scan_converts_absolute_in_root_path_to_posix_relative_path(monkeypatch):
    api = FakeApi(payloads={
        ("GET", "/api/v1/sites/site-1"): {"id": "site-1", "data_path": "C:/data"},
        ("POST", "/api/v1/sites/site-1/scan"): scan_snapshot(
            "scan-1", "completed"
        ),
    })
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)

    result = runner.invoke(app, ["scan", "site-1", "C:/data/jjk\\day"])

    assert result.exit_code == 0, result.stdout
    assert api.last_request.kwargs["json"] == {"path": "jjk/day"}


def test_scan_exits_one_with_server_failure_message(monkeypatch):
    api = FakeApi(payloads={
        ("GET", "/api/v1/sites/site-1"): {"id": "site-1", "data_path": "C:/data"},
        ("POST", "/api/v1/sites/site-1/scan"): scan_snapshot("scan-1", "failed",
                                                              error_message="扫描目录不可读"),
    })
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)

    result = runner.invoke(app, ["scan", "site-1", "jjk/day"])

    assert result.exit_code == 1
    assert "扫描目录不可读" in result.stdout


def test_watch_once_starts_and_completes_one_server_owned_scan(monkeypatch):
    api = FakeApi(payloads={
        ("GET", "/api/v1/sites/site-1"): {"id": "site-1", "data_path": "C:/data"},
        ("POST", "/api/v1/sites/site-1/scan"): scan_snapshot(
            "scan-1", "stability_wait"
        ),
        ("GET", "/api/v1/scans/scan-1"): scan_snapshot(
            "scan-1", "completed", submitted_files=1
        ),
    })
    monkeypatch.setattr("das_sentinel.cli._connect", lambda: api)
    monkeypatch.setattr("das_sentinel.cli.time.sleep", lambda _: None)

    result = runner.invoke(app, ["watch", "site-1", "jjk/day", "--once"])

    assert result.exit_code == 0, result.stdout
    assert sum(path.endswith("/scan") for _, path, _ in api.requests) == 1
    assert "扫描完成" in result.stdout


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
