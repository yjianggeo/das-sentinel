"""Static assertions for the Compose deployment files.

The CI/host environment has no Docker daemon, so these tests validate the
declared configuration (read-only raw-data mounts, GPU reservation, CPU image
purity) without building anything.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(*names: str) -> dict:
    """Merge Compose files like `docker compose -f a -f b config` would."""
    merged: dict = {"services": {}, "volumes": {}}
    for name in names:
        document = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
        for key, value in document.items():
            if isinstance(value, dict) and key in merged:
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


@pytest.fixture(scope="module")
def compose_cpu() -> dict:
    return _load("compose.yaml", "compose.cpu.yaml")


@pytest.fixture(scope="module")
def compose_gpu() -> dict:
    return _load("compose.yaml", "compose.gpu.yaml")


def test_raw_data_mounts_are_read_only(compose_cpu: dict) -> None:
    api_raw = next(v for v in compose_cpu["services"]["api"]["volumes"] if v["target"] == "/data")
    worker_raw = next(
        v for v in compose_cpu["services"]["worker"]["volumes"] if v["target"] == "/data"
    )
    assert api_raw["read_only"] is True
    assert worker_raw["read_only"] is True


def test_app_data_is_a_writable_named_volume(compose_cpu: dict) -> None:
    for service in ("api", "worker"):
        app_data = next(
            v for v in compose_cpu["services"][service]["volumes"] if v["target"] == "/app-data"
        )
        assert app_data.get("read_only") is not True
        assert app_data["type"] == "volume"
    assert "app-data" in compose_cpu["volumes"]


def test_gpu_worker_reserves_nvidia_device(compose_gpu: dict) -> None:
    devices = compose_gpu["services"]["worker"]["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [{"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}]


def test_gpu_worker_requests_cuda_backend(compose_gpu: dict) -> None:
    assert compose_gpu["services"]["worker"]["environment"]["DAS_WORKER_DEVICE"] == "cuda"


def test_cpu_image_does_not_install_cupy() -> None:
    text = (ROOT / "docker" / "Dockerfile.cpu").read_text("utf-8")
    assert "cupy" not in text.lower()


def test_gpu_image_installs_cupy_and_cuda_runtime() -> None:
    text = (ROOT / "docker" / "Dockerfile.gpu").read_text("utf-8").lower()
    assert "nvidia/cuda" in text
    assert "cupy-cuda12x" in text or ".[gpu]" in text


def test_gpu_image_uses_cuda_devel_base() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.gpu").read_text("utf-8")
    assert "FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS runtime" in dockerfile


def test_api_has_healthcheck_and_runs_non_root(compose_cpu: dict) -> None:
    assert "healthcheck" in compose_cpu["services"]["api"]
    dockerfile = (ROOT / "docker" / "Dockerfile.cpu").read_text("utf-8")
    assert "USER appuser" in dockerfile


def test_env_example_covers_required_variables() -> None:
    text = (ROOT / ".env.example").read_text("utf-8")
    for name in ("DAS_HOST_DATA_ROOT", "DAS_SESSION_SECRET", "DAS_WORKER_TOKEN"):
        assert name in text


def test_host_data_root_is_only_a_bind_source(compose_cpu: dict) -> None:
    for service in ("api", "worker"):
        raw = next(
            v
            for v in compose_cpu["services"][service]["volumes"]
            if v["target"] == "/data"
        )
        assert raw["source"] == "${DAS_HOST_DATA_ROOT}"
        assert compose_cpu["services"][service]["environment"]["DAS_DATA_ROOT"] == "/data"


def test_services_restart_and_api_binds_loopback(compose_cpu: dict) -> None:
    assert compose_cpu["services"]["api"]["ports"] == [
        "127.0.0.1:${DAS_HTTP_PORT:-8000}:8000"
    ]
    assert compose_cpu["services"]["api"]["restart"] == "unless-stopped"
    assert compose_cpu["services"]["worker"]["restart"] == "unless-stopped"


def test_gpu_worker_uses_the_nas_runtime_defaults(compose_gpu: dict) -> None:
    worker = compose_gpu["services"]["worker"]
    raw = next(volume for volume in worker["volumes"] if volume["target"] == "/data")
    assert worker["restart"] == "unless-stopped"
    assert worker["environment"]["DAS_DATA_ROOT"] == "/data"
    assert raw["source"] == "${DAS_HOST_DATA_ROOT}"


def test_cpu_image_pins_verified_static_sqlite() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile.cpu").read_text("utf-8")
    builder = (ROOT / "docker" / "build-pysqlite3-wheel.sh").read_text("utf-8")
    assert "sqlite-amalgamation-3530300.zip" in builder
    assert (
        "646421e12aac110282ef8cc68f1a62d4bb15fc7b8f09da0b53e29ee690500431"
        in builder
    )
    assert (
        "fbc69bfdc0cb43a5badd5403b126d5151371b5037e0397ba9802bb440c5b0021"
        in builder
    )
    assert "build_static bdist_wheel" in builder
    assert "SELECT sqlite_version()" in dockerfile


def test_static_sqlite_builder_installs_legacy_build_dependencies() -> None:
    builder = (ROOT / "docker" / "build-pysqlite3-wheel.sh").read_text("utf-8")
    install = "python -m pip install setuptools wheel"
    assert install in builder
    assert builder.index(install) < builder.index("python setup.py build_static bdist_wheel")


def test_ci_declares_required_release_jobs() -> None:
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    assert set(workflow["jobs"]) >= {
        "python",
        "frontend",
        "cpu-container",
        "gpu-image-build",
    }


def test_readme_uses_current_runtime_environment_and_factory_entrypoint() -> None:
    readme = (ROOT / "README.md").read_text("utf-8")
    quickstart = readme.split("## 本地开发", maxsplit=1)[0]
    assert "DAS_HOST_DATA_ROOT" in quickstart
    assert "DAS_DATA_ROOT" not in quickstart
    assert "uvicorn das_sentinel.api.app:create_runtime_app --factory --reload" in readme


def _ci_workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))


def _run_steps(job: dict) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def _assert_ci_release_gates(workflow: dict) -> None:
    assert set(workflow["jobs"]) >= {
        "python",
        "frontend",
        "cpu-container",
        "gpu-image-build",
    }
    for job_name in ("python", "frontend", "cpu-container", "gpu-image-build"):
        assert workflow["jobs"][job_name]["steps"][0] == {
            "uses": "actions/checkout@v4"
        }

    python_job = workflow["jobs"]["python"]
    assert python_job["steps"][1] == {
        "uses": "actions/setup-python@v5",
        "with": {"python-version": "3.12"},
    }
    assert _run_steps(python_job) == [
        "sudo apt-get update && sudo apt-get install -y build-essential curl unzip",
        'docker/build-pysqlite3-wheel.sh "$RUNNER_TEMP/wheels"',
        'python -m pip install "$RUNNER_TEMP"/wheels/*.whl',
        'python -m pip install -e ".[dev]"',
        "python -m pip check",
        'python -c "from das_sentinel.persistence import validate_selected_sqlite_runtime; assert validate_selected_sqlite_runtime() == (3, 53, 3)"',
        "ruff check .",
        "pytest -q",
    ]

    frontend_job = workflow["jobs"]["frontend"]
    assert frontend_job["defaults"] == {"run": {"working-directory": "frontend"}}
    assert frontend_job["steps"][1] == {
        "uses": "actions/setup-node@v4",
        "with": {
            "node-version": "24",
            "cache": "npm",
            "cache-dependency-path": "frontend/package-lock.json",
        },
    }
    assert _run_steps(frontend_job) == ["npm ci", "npm test -- --run", "npm run build"]

    cpu_job = workflow["jobs"]["cpu-container"]
    assert cpu_job.get("env") == {"COMPOSE_PROJECT_NAME": "das-sentinel-ci"}
    cpu_steps = cpu_job["steps"]
    assert cpu_steps[1] == {
        "run": "docker build -f docker/Dockerfile.cpu -t das-sentinel:ci ."
    }
    assert cpu_steps[2] == {
        "name": "Write isolated Compose environment",
        "run": """mkdir -p \"$RUNNER_TEMP/das-data\"
printf '%s\\n' \\
  \"DAS_HOST_DATA_ROOT=$RUNNER_TEMP/das-data\" \\
  \"DAS_SESSION_SECRET=ci-session-secret-0123456789abcdef\" \\
  \"DAS_WORKER_TOKEN=ci-worker-token-0123456789abcdefghi\" \\
  \"DAS_SECURE_COOKIES=false\" \\
  \"DAS_HTTP_PORT=8000\" > .env
""",
    }
    assert cpu_steps[3] == {"run": "docker compose -f compose.yaml -f compose.cpu.yaml config"}
    assert cpu_steps[4] == {"run": "docker compose -f compose.yaml -f compose.cpu.yaml up -d api"}
    assert cpu_steps[5] == {
        "name": "Wait for JSON health",
        "run": """for attempt in $(seq 1 60); do
  python -c \"import json,urllib.request; assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/healthz')) == {'status':'ok'}\" && exit 0
  sleep 2
done
exit 1
""",
    }
    assert cpu_steps[6] == {
        "run": "docker compose -f compose.yaml -f compose.cpu.yaml exec -T api das-sentinel admin init --password ci-admin-password"
    }
    assert cpu_steps[7] == {"run": "docker compose -f compose.yaml -f compose.cpu.yaml up -d worker"}
    assert cpu_steps[8] == {
        "name": "Wait for worker to be running",
        "run": """for attempt in $(seq 1 30); do
  worker_services=\"$(docker compose -f compose.yaml -f compose.cpu.yaml ps --status running --services worker)\"
  if [ \"$worker_services\" = \"worker\" ]; then
    exit 0
  fi
  sleep 2
done
exit 1
""",
    }
    assert cpu_steps[9] == {
        "run": "docker compose -f compose.yaml -f compose.cpu.yaml exec -T api python -c \"import pysqlite3.dbapi2 as db; assert db.sqlite_version == '3.53.3'\""
    }
    assert cpu_steps[10:] == [
        {"if": "always()", "run": "docker compose -f compose.yaml -f compose.cpu.yaml logs --no-color"},
        {"if": "always()", "run": "docker compose -f compose.yaml -f compose.cpu.yaml down --volumes --remove-orphans"},
    ]

    gpu_job = workflow["jobs"]["gpu-image-build"]
    assert _run_steps(gpu_job) == [
        "docker build -f docker/Dockerfile.gpu -t das-sentinel-gpu:ci ."
    ]


def test_ci_enforces_all_release_gates() -> None:
    _assert_ci_release_gates(_ci_workflow())


@pytest.mark.parametrize(
    ("job_name", "step_index"),
    [
        pytest.param("python", 1, id="setup-python"),
        pytest.param("python", 2, id="python-build-dependencies"),
        pytest.param("python", 3, id="static-wheel-build"),
        pytest.param("python", 4, id="static-wheel-install"),
        pytest.param("python", 5, id="editable-dev-install"),
        pytest.param("python", 6, id="pip-check"),
        pytest.param("python", 7, id="sqlite-version"),
        pytest.param("python", 8, id="ruff"),
        pytest.param("python", 9, id="pytest"),
        pytest.param("frontend", 1, id="setup-node"),
        pytest.param("frontend", 2, id="npm-ci"),
        pytest.param("frontend", 3, id="npm-test"),
        pytest.param("frontend", 4, id="npm-build"),
        pytest.param("cpu-container", 1, id="cpu-image-build"),
        pytest.param("cpu-container", 2, id="isolated-environment"),
        pytest.param("cpu-container", 3, id="compose-config"),
        pytest.param("cpu-container", 4, id="api-start"),
        pytest.param("cpu-container", 5, id="json-health"),
        pytest.param("cpu-container", 6, id="admin-init"),
        pytest.param("cpu-container", 7, id="worker-start"),
        pytest.param("cpu-container", 8, id="worker-running"),
        pytest.param("cpu-container", 9, id="container-sqlite"),
        pytest.param("cpu-container", 10, id="always-logs"),
        pytest.param("cpu-container", 11, id="always-down"),
        pytest.param("gpu-image-build", 1, id="gpu-image-build"),
    ],
)
def test_ci_gate_contract_rejects_missing_steps(job_name: str, step_index: int) -> None:
    workflow = deepcopy(_ci_workflow())
    workflow["jobs"][job_name]["steps"].pop(step_index)
    with pytest.raises(AssertionError):
        _assert_ci_release_gates(workflow)


@pytest.mark.parametrize(
    "job_name", ["python", "frontend", "cpu-container", "gpu-image-build"]
)
def test_ci_gate_contract_rejects_downgraded_checkout(job_name: str) -> None:
    workflow = deepcopy(_ci_workflow())
    workflow["jobs"][job_name]["steps"][0] = {"uses": "actions/checkout@v3"}
    with pytest.raises(AssertionError):
        _assert_ci_release_gates(workflow)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda workflow: workflow["jobs"]["python"]["steps"].__setitem__(
                1,
                {"uses": "actions/setup-python@v5", "with": {"python-version": "3.11"}},
            ),
            id="wrong-python-version",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["python"]["steps"][3].__setitem__(
                "run", 'echo docker/build-pysqlite3-wheel.sh "$RUNNER_TEMP/wheels"'
            ),
            id="echo-static-wheel-build",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["frontend"].__setitem__(
                "defaults", {"run": {"working-directory": "."}}
            ),
            id="wrong-frontend-directory",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["frontend"]["steps"].__setitem__(
                1,
                {"uses": "actions/setup-node@v4", "with": {"node-version": "22"}},
            ),
            id="wrong-node-setup",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["frontend"]["steps"][2].__setitem__(
                "run", "echo npm ci"
            ),
            id="echo-npm-ci",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"]["steps"].__setitem__(
                1, {"run": "echo docker build -f docker/Dockerfile.cpu"}
            ),
            id="echo-cpu-build",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"]["steps"].__setitem__(
                2,
                {
                    "name": "Write isolated Compose environment",
                    "run": "echo DAS_HOST_DATA_ROOT",
                },
            ),
            id="incomplete-isolated-environment",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"]["steps"][5].__setitem__(
                "run", "curl --fail http://127.0.0.1:8000/healthz"
            ),
            id="unnamed-json-health-contract",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"]["steps"][8].__setitem__(
                "run", "docker compose ps worker"
            ),
            id="unbounded-worker-contract",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"].pop("env"),
            id="missing-cpu-project-environment",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"].__setitem__(
                "env", {"COMPOSE_PROJECT_NAME": "shared"}
            ),
            id="shared-cpu-project-environment",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["cpu-container"]["steps"][10].__setitem__(
                "if", "success()"
            ),
            id="logs-not-always",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["gpu-image-build"]["steps"].append(
                {"run": "docker run das-sentinel-gpu:ci"}
            ),
            id="gpu-docker-run",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["gpu-image-build"]["steps"].append(
                {"run": "docker run --gpus all das-sentinel-gpu:ci"}
            ),
            id="gpu-gpus-flag",
        ),
        pytest.param(
            lambda workflow: workflow["jobs"]["gpu-image-build"]["steps"].append(
                {"run": "python -m das_sentinel.worker.smoke_test_cuda"}
            ),
            id="gpu-smoke-test",
        ),
    ],
)
def test_ci_gate_contract_rejects_replacements_and_gpu_runtime_steps(mutate) -> None:
    workflow = deepcopy(_ci_workflow())
    mutate(workflow)
    with pytest.raises(AssertionError):
        _assert_ci_release_gates(workflow)
