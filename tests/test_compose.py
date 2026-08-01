"""Static assertions for the Compose deployment files.

The CI/host environment has no Docker daemon, so these tests validate the
declared configuration (read-only raw-data mounts, GPU reservation, CPU image
purity) without building anything.
"""

from __future__ import annotations

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


def test_api_has_healthcheck_and_runs_non_root(compose_cpu: dict) -> None:
    assert "healthcheck" in compose_cpu["services"]["api"]
    dockerfile = (ROOT / "docker" / "Dockerfile.cpu").read_text("utf-8")
    assert "USER appuser" in dockerfile


def test_env_example_covers_required_variables() -> None:
    text = (ROOT / ".env.example").read_text("utf-8")
    for name in ("DAS_DATA_ROOT", "DAS_SESSION_SECRET", "DAS_WORKER_TOKEN"):
        assert name in text
