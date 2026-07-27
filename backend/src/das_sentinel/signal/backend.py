from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class BackendUnavailable(RuntimeError):
    """Raised when the requested array backend cannot be used."""


@dataclass(frozen=True)
class ArrayBackend:
    name: str
    xp: Any
    unavailable_reason: str | None = None

    def asarray(self, value: Any, **kwargs: Any) -> Any:
        return self.xp.asarray(value, **kwargs)

    def asnumpy(self, value: Any) -> np.ndarray:
        if self.name == "cpu":
            return np.asarray(value)
        return self.xp.asnumpy(value)

    def clear_cache(self) -> None:
        if self.name == "cuda":
            self.xp.get_default_memory_pool().free_all_blocks()


_CUPY_LOAD_ERROR: str | None = None


def _load_cupy() -> Any | None:
    """Load CuPy only when CUDA has been requested."""
    global _CUPY_LOAD_ERROR
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            _CUPY_LOAD_ERROR = "no CUDA devices are available"
            return None
    except (ImportError, OSError, RuntimeError) as error:
        _CUPY_LOAD_ERROR = f"CuPy/CUDA is unavailable: {error}"
        return None
    return cupy


def _cuda_backend() -> ArrayBackend | None:
    cupy = _load_cupy()
    if cupy is None:
        return None
    return ArrayBackend(name="cuda", xp=cupy)


def select_backend(device: str) -> ArrayBackend:
    """Select the requested array backend without importing CuPy for CPU work."""
    if device == "cpu":
        return ArrayBackend(name="cpu", xp=np)
    if device not in {"auto", "cuda"}:
        raise ValueError(f"unknown device: {device}")

    cuda = _cuda_backend()
    if cuda is not None:
        return cuda

    reason = _CUPY_LOAD_ERROR or "CuPy/CUDA is unavailable"
    if device == "cuda":
        raise BackendUnavailable(reason)
    return ArrayBackend(name="cpu", xp=np, unavailable_reason=reason)
