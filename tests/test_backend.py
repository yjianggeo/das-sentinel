from types import SimpleNamespace

import numpy as np
import pytest
from das_sentinel.signal.backend import (
    BackendUnavailable,
    select_backend,
    smoke_test_cuda,
)
from das_sentinel.signal.preprocess import fft_bandpass


class FakeCupy:
    float32 = np.float32

    def __init__(self, calls, *, host=None):
        self.calls = calls
        self.host = host
        self.fft = SimpleNamespace(rfft=self._rfft)
        null_stream = SimpleNamespace(synchronize=self._synchronize)
        self.cuda = SimpleNamespace(Stream=SimpleNamespace(null=null_stream))

    def asarray(self, values, *, dtype):
        self.calls.append("asarray")
        return np.asarray(values, dtype=dtype)

    def _rfft(self, values):
        self.calls.append("rfft")
        return np.fft.rfft(values)

    def asnumpy(self, spectrum):
        self.calls.append("asnumpy")
        return spectrum if self.host is None else self.host

    def _synchronize(self):
        self.calls.append("synchronize")


def test_auto_backend_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("das_sentinel.signal.backend._load_cupy", lambda: None)
    monkeypatch.setattr(
        "das_sentinel.signal.backend._CUPY_LOAD_ERROR",
        "CuPy/CUDA is unavailable: test CUDA load failure",
    )

    backend = select_backend("auto")

    assert backend.name == "cpu"
    assert backend.unavailable_reason == "CuPy/CUDA is unavailable: test CUDA load failure"


def test_cuda_backend_raises_recorded_unavailable_reason(monkeypatch):
    monkeypatch.setattr("das_sentinel.signal.backend._load_cupy", lambda: None)
    monkeypatch.setattr(
        "das_sentinel.signal.backend._CUPY_LOAD_ERROR",
        "CuPy/CUDA is unavailable: test CUDA load failure",
    )

    with pytest.raises(BackendUnavailable, match="test CUDA load failure"):
        select_backend("cuda")


def test_cuda_smoke_executes_fft_transfer_and_sync(monkeypatch) -> None:
    calls = []
    fake = FakeCupy(calls)
    monkeypatch.setattr("das_sentinel.signal.backend._load_cupy", lambda: fake)
    smoke_test_cuda()
    assert calls == ["asarray", "rfft", "asnumpy", "synchronize"]


def test_cuda_smoke_rejects_invalid_fft_output(monkeypatch) -> None:
    fake = FakeCupy([], host=np.array([np.nan, 0.0, 1.0]))
    monkeypatch.setattr("das_sentinel.signal.backend._load_cupy", lambda: fake)
    with pytest.raises(BackendUnavailable, match="invalid output"):
        smoke_test_cuda()


def test_cuda_fft_matches_cpu_or_skips_when_unavailable():
    cuda = select_backend("auto")
    if cuda.name != "cuda":
        pytest.skip(cuda.unavailable_reason or "CUDA backend unavailable")

    fs = 100.0
    time = np.arange(1000) / fs
    block = (4.0 + np.sin(2 * np.pi * 10 * time))[:, None].astype(np.float32)
    cpu = fft_bandpass(block, fs, 5.0, 20.0, select_backend("cpu"))
    gpu = cuda.asnumpy(fft_bandpass(block, fs, 5.0, 20.0, cuda))

    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-5)
