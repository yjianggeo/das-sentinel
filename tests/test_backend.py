import numpy as np
import pytest
from das_sentinel.signal.backend import BackendUnavailable, select_backend
from das_sentinel.signal.preprocess import fft_bandpass


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
