from __future__ import annotations

import numpy as np
from das_sentinel.signal.preprocess import PreprocessedBlock


def preprocessed_block(
    data: np.ndarray, *, sample_offset: int = 1_000, sample_rate_hz: float = 100.0
) -> PreprocessedBlock:
    return PreprocessedBlock(
        data=data.astype(np.float32, copy=False),
        bad_channel_mask=np.zeros(data.shape[1], dtype=bool),
        backend_name="cpu",
        sample_offset=sample_offset,
        sample_rate_hz=sample_rate_hz,
        channel_distance_m=np.arange(data.shape[1], dtype=np.float32),
    )


def multichannel_impulse() -> PreprocessedBlock:
    rng = np.random.default_rng(12)
    data = rng.normal(0.0, 0.03, size=(1_200, 12)).astype(np.float32)
    pulse = np.hanning(9).astype(np.float32) * 12.0
    data[496:505, 3:9] += pulse[:, None]
    return preprocessed_block(data)


def single_channel_impulse() -> PreprocessedBlock:
    rng = np.random.default_rng(13)
    data = rng.normal(0.0, 0.03, size=(1_200, 12)).astype(np.float32)
    data[496:505, 3] += np.hanning(9).astype(np.float32) * 12.0
    return preprocessed_block(data)


def sustained_band_limited_event() -> PreprocessedBlock:
    rng = np.random.default_rng(14)
    data = rng.normal(0.0, 0.04, size=(1_200, 12)).astype(np.float32)
    time = np.arange(320, 720, dtype=np.float32) / 100.0
    data[320:720, 2:10] += (2.0 * np.sin(2 * np.pi * 8.0 * time))[:, None]
    return preprocessed_block(data)
