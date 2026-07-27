from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import DasBlock
from das_sentinel.signal.backend import ArrayBackend


@dataclass(frozen=True)
class PreprocessedBlock:
    data: Any
    bad_channel_mask: np.ndarray
    backend_name: str
    sample_offset: int
    sample_rate_hz: float
    channel_distance_m: np.ndarray


def find_bad_channels(data: Any, rms_z: float = 8.0) -> np.ndarray:
    """Flag channels containing non-finite values, no energy, or RMS outliers."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("data must have shape [time, channel]")

    finite = np.isfinite(values).all(axis=0)
    finite_values = np.where(np.isfinite(values), values, 0.0)
    epsilon = np.finfo(values.dtype).eps
    rms = np.sqrt(np.mean(np.square(finite_values), axis=0))
    bad = ~finite | (np.ptp(finite_values, axis=0) <= epsilon) | ~np.isfinite(rms)
    valid_rms = rms[~bad]
    if valid_rms.size:
        median = np.median(valid_rms)
        mad = np.median(np.abs(valid_rms - median))
        tolerance = epsilon * max(1.0, float(abs(median)))
        if mad > tolerance:
            bad |= np.abs(rms - median) > rms_z * 1.4826 * mad
        else:
            bad |= ~np.isclose(rms, median, rtol=1e-5, atol=tolerance)
    return bad.astype(bool, copy=False)


def _raised_cosine_band_gain(frequency: Any, low_hz: float, high_hz: float, xp: Any) -> Any:
    low_width = low_hz * 0.1
    high_width = high_hz * 0.1
    low_start, low_stop = max(0.0, low_hz - low_width), low_hz + low_width
    high_start, high_stop = high_hz - high_width, high_hz + high_width
    low_gain = 0.5 - 0.5 * xp.cos(
        xp.pi * xp.clip((frequency - low_start) / (low_stop - low_start), 0.0, 1.0)
    )
    high_gain = 0.5 + 0.5 * xp.cos(
        xp.pi * xp.clip((frequency - high_start) / (high_stop - high_start), 0.0, 1.0)
    )
    return low_gain * high_gain


def fft_bandpass(
    data: Any,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    backend: ArrayBackend,
) -> Any:
    """Apply a deterministic raised-cosine FFT band-pass on the chosen backend."""
    if sample_rate_hz <= 0 or not 0 <= low_hz < high_hz:
        raise ValueError("frequency range must satisfy 0 <= low < high and sample rate must be positive")
    xp = backend.xp
    values = backend.asarray(data)
    if values.ndim != 2:
        raise ValueError("data must have shape [time, channel]")
    frequency = xp.fft.rfftfreq(values.shape[0], d=1.0 / sample_rate_hz)
    gain = _raised_cosine_band_gain(frequency, low_hz, high_hz, xp)
    spectrum = xp.fft.rfft(values, axis=0)
    return xp.fft.irfft(spectrum * gain[:, None], n=values.shape[0], axis=0).astype(values.dtype)


def _linear_detrend(data: Any, xp: Any) -> Any:
    time = xp.arange(data.shape[0], dtype=data.dtype)
    centered_time = time - xp.mean(time)
    denominator = xp.sum(centered_time * centered_time)
    slope = xp.sum(centered_time[:, None] * (data - xp.mean(data, axis=0)), axis=0) / denominator
    return data - (xp.mean(data, axis=0) + centered_time[:, None] * slope)


def _robust_scale(data: Any, bad_channel_mask: np.ndarray, xp: Any) -> Any:
    median = xp.median(data, axis=0)
    centered = data - median
    scale = 1.4826 * xp.median(xp.abs(centered), axis=0)
    scale = xp.where(scale > xp.finfo(data.dtype).eps, scale, 1.0)
    scaled = centered / scale
    if bad_channel_mask.any():
        scaled[:, bad_channel_mask] = 0.0
    return scaled


def preprocess_block(
    block: DasBlock, profile: ProcessingProfile, backend: ArrayBackend
) -> PreprocessedBlock:
    """Detrend, optionally de-common-mode, band-pass, and robustly scale a DAS block."""
    xp = backend.xp
    bad_channel_mask = find_bad_channels(block.data, profile.bad_channel_rms_z)
    data = backend.asarray(block.data, dtype=xp.float32).copy()
    data = xp.where(xp.isfinite(data), data, 0.0)
    data = _linear_detrend(data, xp)
    good_channels = np.flatnonzero(~bad_channel_mask)
    if profile.common_mode_removal and good_channels.size:
        data = data - xp.median(data[:, good_channels], axis=1, keepdims=True)
    data = fft_bandpass(
        data,
        block.metadata.sample_rate_hz,
        profile.stalta_freq_low_hz,
        profile.stalta_freq_high_hz,
        backend,
    )
    data = _robust_scale(data, bad_channel_mask, xp)
    return PreprocessedBlock(
        data=data,
        bad_channel_mask=bad_channel_mask,
        backend_name=backend.name,
        sample_offset=block.sample_offset,
        sample_rate_hz=block.metadata.sample_rate_hz,
        channel_distance_m=block.metadata.channel_distance_m.copy(),
    )
