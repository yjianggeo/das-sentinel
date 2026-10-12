"""Independent 5 Hz continuous-record analysis; all candidates require review.

``analysis_start``/``analysis_stop`` are half-open indices into ``block.data``.
The reader supplies at least 300 s of real data on either side. Filtering uses
one full-record linear detrend and fourth-order Butterworth SOS, 0.01--0.4 Hz,
forward/backward with 100 s reflected padding, then crops the target. Cutoff
amplitude is approximately 0.5 after two passes. No spatial common mode or
channel amplitude normalization is applied. Invalid/flat channels are zero in
``filtered_data`` and false in ``valid_channel_mask``; source column order stays
unchanged. Input/output units are those of the reader, not calibrated strain.

Frozen engineering detection criteria: 20 s averaged analytic-signal power,
at least four channels and 25% of valid channels above four times their own
background median envelope power; at least 20 s supported time per candidate,
merging gaps of at most 30 s. Background is the real interval 200--30 s before
the target, avoiding the outer 100 s of the provided pre-context. Incomplete
300 s context or targets shorter than 100 s disable automatic candidates.
Neither zero moveout nor the high-frequency persistent-background gate is used.

``measurements`` describes the whole target: band_rms (original input unit),
snr_db (10 log10 mean target power / mean background power), dominant_frequency_hz,
frequency_resolution_hz (FFT bin spacing: 1 / measured duration, without zero
padding; this is not the two-peak resolving ability of the Hann window),
band_fraction_0_01_0_05 / band_fraction_0_05_0_1 / band_fraction_0_1_0_4
(Hann-tapered spectral powers partitioning 0.01--0.4 Hz), active_channel_fraction
(channel mean power >= 4 times its background mean power), coherence (zero-lag
semblance, sum-channel power / [channel count * summed individual power]),
energy_duration_s (5--95% of cumulative squared-signal energy), and validity /
context / algorithm metadata. Unmeasurable numeric fields are None, never NaN.
Dominant frequency and the lowest subband are unavailable below 100 s; a
subband without a Fourier bin is also unavailable. Coherence is descriptive and
is not a detection gate. background_start_seconds/background_end_seconds give
the available, clipped background bounds relative to target start (normally
[-200, -30]); less than 100 s available background is not used for SNR.

Candidate dictionaries contain start_seconds/end_seconds/peak_seconds relative
to the target, score (engineering evidence, not probability), and the same
snr_db, active_channel_fraction, coherence, dominant_frequency_hz,
frequency_resolution_hz, energy_duration_s definitions measured on that interval.
Score = 0.5 * clipped(SNR / 12 dB) + 0.5 * active_channel_fraction.
Warnings describe missing context, unresolved low frequency, and target-clipped
candidate boundaries and candidates shorter than 100 s; these are not wave-type
or earthquake classifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, detrend, hilbert, sosfiltfilt

from das_sentinel.domain import DasBlock

ALGORITHM_VERSION = "low-frequency-v1"
_SAMPLE_RATE = 5.0
_CONTEXT_SECONDS = 300.0
_MIN_DURATION_SECONDS = 20.0
_MERGE_GAP_SECONDS = 30.0
_POWER_RATIO = 4.0
_MIN_CHANNELS = 4
_MIN_CHANNEL_FRACTION = 0.25
_SUBBANDS = (
    ("band_fraction_0_01_0_05", 0.01, 0.05),
    ("band_fraction_0_05_0_1", 0.05, 0.1),
    ("band_fraction_0_1_0_4", 0.1, 0.4),
)


@dataclass(frozen=True)
class LowFrequencyAnalysis:
    filtered_data: np.ndarray
    valid_channel_mask: np.ndarray
    candidates: tuple[dict[str, Any], ...]
    measurements: dict[str, Any]
    warnings: tuple[str, ...]


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _measure(data: np.ndarray, background: np.ndarray | None) -> dict[str, Any]:
    samples, channels = data.shape
    result: dict[str, Any] = dict.fromkeys((
        "band_rms", "snr_db", "dominant_frequency_hz", "frequency_resolution_hz",
        "active_channel_fraction", "coherence", "energy_duration_s",
        *(name for name, _, _ in _SUBBANDS),
    ))
    if samples:
        result["frequency_resolution_hz"] = _SAMPLE_RATE / samples
    if not samples or not channels:
        return result
    channel_power = np.mean(np.square(data), axis=0)
    total_power = float(channel_power.sum())
    result["band_rms"] = _finite(np.sqrt(total_power / channels))
    if background is not None and background.size:
        background_power = np.mean(np.square(background), axis=0)
        reference = float(background_power.sum())
        if reference > 0 and total_power > 0:
            result["snr_db"] = _finite(10.0 * np.log10(total_power / reference))
        supported = background_power > 0
        if supported.all():
            result["active_channel_fraction"] = float(np.mean(
                channel_power >= _POWER_RATIO * background_power
            ))
    if total_power <= 0:
        return result
    result["coherence"] = _finite(np.clip(
        np.mean(np.square(data.sum(axis=1))) / (channels * total_power), 0.0, 1.0
    ))
    cumulative = np.cumsum(np.sum(np.square(data), axis=1))
    lower, upper = np.searchsorted(cumulative, cumulative[-1] * np.array([0.05, 0.95]))
    result["energy_duration_s"] = float((upper - lower) / _SAMPLE_RATE)
    if samples < 2:
        return result
    frequency = np.fft.rfftfreq(samples, d=1.0 / _SAMPLE_RATE)
    tapered = data * np.hanning(samples)[:, None]
    power = np.mean(np.abs(np.fft.rfft(tapered, axis=0)) ** 2, axis=1)
    in_band = (frequency >= 0.01) & (frequency <= 0.4)
    total_spectrum = float(power[in_band].sum())
    if total_spectrum <= 0:
        return result
    if samples / _SAMPLE_RATE >= 100.0:
        result["dominant_frequency_hz"] = float(frequency[in_band][np.argmax(power[in_band])])
    for name, low, high in _SUBBANDS:
        if low == 0.01 and samples / _SAMPLE_RATE < 100.0:
            continue
        selected = (frequency >= low) & (frequency < high)
        if high == 0.4:
            selected = (frequency >= low) & (frequency <= high)
        if selected.any():
            result[name] = _finite(power[selected].sum() / total_spectrum)
    return result


def _candidate_ranges(support: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.pad(support.astype(np.int8), (1, 1)))
    runs = zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)
    merged: list[tuple[int, int]] = []
    gap = round(_MERGE_GAP_SECONDS * _SAMPLE_RATE)
    for start, stop in runs:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], int(stop))
        else:
            merged.append((int(start), int(stop)))
    minimum = round(_MIN_DURATION_SECONDS * _SAMPLE_RATE)
    return [(start, stop) for start, stop in merged if support[start:stop].sum() >= minimum]


def analyze_low_frequency(
    block: DasBlock, analysis_start: int, analysis_stop: int
) -> LowFrequencyAnalysis:
    """Analyze one target in a continuous, already selected 5 Hz source block."""
    if block.metadata.sample_rate_hz != _SAMPLE_RATE:
        raise ValueError("low-frequency input must be sampled at 5 Hz")
    samples, channels = block.data.shape
    if (
        isinstance(analysis_start, (bool, np.bool_))
        or isinstance(analysis_stop, (bool, np.bool_))
        or not isinstance(analysis_start, (int, np.integer))
        or not isinstance(analysis_stop, (int, np.integer))
        or not 0 <= analysis_start < analysis_stop <= samples
    ):
        raise ValueError("analysis bounds must be a nonempty half-open sample range")
    analysis_start, analysis_stop = int(analysis_start), int(analysis_stop)
    duration = (analysis_stop - analysis_start) / _SAMPLE_RATE
    context_complete = (
        analysis_start >= _CONTEXT_SECONDS * _SAMPLE_RATE
        and samples - analysis_stop >= _CONTEXT_SECONDS * _SAMPLE_RATE
    )
    warnings: list[str] = []
    if not context_complete:
        warnings.append("insufficient_filter_context")
    if duration < 100.0:
        warnings.append("insufficient_low_frequency_resolution")
    valid = np.isfinite(block.data).all(axis=0)
    for channel in np.flatnonzero(valid):
        valid[channel] = np.ptp(block.data[:, channel].astype(np.float64)) > 0
    output = np.zeros((analysis_stop - analysis_start, channels), dtype=np.float64)
    background_start = max(0, analysis_start - round(200 * _SAMPLE_RATE))
    background_stop = max(0, analysis_start - round(30 * _SAMPLE_RATE))
    background: np.ndarray | None = None
    candidates: list[dict[str, Any]] = []
    if not valid.any():
        warnings.append("no_valid_channels")
        measurements = _measure(output[:, valid], None)
    else:
        if valid.sum() < _MIN_CHANNELS:
            warnings.append("insufficient_spatial_support_channels")
        selected = np.asarray(block.data[:, valid], dtype=np.float64)
        centered = detrend(selected, type="linear", axis=0) if samples > 1 else selected * 0
        if samples > 1:
            sos = butter(4, (0.01, 0.4), btype="bandpass", fs=_SAMPLE_RATE, output="sos")
            filtered = sosfiltfilt(sos, centered, axis=0,
                                  padlen=min(samples - 1, round(100 * _SAMPLE_RATE)))
        else:
            filtered = centered
        target = filtered[analysis_start:analysis_stop]
        output[:, valid] = target
        if background_stop - background_start >= 100 * _SAMPLE_RATE:
            background = filtered[background_start:background_stop]
        else:
            warnings.append("insufficient_background")
        measurements = _measure(target, background)
        if context_complete and duration >= 100.0 and background is not None:
            envelope_power = uniform_filter1d(
                np.abs(hilbert(filtered, axis=0)) ** 2,
                size=round(20 * _SAMPLE_RATE), axis=0, mode="nearest",
            )
            baseline = np.median(envelope_power[background_start:background_stop], axis=0)
            if (baseline > 0).all():
                target_power = envelope_power[analysis_start:analysis_stop]
                support_count = np.sum(target_power >= _POWER_RATIO * baseline, axis=1)
                support = support_count >= max(
                    _MIN_CHANNELS, int(np.ceil(_MIN_CHANNEL_FRACTION * valid.sum()))
                )
                for start, stop in _candidate_ranges(support):
                    measured = _measure(target[start:stop], background)
                    if (stop - start) / _SAMPLE_RATE < 100.0:
                        warnings.append("candidate_insufficient_low_frequency_resolution")
                    peak = start + int(np.argmax(np.mean(target_power[start:stop], axis=1)))
                    score = 0.5 * float(np.clip((measured["snr_db"] or 0) / 12, 0, 1))
                    score += 0.5 * (measured["active_channel_fraction"] or 0)
                    candidates.append({
                        "start_seconds": start / _SAMPLE_RATE,
                        "end_seconds": stop / _SAMPLE_RATE,
                        "peak_seconds": peak / _SAMPLE_RATE,
                        "score": float(np.clip(score, 0, 1)),
                        **{name: measured[name] for name in (
                            "snr_db", "active_channel_fraction", "coherence",
                            "dominant_frequency_hz", "frequency_resolution_hz", "energy_duration_s",
                        )},
                    })
                    if start == 0:
                        warnings.append("candidate_clipped_start")
                    if stop == target.shape[0]:
                        warnings.append("candidate_clipped_end")
            else:
                warnings.append("zero_background_energy")
    measurements.update({
        "algorithm_version": ALGORITHM_VERSION,
        "frequency_band_hz": [0.01, 0.4],
        "sample_rate_hz": _SAMPLE_RATE,
        "input_unit": str(block.metadata.unit),
        "valid_channel_count": int(valid.sum()),
        "valid_channel_fraction": float(valid.mean()) if channels else 0.0,
        "duration_seconds": duration,
        "background_seconds": (background_stop - background_start) / _SAMPLE_RATE
        if background is not None else 0.0,
        "background_start_seconds": (background_start - analysis_start) / _SAMPLE_RATE,
        "background_end_seconds": (background_stop - analysis_start) / _SAMPLE_RATE,
        "filter_context_complete": context_complete,
        "score_semantics": "engineering evidence; not a probability",
    })
    return LowFrequencyAnalysis(output, valid, tuple(candidates), measurements,
                                tuple(dict.fromkeys(warnings)))
