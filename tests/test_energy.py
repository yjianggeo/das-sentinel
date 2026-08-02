import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.energy import _band_metrics, detect_energy

from tests.synthetic import preprocessed_block, sustained_band_limited_event


@pytest.fixture
def profile() -> ProcessingProfile:
    return ProcessingProfile(
        energy_window_seconds=0.5,
        energy_ratio_threshold=3.0,
        min_coincidence=4,
        min_duration_seconds=0.2,
    )


@pytest.fixture
def preprocessed_sustained():
    return sustained_band_limited_event()


def test_energy_finds_sustained_band_limited_event(preprocessed_sustained, profile):
    """Removing narrow-band evidence would fail to distinguish the injected tone."""
    candidates = detect_energy(preprocessed_sustained, profile)

    assert len(candidates) == 1
    assert candidates[0].metrics["spectral_entropy"] < 0.9
    assert candidates[0].end_sample - candidates[0].start_sample >= 200
    assert candidates[0].metrics["peak_ratio"] >= profile.energy_ratio_threshold
    assert 0.0 <= candidates[0].score <= 1.0


def test_energy_band_metrics_match_windowed_overlapped_welch_average():
    """A one-shot periodogram gives different band powers for this brief high-frequency burst."""
    sample_rate_hz = 128.0
    samples = 768
    time = np.arange(samples) / sample_rate_hz
    signal = np.sin(2 * np.pi * 4.0 * time)
    signal[256:320] += 5.0 * np.sin(2 * np.pi * 32.0 * time[256:320])
    bands = ((0.5, 5.0), (5.0, 20.0), (20.0, 60.0))

    band_powers, _ = _band_metrics(signal[:, None].astype(np.float32), sample_rate_hz, bands, np)

    window = np.hanning(256)
    periodograms = []
    for start in range(0, samples - 256 + 1, 128):
        frame = signal[start : start + 256] * window
        periodograms.append(np.abs(np.fft.rfft(frame)) ** 2 / (sample_rate_hz * np.sum(window**2)))
    average = np.mean(periodograms, axis=0)
    frequency = np.fft.rfftfreq(256, d=1.0 / sample_rate_hz)
    expected = np.array([average[(frequency >= low) & (frequency < high)].sum() for low, high in bands])
    expected /= expected.sum()
    np.testing.assert_allclose(band_powers, expected, rtol=1e-5, atol=1e-6)


def test_energy_keeps_event_at_first_complete_rms_window(profile):
    """A second leading-window mask would discard this first eligible sustained event."""
    data = np.full((300, 5), 0.01, dtype=np.float32)
    time = np.arange(80, dtype=np.float32) / 100.0
    data[:80, :4] += (3.0 * np.sin(2 * np.pi * 8.0 * time))[:, None]

    candidates = detect_energy(preprocessed_block(data), profile)

    assert len(candidates) == 1
    assert candidates[0].start_sample == 1_049


def test_energy_ignores_bad_channel_when_selecting_peak(profile):
    """Including a masked channel in peak selection would move this peak to the large spike."""
    data = np.full((800, 5), 0.01, dtype=np.float32)
    time = np.arange(100, 320, dtype=np.float32) / 100.0
    data[100:320, :4] += (3.0 * np.sin(2 * np.pi * 8.0 * time))[:, None]
    data[300:310, 4] = 1_000.0
    block = preprocessed_block(data, bad_channel_mask=np.array([False, False, False, False, True]))

    candidates = detect_energy(block, profile)

    assert len(candidates) == 1
    assert candidates[0].peak_sample < 1_300
    assert candidates[0].metrics["peak_coincidence"] == 4.0
