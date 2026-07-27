import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.coherence import detect_coherence
from das_sentinel.signal.backend import select_backend

from tests.synthetic import preprocessed_block


@pytest.fixture
def profile() -> ProcessingProfile:
    return ProcessingProfile(
        coherence_threshold=0.08,
        coherence_window_seconds=2.0,
        min_coincidence=4,
        slowness_max_s_per_m=0.001,
        slowness_points=41,
    )


@pytest.fixture
def cpu_backend():
    return select_backend("cpu")


@pytest.fixture
def moving_wavefront():
    """Using channel indices instead of physical distances selects the wrong moveout."""
    sample_rate_hz = 100.0
    distances = np.array([0.0, 4.0, 11.0, 23.0, 38.0, 60.0], dtype=np.float32)
    data = np.random.default_rng(22).normal(0.0, 0.02, size=(600, len(distances))).astype(np.float32)
    pulse = 3.0 * np.hanning(11).astype(np.float32)
    for channel, distance in enumerate(distances):
        arrival = 240 + round(2.5e-4 * distance * sample_rate_hz)
        data[arrival : arrival + len(pulse), channel] += pulse
    block = preprocessed_block(data, sample_offset=4_000, sample_rate_hz=sample_rate_hz)
    return block.__class__(
        **{**block.__dict__, "channel_distance_m": distances}
    )


@pytest.fixture
def common_mode_pulse():
    data = np.random.default_rng(23).normal(0.0, 0.02, size=(600, 6)).astype(np.float32)
    data[240:251] += (3.0 * np.hanning(11).astype(np.float32))[:, None]
    return preprocessed_block(data, sample_rate_hz=100.0)


def test_coherence_recovers_moving_wavefront(moving_wavefront, profile, cpu_backend):
    """Ignoring physical moveout leaves this nonuniform-array pulse incoherent."""
    candidate = detect_coherence(moving_wavefront, profile, cpu_backend)[0]

    assert candidate.metrics["semblance"] > 0.1
    assert candidate.metrics["slowness_s_per_m"] == pytest.approx(2.5e-4, abs=5e-5)
    assert candidate.metrics["apparent_velocity_mps"] == pytest.approx(4_000.0, rel=0.2)
    assert candidate.metrics["direction"] == 1.0
    assert candidate.peak_sample == 4_245


def test_coherence_flags_zero_slowness_common_mode(common_mode_pulse, profile, cpu_backend):
    """Treating common-mode energy as travelling evidence reports a false direction."""
    candidates = detect_coherence(common_mode_pulse, profile, cpu_backend)

    assert candidates == [] or "zero_slowness_common_mode" in candidates[0].quality_flags


def test_coherence_excludes_bad_channel_from_moveout_normalization(profile, cpu_backend):
    """Including a masked high-amplitude trace changes the selected slowest moveout."""
    data = np.random.default_rng(24).normal(0.0, 0.02, size=(600, 6)).astype(np.float32)
    distances = np.array([0.0, 4.0, 11.0, 23.0, 38.0, 60.0], dtype=np.float32)
    pulse = 3.0 * np.hanning(11).astype(np.float32)
    for channel, distance in enumerate(distances[:5]):
        arrival = 240 + round(2.5e-4 * distance * 100.0)
        data[arrival : arrival + len(pulse), channel] += pulse
    data[410:421, 5] += 100.0 * np.hanning(11).astype(np.float32)
    block = preprocessed_block(
        data,
        sample_rate_hz=100.0,
        bad_channel_mask=np.array([False, False, False, False, False, True]),
    )
    block = block.__class__(**{**block.__dict__, "channel_distance_m": distances})

    candidate = detect_coherence(block, profile, cpu_backend)[0]

    assert candidate.metrics["slowness_s_per_m"] == pytest.approx(2.5e-4, abs=5e-5)
    assert candidate.channel_stop == 5
