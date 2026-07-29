import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.stalta import detect_stalta

from tests.synthetic import (
    multichannel_impulse,
    preprocessed_block,
    single_channel_impulse,
)


@pytest.fixture
def profile() -> ProcessingProfile:
    return ProcessingProfile(
        sta_seconds=0.05,
        lta_seconds=0.5,
        stalta_threshold=3.0,
        min_coincidence=4,
        min_duration_seconds=0.02,
    )


@pytest.fixture
def preprocessed_impulse():
    return multichannel_impulse()


@pytest.fixture
def preprocessed_single_channel():
    return single_channel_impulse()


def test_stalta_finds_multichannel_impulse(preprocessed_impulse, profile):
    """Dropping channel coincidence must not accept a single-channel transient."""
    candidates = detect_stalta(preprocessed_impulse, profile)

    assert len(candidates) == 1
    assert 490 + preprocessed_impulse.sample_offset <= candidates[0].peak_sample <= 510 + preprocessed_impulse.sample_offset
    assert candidates[0].metrics["peak_ratio"] >= profile.stalta_threshold
    assert 0.0 <= candidates[0].score <= 1.0


def test_stalta_rejects_single_channel_impulse(preprocessed_single_channel, profile):
    """Removing coincidence enforcement would make this one-channel event appear valid."""
    assert detect_stalta(preprocessed_single_channel, profile) == []


def test_stalta_keeps_event_at_first_complete_lta_window(profile):
    """A second leading-window mask would discard this first eligible transient."""
    data = np.zeros((180, 5), dtype=np.float32)
    data[42:50, :4] = 10.0

    candidates = detect_stalta(preprocessed_block(data), profile)

    assert len(candidates) == 1
    assert candidates[0].start_sample == 1_049


def test_stalta_ignores_bad_channel_when_selecting_peak(profile):
    """Including a masked channel in peak selection would move this peak to sample 110."""
    data = np.zeros((300, 5), dtype=np.float32)
    data[100:108, :4] = 10.0
    data[110, 4] = 1_000.0
    block = preprocessed_block(data, bad_channel_mask=np.array([False, False, False, False, True]))

    candidates = detect_stalta(block, profile)

    assert len(candidates) == 1
    assert candidates[0].peak_sample < 1_110
    assert candidates[0].metrics["peak_coincidence"] == 4.0


def test_default_twenty_second_block_detects_event_after_complete_lta():
    sample_rate_hz = 200.0
    profile = ProcessingProfile()
    data = np.zeros((round(profile.block_seconds * sample_rate_hz), 8), dtype=np.float32)
    event_start = round(15.0 * sample_rate_hz)
    event_samples = round(0.3 * sample_rate_hz)
    burst = np.hanning(event_samples).astype(np.float32) * 20.0
    data[event_start : event_start + event_samples, :4] = burst[:, None]

    candidates = detect_stalta(
        preprocessed_block(data, sample_offset=0, sample_rate_hz=sample_rate_hz),
        profile,
    )

    assert len(candidates) == 1
    assert event_start <= candidates[0].peak_sample < event_start + event_samples
