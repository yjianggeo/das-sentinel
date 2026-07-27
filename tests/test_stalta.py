import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.stalta import detect_stalta

from tests.synthetic import multichannel_impulse, single_channel_impulse


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
