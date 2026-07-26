import pytest
from das_sentinel.config import ProcessingProfile
from pydantic import ValidationError


def test_profile_has_approved_block_defaults():
    profile = ProcessingProfile()

    assert profile.block_seconds == 20.0
    assert profile.overlap_seconds == 5.0
    assert profile.device == "auto"


def test_profile_rejects_overlap_not_smaller_than_block():
    with pytest.raises(ValidationError):
        ProcessingProfile(block_seconds=5.0, overlap_seconds=5.0)


@pytest.mark.parametrize(
    ("low_field", "high_field", "low", "high"),
    [
        ("stalta_freq_low_hz", "stalta_freq_high_hz", 0.0, 40.0),
        ("stalta_freq_low_hz", "stalta_freq_high_hz", 40.0, 40.0),
        ("coherence_freq_low_hz", "coherence_freq_high_hz", 0.0, 40.0),
        ("coherence_freq_low_hz", "coherence_freq_high_hz", 40.0, 40.0),
        ("trajectory_freq_low_hz", "trajectory_freq_high_hz", 0.0, 30.0),
        ("trajectory_freq_low_hz", "trajectory_freq_high_hz", 30.0, 30.0),
    ],
)
def test_profile_rejects_invalid_frequency_order(low_field, high_field, low, high):
    with pytest.raises(ValidationError):
        ProcessingProfile(**{low_field: low, high_field: high})


@pytest.mark.parametrize("window_field", ["energy_window_seconds", "coherence_window_seconds"])
def test_profile_rejects_nonpositive_sampling_windows(window_field):
    with pytest.raises(ValidationError):
        ProcessingProfile(**{window_field: 0.0})
