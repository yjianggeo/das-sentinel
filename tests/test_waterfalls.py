import pytest
import numpy as np

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.waterfalls import (
    WaterfallFilter,
    channel_slice_for_distance,
    resolve_waterfall_filter,
    usable_channel_indices,
)


def _candidate(detector: str, *, score: float, peak_sample: int) -> Candidate:
    return Candidate(
        detector=detector,
        start_sample=peak_sample - 1,
        end_sample=peak_sample + 1,
        peak_sample=peak_sample,
        channel_start=0,
        channel_stop=1,
        score=score,
    )


def test_resolves_primary_detector_band_from_run_profile():
    """Using default STA/LTA frequencies instead of the run profile must fail."""
    profile = ProcessingProfile(stalta_freq_low_hz=3.5, stalta_freq_high_hz=27.25)
    evidence = (
        _candidate("coherence", score=0.70, peak_sample=80),
        _candidate("stalta:coincident", score=0.91, peak_sample=90),
    )

    result = resolve_waterfall_filter(evidence, profile)

    assert result == WaterfallFilter("stalta", 3.5, 27.25)


def test_energy_filter_spans_the_configured_energy_bank():
    """Selecting an individual energy band instead of the configured bank must fail."""
    profile = ProcessingProfile(energy_bands_hz=((0.75, 4.0), (6.0, 33.0)))

    result = resolve_waterfall_filter(
        (_candidate("energy", score=0.8, peak_sample=20),), profile
    )

    assert result == WaterfallFilter("energy", 0.75, 33.0)


def test_primary_evidence_ties_use_peak_then_detector_and_remove_suffix():
    """Changing the stable primary-evidence ordering must fail."""
    profile = ProcessingProfile(
        coherence_freq_low_hz=4.0,
        coherence_freq_high_hz=12.0,
    )
    evidence = (
        _candidate("stalta", score=0.8, peak_sample=10),
        _candidate("coherence:network", score=0.8, peak_sample=9),
        _candidate("energy", score=0.8, peak_sample=9),
    )

    result = resolve_waterfall_filter(evidence, profile)

    assert result == WaterfallFilter("coherence", 4.0, 12.0)


def test_rejects_empty_evidence_and_unknown_detector():
    """Silently choosing a fallback filter must fail."""
    profile = ProcessingProfile()

    with pytest.raises(ValueError, match="evidence"):
        resolve_waterfall_filter((), profile)
    with pytest.raises(ValueError, match="unknown"):
        resolve_waterfall_filter(
            (_candidate("unknown", score=0.8, peak_sample=20),), profile
        )


def test_distance_selection_is_inclusive_and_removes_bad_channels():
    """Excluding either boundary or retaining a bad channel must fail."""
    distances = np.array([8.0, 12.0, 16.0, 20.0, 24.0])
    bad = np.array([False, False, True, False, False])

    assert channel_slice_for_distance(distances, 12.0, 20.0) == slice(1, 4)
    np.testing.assert_array_equal(
        usable_channel_indices(distances, bad, 12.0, 20.0),
        np.array([1, 3]),
    )


def test_distance_selection_uses_source_slice_for_non_monotonic_distances():
    """Sorting distances or dropping an intervening source channel must fail."""
    distances = np.array([8.0, 12.0, 100.0, 20.0, 24.0])
    bad = np.array([False, False, False, False, False])

    assert channel_slice_for_distance(distances, 12.0, 20.0) == slice(1, 4)
    np.testing.assert_array_equal(
        usable_channel_indices(distances, bad, 12.0, 20.0),
        np.array([1, 3]),
    )


def test_distance_selection_returns_all_channels_for_a_fully_in_range_request():
    """Trimming a request that contains every channel must fail."""
    distances = np.array([8.0, 12.0, 16.0, 20.0, 24.0])
    bad = np.zeros(5, dtype=bool)

    assert channel_slice_for_distance(distances, 8.0, 24.0) == slice(0, 5)
    np.testing.assert_array_equal(
        usable_channel_indices(distances, bad, 8.0, 24.0),
        np.array([0, 1, 2, 3, 4]),
    )


def test_distance_selection_rejects_invalid_or_empty_inputs():
    """Accepting invalid bounds or an empty usable selection must fail."""
    distances = np.array([8.0, 12.0, 16.0])

    with pytest.raises(ValueError, match="finite"):
        channel_slice_for_distance(np.array([8.0, np.nan]), 8.0, 12.0)
    with pytest.raises(ValueError, match="start_m"):
        channel_slice_for_distance(distances, 16.0, 12.0)
    with pytest.raises(ValueError, match="selected"):
        channel_slice_for_distance(distances, 20.0, 24.0)
    with pytest.raises(ValueError, match="usable"):
        usable_channel_indices(
            distances,
            np.array([False, True, False]),
            12.0,
            12.0,
        )
