import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.energy import detect_energy

from tests.synthetic import sustained_band_limited_event


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
