from dataclasses import replace

import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.detectors.trajectory import (
    Template,
    detect_trajectories,
    score_template,
)

from tests.synthetic import preprocessed_block


@pytest.fixture
def profile() -> ProcessingProfile:
    return ProcessingProfile(
        trajectory_velocity_min_mps=4.0,
        trajectory_velocity_max_mps=40.0,
        trajectory_threshold=0.65,
        min_coincidence=4,
    )


@pytest.fixture
def vehicle_diagonal():
    sample_rate_hz = 100.0
    distances = np.cumsum(np.array([0.0, 1.1, 1.8, 0.7, 2.4, 1.0, 1.5, 0.9], dtype=np.float32))
    data = np.random.default_rng(31).normal(0.0, 0.01, size=(900, distances.size)).astype(np.float32)
    pulse = np.hanning(11).astype(np.float32) * 5.0
    for channel, distance in enumerate(distances):
        center = 260 + round(distance / 18.0 * sample_rate_hz)
        data[center - 5 : center + 6, channel] += pulse
    return replace(
        preprocessed_block(data, sample_rate_hz=sample_rate_hz), channel_distance_m=distances
    )


@pytest.fixture
def blast_block():
    data = np.random.default_rng(32).normal(0.0, 0.01, size=(800, 8)).astype(np.float32)
    pulse = np.hanning(13).astype(np.float32) * 4.0
    for center in (180, 350, 540):
        data[center - 6 : center + 7] += pulse[:, None]
    return preprocessed_block(data)


@pytest.fixture
def unrelated_block():
    data = np.random.default_rng(33).normal(0.0, 0.02, size=(800, 8)).astype(np.float32)
    data[120:500] += np.sin(np.arange(380, dtype=np.float32)[:, None] * 0.91)
    return preprocessed_block(data)


def test_trajectory_estimates_vehicle_speed_from_real_channel_distances(vehicle_diagonal, profile):
    candidate = detect_trajectories(vehicle_diagonal, profile)[0]

    assert candidate.detector == "trajectory"
    assert candidate.metrics["apparent_velocity_mps"] == pytest.approx(18.0, rel=0.2)
    assert candidate.metrics["direction"] == 1.0
    assert candidate.metrics["trajectory_strength"] >= profile.trajectory_threshold


def test_trajectory_rejects_horizontal_common_mode(profile):
    data = np.zeros((700, 8), dtype=np.float32)
    data[300:311] = np.hanning(11).astype(np.float32)[:, None] * 5.0

    assert detect_trajectories(preprocessed_block(data), profile) == []


def test_trajectory_rejects_common_mode_wider_than_maximum_channel_delay(profile):
    data = np.zeros((900, 8), dtype=np.float32)
    data[250:500] = np.hanning(250).astype(np.float32)[:, None] * 5.0

    assert detect_trajectories(preprocessed_block(data), profile) == []


def test_trajectory_ignores_bad_channels(vehicle_diagonal, profile):
    bad_channel_mask = np.zeros(vehicle_diagonal.data.shape[1], dtype=bool)
    bad_channel_mask[-1] = True
    corrupted = vehicle_diagonal.data.copy()
    corrupted[:, -1] = 100.0
    block = replace(vehicle_diagonal, data=corrupted, bad_channel_mask=bad_channel_mask)

    candidate = detect_trajectories(block, profile)[0]
    assert candidate.channel_stop == vehicle_diagonal.data.shape[1] - 1


def test_template_score_prefers_matching_pulse_train(blast_block, unrelated_block):
    template = Template.from_event(blast_block, event_id="reviewed-blast", site_id="site-a", profile_version="p1")

    assert score_template(blast_block, template) > 0.8
    assert score_template(unrelated_block, template) < 0.4
    assert template.sample_rate_hz == blast_block.sample_rate_hz
    assert template.site_id == "site-a"
    assert template.profile_version == "p1"


def test_template_rejects_mismatched_metadata_and_bad_channels(blast_block):
    template = Template.from_event(blast_block, event_id="reviewed-blast", site_id="site-a", profile_version="p1")
    object.__setattr__(blast_block, "site_id", "site-b")
    object.__setattr__(blast_block, "profile_version", "p1")
    assert score_template(blast_block, template) == 0.0

    object.__setattr__(blast_block, "site_id", "site-a")
    object.__setattr__(blast_block, "profile_version", "p1")
    bad = np.zeros(blast_block.data.shape[1], dtype=bool)
    bad[0] = True
    masked = replace(blast_block, bad_channel_mask=bad)
    assert score_template(masked, template) > 0.8


@pytest.mark.parametrize("event_id", ("", "   "))
@pytest.mark.parametrize("sample_rate_hz", (0.0, float("nan")))
def test_template_creation_rejects_blank_event_id_or_invalid_sample_rate(blast_block, event_id, sample_rate_hz):
    invalid_block = replace(blast_block, sample_rate_hz=sample_rate_hz)

    with pytest.raises(ValueError):
        Template.from_event(invalid_block, event_id=event_id)


def test_template_score_rejects_manual_empty_event_id_or_invalid_sample_rate(blast_block):
    template = Template.from_event(blast_block, event_id="reviewed-blast")

    assert score_template(blast_block, replace(template, source_event_id="")) == 0.0
    assert score_template(blast_block, replace(template, sample_rate_hz=float("nan"))) == 0.0
