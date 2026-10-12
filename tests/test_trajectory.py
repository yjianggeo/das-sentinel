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


def legacy_line_energy(energy, distances_m, sample_rate_hz, velocity_mps):
    """Frozen scalar search: preserves the pre-optimization reduction and ties."""
    delays = np.rint((distances_m - distances_m[0]) / velocity_mps * sample_rate_hz).astype(int)
    best_energy, best_start, best_indices = -1.0, 0, np.empty(0, dtype=int)
    for start in range(energy.shape[0]):
        indices = start + delays
        if ((indices >= 0) & (indices < energy.shape[0])).all():
            value = float(energy[indices, np.arange(energy.shape[1])].sum())
            if value > best_energy:
                best_energy, best_start, best_indices = value, start, indices
    return best_energy, best_start, best_indices


@pytest.mark.parametrize("distance", ([0., 1., 3.5, 8.], [-4., -2., 1., 7.], [3., -2., 7., 0.]))
@pytest.mark.parametrize("velocity", [-1., 1., -80., 80.])
@pytest.mark.parametrize("count", [0, 1, 35, 600])
def test_line_energy_exactly_matches_scalar_search(distance, velocity, count):
    from das_sentinel.detectors.trajectory import _line_energy

    rng = np.random.default_rng(9)
    energy = rng.random((count, 4))
    if count > 2:
        energy[1, 0] = np.nan
    expected = legacy_line_energy(energy, np.array(distance), 100., velocity)
    actual = _line_energy(energy, np.array(distance), 100., velocity)
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])


def test_line_energy_ties_keep_earliest_valid_signed_start():
    from das_sentinel.detectors.trajectory import _line_energy

    result = _line_energy(np.ones((600, 3)), np.array([0., -2., 3.]), 1., 1.)
    assert result[:2] == (3., 2)
    np.testing.assert_array_equal(result[2], [2, 0, 5])


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("channels", [1, 7, 63, 64])
def test_line_energy_preserves_reduction_order_across_chunks(dtype, channels):
    from das_sentinel.detectors.trajectory import _line_energy

    rng = np.random.default_rng(81)
    energy = (rng.random((3000, channels)) * np.geomspace(1e-10, 1e10, channels)).astype(dtype)
    distance = np.cumsum(rng.uniform(.5, 2., channels)) - 10
    expected = legacy_line_energy(energy, distance, 100., -18.)
    actual = _line_energy(energy, distance, 100., -18.)
    assert actual[:2] == expected[:2]
    np.testing.assert_array_equal(actual[2], expected[2])


def test_line_energy_python_work_is_bounded_by_chunks():
    import sys

    from das_sentinel.detectors.trajectory import _line_energy

    line_events = 0

    def trace(frame, event, arg):
        nonlocal line_events
        if frame.f_code is _line_energy.__code__ and event == "line":
            line_events += 1
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        result = _line_energy(np.ones((3000, 8)), np.arange(8.), 100., 10.)
    finally:
        sys.settrace(previous)
    assert result[0] == 8.
    assert line_events < 500, "per-start Python work returned to trajectory scoring"


def test_trajectory_candidates_exactly_match_scalar_search(vehicle_diagonal, profile, monkeypatch):
    from dataclasses import asdict

    import das_sentinel.detectors.trajectory as module

    actual = [asdict(value) for value in module.detect_trajectories(vehicle_diagonal, profile)]
    monkeypatch.setattr(module, "_line_energy", legacy_line_energy)
    expected = [asdict(value) for value in module.detect_trajectories(vehicle_diagonal, profile)]
    assert actual == expected


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


def test_template_constructor_rejects_manual_empty_event_id_or_invalid_sample_rate(blast_block):
    template = Template.from_event(blast_block, event_id="reviewed-blast")

    with pytest.raises(ValueError):
        replace(template, source_event_id="")
    with pytest.raises(ValueError):
        replace(template, sample_rate_hz=float("nan"))


def test_direct_template_normalizes_scaled_arrays_without_changing_ncc(blast_block):
    standard = Template.from_event(blast_block, event_id="reviewed-blast")
    scaled = Template(
        waveform=standard.waveform * 7.0,
        spectrum=standard.spectrum * 11.0,
        site_id=standard.site_id,
        profile_version=standard.profile_version,
        source_event_id=standard.source_event_id,
        sample_rate_hz=standard.sample_rate_hz,
    )

    assert np.linalg.norm(scaled.waveform) == pytest.approx(1.0)
    assert np.linalg.norm(scaled.spectrum) == pytest.approx(1.0)
    assert not np.shares_memory(scaled.waveform, standard.waveform)
    assert score_template(blast_block, scaled) == pytest.approx(score_template(blast_block, standard))


@pytest.mark.parametrize(
    ("waveform", "spectrum"),
    (
        (np.zeros(3), np.ones(2)),
        (np.array([1.0, float("nan")]), np.ones(2)),
        (np.ones(3), np.zeros(2)),
        (np.ones(3), np.array([1.0, float("inf")])),
    ),
)
def test_direct_template_rejects_zero_or_nonfinite_arrays(waveform, spectrum):
    with pytest.raises(ValueError):
        Template(
            waveform=waveform,
            spectrum=spectrum,
            site_id="site-a",
            profile_version="p1",
            source_event_id="event-1",
            sample_rate_hz=100.0,
        )
