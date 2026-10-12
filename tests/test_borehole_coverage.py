import numpy as np
import pytest


def signals():
    distance = np.arange(0.0, 3500.0, 25.0)
    seconds = np.arange(3000) / 100
    rng = np.random.default_rng(81)
    background = rng.normal(0, 0.03, (3000, len(distance)))
    pulse = np.sin(2 * np.pi * 7 * seconds) * np.exp(-((seconds - 11) / 2) ** 2)
    wide = background + pulse[:, None]
    local = background.copy()
    local[:, (distance >= 500) & (distance < 1000)] += 100 * pulse[:, None]
    return distance, wide, local


def measure(data, distance, valid=None, rate=100):
    from das_sentinel.borehole_coverage import measure_borehole_coverage

    if valid is None:
        valid = np.ones(len(distance), dtype=bool)
    return measure_borehole_coverage(data, distance, valid, rate)


def test_wide_signal_ranks_above_much_stronger_local_signal():
    # Catches an amplitude-based score that promotes shallow local energy.
    distance, wide, local = signals()
    broad = measure(wide, distance)
    narrow = measure(local, distance)

    assert broad["score"] > narrow["score"] + 30
    assert broad["coverage_fraction"] > 0.95
    assert broad["deep_coverage_fraction"] > 0.95
    assert broad["median_correlation"] > 0.9
    assert 0 <= narrow["score"] <= broad["score"] <= 100


def test_deep_channel_gain_attenuation_preserves_wide_signal_score():
    # Catches comparing absolute channel amplitudes without channel normalization.
    distance, wide, _ = signals()
    gain = np.where(distance < 1000, 50.0, 0.01)
    original = measure(wide, distance)
    attenuated = measure(wide * gain, distance)

    assert attenuated["score"] == pytest.approx(original["score"])
    assert attenuated["deep_coverage_fraction"] == original["deep_coverage_fraction"]


def test_unrelated_shallow_signal_does_not_change_borehole_score():
    # Catches accidentally including physical distances below 500 m.
    distance, wide, _ = signals()
    changed = wide.copy()
    changed[:, distance < 500] *= 1e6

    assert measure(changed, distance) == measure(wide, distance)


def test_missing_channels_reduce_valid_coverage_and_break_continuous_span():
    # Catches treating invalid channels as zero signal or bridging a masked gap.
    distance, wide, _ = signals()
    valid = ~((distance >= 1150) & (distance < 1850))
    changed = wide.copy()
    changed[:, ~valid] = np.nan
    full = measure(wide, distance)
    missing = measure(changed, distance, valid)

    assert missing["score"] is not None
    assert missing["score"] < full["score"]
    assert 0.7 < missing["valid_coverage_fraction"] < 0.8
    assert missing["continuous_span_m"] < 1700
    assert missing["coverage_fraction"] < full["coverage_fraction"]


def test_missing_physical_positions_do_not_bridge_a_distance_gap():
    # Catches using adjacent array indices as contiguous physical channels.
    distance, wide, _ = signals()
    keep = ~((distance >= 1150) & (distance < 1850))
    missing = measure(wide[:, keep], distance[keep])

    assert missing["score"] is not None
    assert missing["valid_coverage_fraction"] < 0.8
    assert missing["continuous_span_m"] < 1700


@pytest.mark.parametrize("value", [0.0, 5.0, 0.1])
def test_constant_waveform_has_unknown_score_instead_of_zero(value):
    # Catches presenting an unmeasurable flat waveform as a confident low score.
    distance, wide, _ = signals()
    result = measure(np.full_like(wide, value), distance)

    assert result["score"] is None
    assert result["coverage_fraction"] is None
    assert result["median_correlation"] is None
    assert result["common_mode_energy_fraction"] is None
    assert result["common_mode_channel_fraction"] is None
    assert result["residual_coverage_score"] is None
    assert result["common_mode_status"] == "unknown"


@pytest.mark.parametrize("case", ["short", "shallow", "no_valid", "one_deep_segment"])
def test_insufficient_spatial_or_temporal_support_is_unknown(case):
    # Catches returning a numeric score without enough deep physical segments.
    distance, wide, _ = signals()
    valid = np.ones(len(distance), dtype=bool)
    if case == "short":
        wide = wide[:100]
    elif case == "shallow":
        wide, distance = wide[:, :20], distance[:20]
    elif case == "no_valid":
        valid[:] = False
    else:
        valid = (distance >= 1000) & (distance < 1500)
    result = measure(wide, distance, valid[:len(distance)])

    assert result["score"] is None
    assert result["deep_coverage_fraction"] is None
    assert isinstance(result["valid_coverage_fraction"], float)
    assert 0 <= result["valid_coverage_fraction"] <= 1
    if case in ("shallow", "no_valid"):
        assert result["valid_coverage_fraction"] == 0
    elif case == "one_deep_segment":
        assert result["valid_coverage_fraction"] == pytest.approx(1 / 6)
    else:
        assert result["valid_coverage_fraction"] == 1


@pytest.mark.parametrize("case", ["data_shape", "distance_shape", "unordered", "distance_nan", "valid_shape", "waveform_nan", "rate_zero", "rate_nan"])
def test_invalid_input_has_unknown_metrics(case):
    # Catches malformed arrays silently receiving apparently valid coverage.
    distance, wide, _ = signals()
    valid = np.ones(len(distance), dtype=bool)
    rate = 100
    if case == "data_shape":
        wide = wide[:, 0]
    elif case == "distance_shape":
        distance = distance[:-1]
    elif case == "unordered":
        distance[60] = distance[59]
    elif case == "distance_nan":
        distance[60] = np.nan
    elif case == "valid_shape":
        valid = valid[:-1]
    elif case == "waveform_nan":
        wide[10, 80] = np.nan
    elif case == "rate_zero":
        rate = 0
    else:
        rate = np.nan
    result = measure(wide, distance, valid, rate)

    assert result["score"] is None
    assert result["coverage_fraction"] is None
    assert result["median_correlation"] is None
    assert isinstance(result["valid_coverage_fraction"], float)
    assert 0 <= result["valid_coverage_fraction"] <= 1


def test_measurement_contract_is_json_safe_and_does_not_mutate_input():
    # Catches changing caller data or leaking NumPy/nonfinite scalars into JSON.
    import json

    from das_sentinel.borehole_coverage import BOREHOLE_COVERAGE_VERSION

    distance, wide, _ = signals()
    before = wide.copy()
    result = measure(wide, distance)

    assert set(result) == {
        "algorithm_version", "score", "coverage_fraction", "deep_coverage_fraction",
        "continuous_span_m", "median_correlation", "valid_coverage_fraction",
        "coverage_score", "propagation_slowness_s_per_m", "apparent_velocity_mps",
        "propagation_residual_s", "propagation_direction", "propagation_confidence",
        "propagation_status", "common_mode_status", "common_mode_penalty",
        "common_mode_energy_fraction", "common_mode_channel_fraction",
        "residual_coverage_score",
    }
    assert result["algorithm_version"] == BOREHOLE_COVERAGE_VERSION
    assert all(v is None or isinstance(v, float) and np.isfinite(v) for k, v in result.items()
               if k not in ("algorithm_version", "propagation_status", "common_mode_status", "propagation_direction"))
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    np.testing.assert_array_equal(wide, before)



def test_zero_background_wide_transient_is_measurable_and_gain_invariant():
    # Catches rejecting informative transients because their median energy is zero.
    distance, wide, _ = signals()
    data = np.zeros_like(wide)
    data[1000:2000] = np.sin(2 * np.pi * 7 * np.arange(1000) / 100)[:, None]
    broad = measure(data, distance)
    scaled = measure(data * np.linspace(0.001, 100, len(distance)), distance)

    assert broad["coverage_score"] > 95
    assert broad["residual_coverage_score"] is None
    assert broad["score"] == broad["coverage_score"]
    assert broad["deep_coverage_fraction"] > 0.95
    assert scaled["score"] == pytest.approx(broad["score"])


@pytest.mark.parametrize("start", [10, 2910])
def test_edge_transient_has_the_same_coverage_as_middle_transient(start):
    # Catches dropping real first/last-second signals from an already filtered cache.
    distance, wide, _ = signals()
    background = np.random.default_rng(82).normal(0, 0.03, wide.shape)
    pulse = np.sin(2 * np.pi * 7 * np.arange(70) / 100)[:, None]
    middle = background.copy()
    middle[1110:1180] += pulse
    edge = background.copy()
    edge[start:start + 70] += pulse
    reference = measure(middle, distance)
    observed = measure(edge, distance)

    assert observed["score"] == pytest.approx(reference["score"], abs=1)
    assert observed["deep_coverage_fraction"] == reference["deep_coverage_fraction"]


def propagation_signals(slowness=0.0004, *, polarity=False):
    distance = np.arange(0.0, 3500.0, 25.0)
    rng = np.random.default_rng(931)
    data = rng.normal(0, 0.015, (3000, len(distance)))
    pulse = np.convolve(rng.normal(size=120), np.ones(3) / 3, mode="same") * np.hanning(120)
    for channel, position in enumerate(distance):
        start = 1100 + round(slowness * (position - 500) * 100)
        sign = -1 if polarity and channel % 2 else 1
        data[start:start + 120, channel] += sign * pulse
    return distance, data


@pytest.mark.parametrize("slowness,direction", [(0.0004, "+"), (-0.0004, "-")])
def test_long_baselines_resolve_native_waveform_propagation(slowness, direction):
    # Catches estimating speed from one-second energies or adjacent short baselines.
    distance, data = propagation_signals(slowness)
    result = measure(data, distance)

    assert result["propagation_status"] == "resolved"
    assert result["propagation_slowness_s_per_m"] == pytest.approx(slowness, abs=0.000025)
    assert result["apparent_velocity_mps"] == pytest.approx(2500, rel=0.08)
    assert result["propagation_direction"] == direction
    assert result["propagation_residual_s"] < 0.015
    assert 0.6 <= result["propagation_confidence"] <= 1
    assert result["score"] == result["coverage_score"]


def test_broadband_polarity_reversals_preserve_supported_propagation():
    # Catches treating an inverted waveform's half-cycle positive peak as its arrival.
    distance, data = propagation_signals(polarity=True)
    result = measure(data, distance)

    assert result["propagation_status"] == "resolved"
    assert result["apparent_velocity_mps"] == pytest.approx(2500, rel=0.08)
    assert result["common_mode_penalty"] == 0


@pytest.mark.parametrize("polarity", [False, True])
def test_synchronous_transient_quantifies_signed_component_without_inventing_velocity(polarity):
    # Catches using zero lag or polarity alignment as the new component measurement.
    distance, data = propagation_signals(0, polarity=polarity)
    result = measure(data, distance)

    assert result["propagation_status"] == "unresolved_zero_lag"
    assert result["apparent_velocity_mps"] is None
    assert result["propagation_slowness_s_per_m"] is None
    if polarity:
        assert result["common_mode_energy_fraction"] < 0.05
        assert result["common_mode_penalty"] == 0
        assert result["score"] == result["coverage_score"]
    else:
        assert result["common_mode_status"] == "component_supported"
        assert result["common_mode_energy_fraction"] > 0.95
        assert result["common_mode_channel_fraction"] == 1
        assert result["residual_coverage_score"] < 0.1 * result["coverage_score"]
        assert 0 < result["common_mode_penalty"] <= 0.5 * result["coverage_score"]


def persistent_common_signal():
    distance = np.arange(0.0, 3500.0, 25.0)
    rng = np.random.default_rng(940)
    shared = rng.choice([-1.0, 1.0], size=3000)
    # Repeated amplitude changes span the complete window, rather than one onset.
    shared *= np.where(np.arange(3000) // 100 % 2, 0.45, 1.5)
    shared *= 1 + 0.02 * np.exp(-((np.arange(3000) / 100 - 15) / 6) ** 2)
    data = shared[:, None] + rng.normal(0, 0.005, (3000, len(distance)))
    return distance, data


def test_supported_persistent_common_signal_only_conservatively_reduces_ranking():
    # Catches either no mitigation or overwriting the independently measured coverage.
    distance, data = persistent_common_signal()
    result = measure(data, distance)

    assert result["coverage_score"] > 90
    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_energy_fraction"] > 0.99
    assert result["common_mode_channel_fraction"] == 1
    assert result["residual_coverage_score"] < 0.1 * result["coverage_score"]
    assert 0.4 * result["coverage_score"] < result["common_mode_penalty"] <= 0.5 * result["coverage_score"]
    assert result["score"] == pytest.approx(result["coverage_score"] - result["common_mode_penalty"])
    assert result["apparent_velocity_mps"] is None


def test_common_background_with_an_independent_pulse_is_not_penalized():
    # Catches letting the persistent shared component erase a real independent onset.
    distance, common = persistent_common_signal()
    _, transient = propagation_signals()
    result = measure(common + 8 * transient, distance)

    assert result["propagation_status"] == "resolved"
    assert result["common_mode_energy_fraction"] > 0.5
    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


def test_unselected_native_channel_pulse_protects_against_common_mode_penalty():
    # Catches missing an independent pulse between the fixed propagation anchors.
    distance, data = persistent_common_signal()
    channel = int(np.flatnonzero(distance == 1700)[0])
    pulse = 3 * np.sin(2 * np.pi * 7 * np.arange(100) / 100) * np.hanning(100)
    data[1200:1300, channel] += pulse
    result = measure(data, distance)

    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


def test_small_missing_distance_gap_retains_component_measurement():
    # Catches coupling component quantification to propagation's complete geometry gate.
    distance, data = persistent_common_signal()
    valid = ~((distance >= 1200) & (distance < 1400))
    result = measure(data, distance, valid)

    assert result["propagation_status"] == "insufficient_support"
    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_energy_fraction"] > 0.99
    assert 0 <= result["common_mode_penalty"] <= 0.5 * result["coverage_score"]


def test_native_subsecond_transient_is_protected_when_one_second_energy_is_unchanged():
    # Catches relying on coarse energy ratios instead of all native-channel quiet flanks.
    distance, data = persistent_common_signal()
    channel = int(np.flatnonzero(distance == 1700)[0])
    original = data[1200:1300, channel].copy()
    data[1200:1300, channel] = 0
    data[1200:1220, channel] = original[:20] * np.sqrt(np.sum(original ** 2) / np.sum(original[:20] ** 2))
    result = measure(data, distance)

    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


def test_synchronous_transient_below_later_background_still_quantifies_common_component():
    # Catches using a separate zero-lag onset as evidence against a measured component.
    distance = np.arange(0.0, 3500.0, 25.0)
    time = np.arange(3000) / 100
    rng = np.random.default_rng(940)
    shared = rng.choice([-1.0, 1.0], size=3000)
    shared *= np.where(time < 10, 0.45, np.where(time < 20, 1.5, np.sqrt(1.2)))
    shared *= 1 + 0.02 * np.exp(-((time - 15) / 4) ** 2)
    data = shared[:, None] + rng.normal(0, 0.001, (3000, len(distance)))
    pulse = 3 * np.sin(2 * np.pi * 7 * np.arange(100) / 100) * np.hanning(100)
    data[500:600] += pulse[:, None]
    result = measure(data, distance)

    assert result["common_mode_energy_fraction"] > 0.99
    assert result["common_mode_status"] == "component_supported"
    assert 0 <= result["common_mode_penalty"] <= 0.5 * result["coverage_score"]


@pytest.mark.parametrize("start", [0, 2900])
def test_hidden_edge_transient_retains_component_measurement(start):
    # Catches propagation's edge veto also erasing independent common-mode metrics.
    distance = np.arange(0.0, 3500.0, 25.0)
    time = np.arange(3000) / 100
    rng = np.random.default_rng(111)
    shared = rng.choice([-1.0, 1.0], size=3000)
    shared *= 0.45 * np.exp(0.08 * np.minimum(time, 30 - time))
    data = shared[:, None] + rng.normal(0, 0.001, (3000, len(distance)))
    pulse = 2 * np.sin(2 * np.pi * 7 * np.arange(100) / 100) * np.hanning(100)
    data[start:start + 100] += pulse[:, None]
    result = measure(data, distance)

    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_energy_fraction"] > 0.99
    assert 0 <= result["common_mode_penalty"] <= 0.5 * result["coverage_score"]


def test_one_missing_native_position_makes_propagation_diagnostics_unknown():
    # Catches estimating propagation across a hole hidden by a permissive metre threshold.
    distance, data = persistent_common_signal()
    valid = distance != 1200
    result = measure(data, distance, valid)

    assert result["propagation_status"] == "insufficient_support"
    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_energy_fraction"] > 0.99
    assert 0 <= result["common_mode_penalty"] <= 0.5 * result["coverage_score"]


def test_coarse_native_sampling_keeps_coverage_without_inventing_propagation():
    # Catches a valid v1 input crashing when a two-second lag window has too few samples.
    distance = np.arange(0.0, 3500.0, 25.0)
    data = np.random.default_rng(27).normal(0, 0.001, (8, len(distance)))
    data[4] += 1
    result = measure(data, distance, rate=1)

    assert result["coverage_score"] > 90
    assert result["propagation_status"] == "insufficient_support"
    assert result["apparent_velocity_mps"] is None
    assert result["common_mode_energy_fraction"] is None
    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


def test_periodic_lag_alias_does_not_erase_signed_component_measurement():
    # Catches conflating unknown periodic velocity with unknown signed projection energy.
    distance, _, _ = signals()
    time = np.arange(3000) / 100
    pulse = np.sin(2 * np.pi * 10 * time) * (1 + 0.6 * np.exp(-((time - 15) / 7) ** 2))
    data = np.tile(pulse[:, None], (1, len(distance)))
    result = measure(data, distance)

    assert result["propagation_status"] == "ambiguous_periodic"
    assert result["apparent_velocity_mps"] is None
    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_energy_fraction"] == pytest.approx(1)
    assert result["common_mode_channel_fraction"] == 1
    assert result["residual_coverage_score"] is None
    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


@pytest.mark.parametrize("case", ["edge", "gap", "independent", "mixed_direction"])
def test_unreliable_or_incomplete_evidence_cannot_reduce_ranking(case):
    # Catches a quality failure being interpreted as evidence that a window is noise.
    distance, data = propagation_signals()
    valid = np.ones(len(distance), dtype=bool)
    if case == "edge":
        data = np.roll(data, -1080, axis=0)
    elif case == "gap":
        valid[(distance >= 1000) & (distance < 2200)] = False
    elif case == "independent":
        data = np.random.default_rng(19).normal(size=data.shape)
    else:
        _, reverse = propagation_signals(-0.0004)
        data[:, distance >= 2000] = reverse[:, distance >= 2000]
    result = measure(data, distance, valid)

    assert result["propagation_status"] != "resolved"
    assert result["propagation_status"] in {
        "resolved", "unresolved_zero_lag", "low_correlation", "ambiguous_periodic",
        "insufficient_support", "edge_or_boundary",
    }
    assert result["apparent_velocity_mps"] is None
    assert result["common_mode_status"] == "not_supported"
    assert result["common_mode_channel_fraction"] < 0.6
    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


def test_signed_projection_does_not_flip_minority_polarity_or_modify_input():
    # Catches aligning opposite signs before the median or projecting negative correlations.
    distance, data = persistent_common_signal()
    selected = np.flatnonzero(distance >= 500)
    opposite = selected[-80:].reshape(4, 20)[:, :11].ravel()
    data[:, opposite] *= -1
    before = data.copy()
    result = measure(data, distance)

    assert result["common_mode_energy_fraction"] == pytest.approx(76 / 120, abs=0.01)
    assert result["common_mode_channel_fraction"] == pytest.approx(76 / 120)
    assert result["residual_coverage_score"] > 15
    assert result["common_mode_penalty"] <= 0.5 * result["coverage_score"]
    np.testing.assert_array_equal(data, before)


def test_common_component_with_broad_independent_residual_keeps_ranking():
    # Catches reducing candidates whose remaining native-channel signal still covers the borehole.
    distance, data = persistent_common_signal()
    rng = np.random.default_rng(918)
    independent = rng.normal(size=data.shape)
    independent *= np.exp(-((np.arange(3000) / 100 - 10) / 1.5) ** 2)[:, None]
    result = measure(data + independent, distance)

    assert result["common_mode_energy_fraction"] > 0.5
    assert result["common_mode_channel_fraction"] > 0.9
    assert result["residual_coverage_score"] > 0.9 * result["coverage_score"]
    assert result["common_mode_penalty"] == 0
    assert result["score"] == result["coverage_score"]


def test_common_component_measurement_survives_a_few_constant_channels():
    # Catches losing valid common-mode measurements because a few native channels are flat.
    distance, data = persistent_common_signal()
    data[:, (distance >= 1700) & (distance < 1775)] = 2
    result = measure(data, distance)

    assert result["propagation_status"] == "insufficient_support"
    assert result["common_mode_status"] == "component_supported"
    assert result["common_mode_energy_fraction"] > 0.99
    assert result["common_mode_channel_fraction"] == 1
    assert 0 <= result["common_mode_penalty"] <= 0.5 * result["coverage_score"]
