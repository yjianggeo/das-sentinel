from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate, DasBlock, DasMetadata
from das_sentinel.screening import (
    ScreeningOptions,
    _ratio_candidates,
    _suppress_duplicates,
    fuse_screening_candidates,
    screen_block,
)
from das_sentinel.signal.backend import select_backend
from das_sentinel.signal.preprocess import PreprocessedBlock

CPU = select_backend("cpu")


def _block(
    data: np.ndarray,
    *,
    sample_rate_hz: float = 100.0,
    spacing_m: float = 5.0,
    sample_offset: int = 10_000,
    distances_m: np.ndarray | None = None,
) -> DasBlock:
    return DasBlock(
        data=np.asarray(data, dtype=np.float32),
        metadata=DasMetadata(
            source_path="synthetic.h5",
            start_time_utc=datetime(2026, 9, 10, tzinfo=UTC),
            sample_rate_hz=sample_rate_hz,
            channel_distance_m=(
                np.arange(data.shape[1], dtype=np.float32) * spacing_m
                if distances_m is None
                else np.asarray(distances_m, dtype=np.float32)
            ),
            unit="strain_rate",
        ),
        sample_offset=sample_offset,
    )


def _profile(**updates: object) -> ProcessingProfile:
    values: dict[str, object] = {
        "common_mode_removal": False,
        "stalta_freq_low_hz": 2.0,
        "stalta_freq_high_hz": 35.0,
        "sta_seconds": 0.05,
        "lta_seconds": 0.5,
        "stalta_threshold": 3.0,
        "stalta_strong_ratio": 6.0,
        "min_coincidence": 4,
        "min_duration_seconds": 0.01,
        "energy_bands_hz": ((2.0, 12.0), (12.0, 35.0)),
        "energy_window_seconds": 0.1,
        "energy_ratio_threshold": 4.0,
        "coherence_freq_low_hz": 2.0,
        "coherence_freq_high_hz": 35.0,
        "coherence_window_seconds": 0.2,
        "coherence_threshold": 0.08,
        "slowness_max_s_per_m": 0.01,
        "slowness_points": 21,
        "trajectory_freq_low_hz": 2.0,
        "trajectory_freq_high_hz": 25.0,
        "trajectory_velocity_min_mps": 5.0,
        "trajectory_velocity_max_mps": 40.0,
        "trajectory_threshold": 0.60,
        "merge_gap_seconds": 0.2,
    }
    values.update(updates)
    return ProcessingProfile(**values)


def _noise(samples: int, channels: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.02, (samples, channels)).astype(np.float32)


def _pulse(length: int, amplitude: float) -> np.ndarray:
    return np.hanning(length).astype(np.float32) * amplitude


def _amplitude_options(**updates: object) -> ScreeningOptions:
    options = ScreeningOptions(
        spatial_window_m=50.0,
        spatial_overlap=0.5,
        off_ratio=1.5,
        release_gap_seconds=0.08,
        short_sta_seconds=0.03,
        short_lta_seconds=0.25,
        coincidence_tolerance_seconds=0.02,
        common_mode_removal=False,
        enable_coherence=False,
        enable_trajectory=False,
    )
    return replace(options, **updates)


def _candidate(
    detector: str,
    score: float,
    *,
    start_sample: int = 100,
    end_sample: int = 120,
    peak_sample: int = 110,
    channel_start: int = 2,
    channel_stop: int = 8,
    metrics: dict[str, float] | None = None,
) -> Candidate:
    return Candidate(
        detector=detector,
        start_sample=start_sample,
        end_sample=end_sample,
        peak_sample=peak_sample,
        channel_start=channel_start,
        channel_stop=channel_stop,
        score=score,
        metrics={"sample_rate_hz": 100.0} if metrics is None else metrics,
    )


def test_strong_stalta_uses_profile_strong_ratio_and_can_form_an_event() -> None:
    data = _noise(500, 16, 1)
    data[300:311, 5:11] += _pulse(11, 8.0)[:, None]

    result = screen_block(_block(data), _profile(), CPU, _amplitude_options())

    stalta = [candidate for candidate in result.candidates if candidate.detector.startswith("stalta:")]
    assert stalta
    strongest = max(stalta, key=lambda candidate: candidate.metrics["robust_peak_ratio"])
    assert strongest.metrics["robust_peak_ratio"] >= 6.0
    assert strongest.score >= 0.85
    assert result.events


def test_scan_distance_excludes_external_pulses_and_preserves_original_channels():
    data = _noise(700, 32, 177)
    data[300:311, :8] += _pulse(11, 8.0)[:, None]
    data[500:511, 18:24] += _pulse(11, 8.0)[:, None]
    block = _block(data, spacing_m=20.0)
    options = _amplitude_options(spatial_window_m=160.0)
    full = screen_block(block, _profile(), CPU, options)
    assert any(item.channel_start < 15 for item in full.events)

    profile = _profile(scan_distance_start_m=300, scan_distance_stop_m=540)
    ranged = screen_block(block, profile, CPU, options)
    assert ranged.events
    assert all(15 <= item.channel_start < item.channel_stop <= 28 for item in ranged.candidates)
    assert any(10_480 <= item.peak_sample <= 10_530 for item in ranged.events)
    assert not any(10_280 <= item.peak_sample <= 10_330 for item in ranged.events)

    contaminated = data.copy()
    contaminated[:, :15] *= 1e30
    contaminated[:, 28:] *= 1e30
    changed = screen_block(_block(contaminated, spacing_m=20.0), profile, CPU, options)
    assert changed.candidates == ranged.candidates
    np.testing.assert_array_equal(block.data, data)


def test_scan_distance_rejects_a_range_without_source_channels():
    with pytest.raises(ValueError, match="scan distance range"):
        screen_block(_block(_noise(100, 8, 9)), _profile(scan_distance_start_m=300), CPU)


def test_scattered_channels_do_not_count_as_local_support() -> None:
    pulse = _pulse(11, 8.0)
    scattered = _noise(500, 24, 2)
    scattered[300:311, [0, 6, 12, 18]] += pulse[:, None]
    local = _noise(500, 24, 2)
    local[300:311, 8:12] += pulse[:, None]
    options = _amplitude_options(spatial_window_m=35.0)

    scattered_result = screen_block(_block(scattered, spacing_m=10.0), _profile(), CPU, options)
    local_result = screen_block(_block(local, spacing_m=10.0), _profile(), CPU, options)

    assert not any(candidate.detector.startswith("stalta:") for candidate in scattered_result.candidates)
    assert any(candidate.detector.startswith("stalta:") for candidate in local_result.candidates)
    assert all(event.channel_stop - event.channel_start < 10 for event in local_result.events)


def test_release_hysteresis_bridges_a_short_below_threshold_gap() -> None:
    data = _noise(500, 12, 3)
    pulse = _pulse(9, 7.0)
    data[280:289, 3:9] += pulse[:, None]
    data[294:303, 3:9] += pulse[:, None]

    result = screen_block(
        _block(data),
        _profile(),
        CPU,
        _amplitude_options(release_gap_seconds=0.1),
    )

    long_scale = [
        candidate
        for candidate in result.candidates
        if candidate.detector == "stalta:long"
        and 10_250 <= candidate.peak_sample <= 10_330
    ]
    assert long_scale
    assert any(
        candidate.start_sample <= 10_288 and candidate.end_sample >= 10_294
        for candidate in long_scale
    )


def test_short_scale_detects_during_long_scale_cold_start_and_reports_coverage() -> None:
    data = _noise(400, 12, 4)
    data[70:81, 3:9] += _pulse(11, 8.0)[:, None]
    profile = _profile(lta_seconds=2.0)
    options = _amplitude_options(short_lta_seconds=0.25, short_sta_seconds=0.03)

    result = screen_block(_block(data, sample_offset=7_000), profile, CPU, options)

    early = [candidate for candidate in result.candidates if candidate.peak_sample < 7_200]
    assert any(candidate.detector == "stalta:short" for candidate in early)
    assert not any(candidate.detector == "stalta:long" for candidate in early)
    assert result.diagnostics["cold_start_seconds"] == pytest.approx(0.28)
    coverage = result.diagnostics["effective_sample_coverage"]
    assert coverage["start_sample"] == 7_028
    assert coverage["fully_conditioned_start_sample"] == 7_205
    assert 0.0 < coverage["fraction"] < 1.0


def test_coherence_keeps_two_separated_pulses_from_one_block() -> None:
    data = _noise(700, 10, 5)
    distances = np.arange(10) * 5.0
    pulse = _pulse(13, 4.0)
    for channel, distance in enumerate(distances):
        delay = round(0.002 * distance * 100.0)
        for center in (220, 500):
            data[center + delay : center + delay + pulse.size, channel] += pulse
    options = _amplitude_options(enable_coherence=True)

    result = screen_block(_block(data), _profile(), CPU, options)

    coherent = sorted(
        (candidate for candidate in result.candidates if candidate.detector == "coherence"),
        key=lambda candidate: candidate.peak_sample,
    )
    assert len(coherent) >= 2
    assert coherent[0].peak_sample < 10_300
    assert coherent[-1].peak_sample > 10_450


def test_near_zero_moveout_is_retained_as_ambiguous_evidence() -> None:
    data = _noise(500, 10, 6)
    data[260:273] += _pulse(13, 4.0)[:, None]
    options = _amplitude_options(enable_coherence=True)

    result = screen_block(_block(data), _profile(), CPU, options)

    near_zero = [
        candidate
        for candidate in result.candidates
        if candidate.detector == "coherence"
        and "near_zero_moveout_ambiguous" in candidate.quality_flags
    ]
    assert near_zero
    assert "apparent_velocity_mps" not in near_zero[0].metrics
    assert any(near_zero[0] in event.evidence for event in result.events)


def test_persistent_coherence_is_audited_but_does_not_create_interior_events():
    fs = 200.0
    data = _noise(4000, 64, 20260913)
    data += (0.0085 * np.sin(2 * np.pi * 8 * np.arange(4000) / fs))[:, None]
    profile = ProcessingProfile()
    result = screen_block(
        _block(data, sample_rate_hz=fs, spacing_m=4), profile, CPU,
        ScreeningOptions(enable_trajectory=False),
    )
    persistent = [
        candidate for candidate in result.candidates
        if "persistent_coherence_background" in candidate.quality_flags
    ]
    assert persistent
    assert all(candidate.detector == "coherence" for candidate in persistent)
    assert all(candidate.metrics["coherence_peak_prominence"] < 0.04 for candidate in persistent)
    assert all(candidate.metrics["coherence_beam_energy_ratio"] < 1.5 for candidate in persistent)
    assert result.diagnostics["detectors"]["coherence"]["persistent_background_count"] == len(persistent)
    assert all(candidate not in persistent for event in result.events for candidate in event.evidence)
    unmarked = [replace(candidate, quality_flags=()) for candidate in result.candidates]
    assert len(result.events) < len(fuse_screening_candidates(unmarked, profile))


def test_weak_coherence_only_transient_remains_a_primary_event():
    fs = 200.0
    data = _noise(4000, 64, 20260913)
    data[1600:2200] += (
        0.012 * np.hanning(600) * np.sin(2 * np.pi * 8 * np.arange(600) / fs)
    )[:, None]
    result = screen_block(
        _block(data, sample_rate_hz=fs, spacing_m=4), ProcessingProfile(), CPU,
        ScreeningOptions(enable_trajectory=False),
    )
    assert result.events
    assert {candidate.detector for candidate in result.candidates} == {"coherence"}
    primary = max(result.candidates, key=lambda candidate: candidate.metrics["semblance"])
    assert primary.metrics["coherence_peak_prominence"] >= 0.04
    assert "near_zero_moveout_ambiguous" in primary.quality_flags
    assert "persistent_coherence_background" not in primary.quality_flags
    assert any(primary in event.evidence for event in result.events)


@pytest.mark.parametrize(
    "prominence,current_power,before_power,after_power,peak_index,persistent",
    [
        (0.02, 1.0, 1.0, 1.0, 100, True),
        (0.04, 1.0, 1.0, 1.0, 100, False),
        (0.05, 1.0, 1.0, 1.0, 100, False),
        (0.02, 1.5, 1.0, 1.0, 100, False),
        (0.02, 1.6, 1.0, 1.0, 100, False),
        (0.02, 1.6, 2.0, 1.0, 100, False),
        (0.02, 1.0, 1.0, 1.0, 10, False),
        (0.02, 1.0, 1.0, 1.0, 170, False),
    ],
)
def test_coherence_background_requires_both_complete_quiet_flanks(
    monkeypatch, prominence, current_power, before_power, after_power, peak_index, persistent,
):
    from das_sentinel import screening

    width, samples = 20, 200
    curve = np.full(samples - width + 1, 0.2)
    curve[peak_index] += prominence
    stack = np.ones(samples)
    stack[max(0, peak_index - width):peak_index] = np.sqrt(before_power)
    stack[peak_index:peak_index + width] = np.sqrt(current_power)
    stack[peak_index + width:peak_index + 2 * width] = np.sqrt(after_power)
    monkeypatch.setattr(screening, "_moveout_semblance", lambda *args: (curve, stack))
    block = PreprocessedBlock(
        data=np.zeros((samples, 8), dtype=np.float32),
        bad_channel_mask=np.zeros(8, dtype=bool), backend_name="cpu",
        sample_offset=1000, sample_rate_hz=10,
        channel_distance_m=np.arange(8, dtype=float),
    )
    profile = _profile(coherence_window_seconds=2, coherence_freq_high_hz=4)
    candidates, _ = screening._coherence_candidates(
        block, [np.arange(8)], profile, ScreeningOptions(), CPU,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.metrics["coherence_peak_prominence"] == pytest.approx(prominence)
    complete = width <= peak_index and peak_index + 2 * width <= samples
    assert candidate.metrics["coherence_background_complete"] == float(complete)
    if complete:
        assert candidate.metrics["coherence_beam_energy_ratio"] == pytest.approx(
            current_power / min(before_power, after_power)
        )
    else:
        assert "coherence_beam_energy_ratio" not in candidate.metrics
    assert ("persistent_coherence_background" in candidate.quality_flags) is persistent


def test_coherence_global_maximum_fallback_keeps_missing_background(monkeypatch):
    from das_sentinel import screening

    curve = np.full(181, 0.2)
    monkeypatch.setattr(screening, "_moveout_semblance", lambda *args: (curve, np.ones(200)))
    block = PreprocessedBlock(
        data=np.zeros((200, 8), dtype=np.float32),
        bad_channel_mask=np.zeros(8, dtype=bool), backend_name="cpu",
        sample_offset=1000, sample_rate_hz=10,
        channel_distance_m=np.arange(8, dtype=float),
    )
    profile = _profile(coherence_window_seconds=2, coherence_freq_high_hz=4)
    candidates, _ = screening._coherence_candidates(
        block, [np.arange(8)], profile, ScreeningOptions(), CPU,
    )
    assert len(candidates) == 1
    assert candidates[0].metrics["coherence_peak_prominence"] == 0.0
    assert candidates[0].metrics["coherence_background_complete"] == 0.0
    assert "persistent_coherence_background" not in candidates[0].quality_flags
    assert fuse_screening_candidates(candidates, profile)


def test_coherence_background_must_remain_complete_after_moveout(monkeypatch):
    from das_sentinel import screening

    curve = np.full(181, 0.2)
    curve[80] = 0.22

    def moveout(data, delays, width, xp):
        return (curve if int(delays.min()) == -70 else np.zeros(181)), np.ones(200)

    monkeypatch.setattr(screening, "_moveout_semblance", moveout)
    block = PreprocessedBlock(
        data=np.zeros((200, 8), dtype=np.float32),
        bad_channel_mask=np.zeros(8, dtype=bool), backend_name="cpu",
        sample_offset=1000, sample_rate_hz=10,
        channel_distance_m=np.arange(8, dtype=float) * 10,
    )
    profile = _profile(coherence_window_seconds=2, coherence_freq_high_hz=4,
                       slowness_max_s_per_m=0.1, slowness_points=3)
    candidates, _ = screening._coherence_candidates(
        block, [np.arange(8)], profile, ScreeningOptions(), CPU,
    )
    assert len(candidates) == 1
    assert candidates[0].metrics["slowness_s_per_m"] == pytest.approx(-0.1)
    assert candidates[0].metrics["coherence_background_complete"] == 0.0
    assert "persistent_coherence_background" not in candidates[0].quality_flags


def test_persistent_coherence_does_not_suppress_a_primary_spatial_candidate():
    background = replace(
        _candidate("coherence", 0.99, channel_start=0, channel_stop=20),
        quality_flags=("persistent_coherence_background",),
    )
    primary = _candidate("coherence", 0.7, channel_start=4, channel_stop=8)
    candidates = _suppress_duplicates([background, primary], peak_tolerance=3)
    assert background in candidates
    assert primary in candidates


@pytest.mark.parametrize("amplitude_score", [0.60, 0.85])
def test_persistent_coherence_cannot_vote_or_change_primary_event_bounds(amplitude_score):
    profile = _profile()
    background = replace(
        _candidate("coherence", 1.0, start_sample=0, end_sample=1000,
                   peak_sample=10, channel_start=0, channel_stop=20),
        quality_flags=("persistent_coherence_background",),
    )
    amplitude = _candidate("stalta:long", amplitude_score)
    assert fuse_screening_candidates((background,), profile) == ()
    events = fuse_screening_candidates((background, amplitude), profile)
    if amplitude_score < profile.fusion_strong_score:
        assert events == ()
    else:
        assert len(events) == 1
        assert events[0].evidence == (amplitude,)
        assert events[0].start_sample == amplitude.start_sample
        assert events[0].end_sample == amplitude.end_sample
        assert events[0].peak_sample == amplitude.peak_sample
        assert events[0].channel_start == amplitude.channel_start
        assert events[0].channel_stop == amplitude.channel_stop


def test_trajectory_scans_a_feasible_local_part_of_a_long_cable() -> None:
    samples, channels, sample_rate_hz = 800, 100, 100.0
    data = _noise(samples, channels, 7)
    distances = np.concatenate(
        (np.linspace(0.0, 0.01, 10), 1_000.0 + np.arange(90) * 5.0)
    ).astype(np.float32)
    pulse = _pulse(11, 5.0)
    for channel in range(42, 52):
        center = 240 + round((distances[channel] - distances[42]) / 20.0 * sample_rate_hz)
        data[center : center + pulse.size, channel] += pulse
    options = _amplitude_options(
        spatial_window_m=60.0,
        enable_trajectory=True,
    )

    result = screen_block(
        _block(data, sample_rate_hz=sample_rate_hz, distances_m=distances),
        _profile(),
        CPU,
        options,
    )

    trajectories = [candidate for candidate in result.candidates if candidate.detector == "trajectory"]
    assert trajectories
    assert any(
        candidate.channel_start < 52 and candidate.channel_stop > 42
        for candidate in trajectories
    )
    assert all(candidate.channel_stop - candidate.channel_start <= 15 for candidate in trajectories)
    assert result.diagnostics["skipped_geometry"]["trajectory_infeasible_windows"] > 0


def test_amplitude_detectors_share_one_fusion_family_but_energy_can_be_strong() -> None:
    profile = _profile()
    normal = profile.fusion_normal_score
    strong = profile.fusion_strong_score

    assert fuse_screening_candidates(
        (_candidate("stalta:short", normal), _candidate("energy:0", normal)),
        profile,
    ) == ()
    assert len(
        fuse_screening_candidates(
            (
                _candidate("stalta:short", normal),
                _candidate("energy:0", normal),
                _candidate("coherence", normal),
            ),
            profile,
        )
    ) == 1
    assert len(fuse_screening_candidates((_candidate("energy:0", strong),), profile)) == 1


def test_normal_amplitude_support_metrics_do_not_create_an_independent_vote() -> None:
    profile = _profile()
    metrics = {
        "sample_rate_hz": 100.0,
        "robust_peak_ratio": 3.5,
        "peak_support_channels": 4.0,
        "local_valid_channels": 64.0,
        "local_support_fraction": 4.0 / 64.0,
        "strong_ratio_threshold": 6.0,
    }

    events = fuse_screening_candidates(
        (_candidate("stalta:short", profile.fusion_normal_score, metrics=metrics),),
        profile,
    )

    assert events == ()


def test_ratio_candidate_duration_and_end_include_the_low_threshold_tail() -> None:
    profile = _profile(min_duration_seconds=0.5)
    options = _amplitude_options(
        coincidence_tolerance_seconds=0.0,
        release_gap_seconds=0.08,
    )
    ratios = np.ones((100, 4), dtype=np.float32)
    ratios[10:20] = 4.0
    ratios[20:80] = 2.0
    powers = np.ones_like(ratios)
    block = PreprocessedBlock(
        data=np.zeros((100, 4), dtype=np.float32),
        bad_channel_mask=np.zeros(4, dtype=bool),
        backend_name="cpu",
        sample_offset=10_000,
        sample_rate_hz=100.0,
        channel_distance_m=np.arange(4, dtype=np.float32),
    )

    candidates = _ratio_candidates(
        ratios=ratios,
        sample_indices=np.arange(100),
        current_power=powers,
        background_power=powers,
        window_channels=np.arange(4),
        good_local=np.ones(4, dtype=bool),
        block=block,
        profile=profile,
        options=options,
        detector="stalta:long",
        threshold=3.0,
        strong_threshold=6.0,
        extra_metrics={},
    )

    assert len(candidates) == 1
    assert (candidates[0].start_sample, candidates[0].end_sample) == (10_010, 10_080)


def test_duplicate_suppression_keeps_distant_peaks_with_partial_overlap() -> None:
    first = _candidate(
        "stalta:long",
        0.85,
        start_sample=100,
        end_sample=250,
        peak_sample=110,
        channel_start=0,
        channel_stop=10,
    )
    second = _candidate(
        "stalta:long",
        0.85,
        start_sample=240,
        end_sample=260,
        peak_sample=250,
        channel_start=9,
        channel_stop=20,
    )

    assert _suppress_duplicates((first, second), peak_tolerance=2) == [first, second]


def test_duplicate_suppression_keeps_unique_spatial_coverage_at_the_same_peak() -> None:
    left = _candidate(
        "coherence",
        0.85,
        peak_sample=110,
        channel_start=0,
        channel_stop=10,
    )
    right = _candidate(
        "coherence",
        0.84,
        peak_sample=110,
        channel_start=9,
        channel_stop=20,
    )

    assert _suppress_duplicates((left, right), peak_tolerance=2) == [left, right]


def test_duplicate_suppression_keeps_a_lower_score_candidate_with_wider_coverage() -> None:
    narrow = _candidate(
        "coherence",
        0.85,
        start_sample=100,
        end_sample=120,
        peak_sample=110,
        channel_start=5,
        channel_stop=10,
    )
    wide = _candidate(
        "coherence",
        0.84,
        start_sample=90,
        end_sample=130,
        peak_sample=110,
        channel_start=0,
        channel_stop=20,
    )

    assert _suppress_duplicates((narrow, wide), peak_tolerance=2) == [wide, narrow]


def test_fusion_does_not_join_distant_events_through_a_candidate_chain() -> None:
    profile = _profile(merge_gap_seconds=0.2)
    candidates = tuple(
        _candidate(
            "coherence",
            0.85 if index in {0, 10} else 0.60,
            start_sample=100 + 30 * index,
            end_sample=120 + 30 * index,
            peak_sample=110 + 30 * index,
            channel_start=2 + 5 * index,
            channel_stop=8 + 5 * index,
        )
        for index in range(11)
    )

    events = fuse_screening_candidates(candidates, profile)

    assert len(events) == 2
    assert [event.peak_sample for event in events] == [110, 410]


def test_fusion_still_combines_pairwise_compatible_evidence_for_a_long_event() -> None:
    profile = _profile(merge_gap_seconds=0.2)
    candidates = (
        _candidate(
            "stalta:long",
            0.60,
            start_sample=100,
            end_sample=500,
            peak_sample=180,
            channel_start=0,
            channel_stop=30,
        ),
        _candidate(
            "coherence",
            0.60,
            start_sample=140,
            end_sample=480,
            peak_sample=420,
            channel_start=5,
            channel_stop=35,
        ),
        _candidate(
            "trajectory",
            0.60,
            start_sample=160,
            end_sample=450,
            peak_sample=300,
            channel_start=10,
            channel_stop=40,
        ),
    )

    events = fuse_screening_candidates(candidates, profile)

    assert len(events) == 1
    assert (events[0].start_sample, events[0].end_sample) == (100, 500)
    assert (events[0].channel_start, events[0].channel_stop) == (0, 40)


@pytest.mark.parametrize("sample_rate_hz", (1.0, 2.0, 4.0))
def test_unavailable_detector_bands_are_skipped_with_diagnostics(
    sample_rate_hz: float,
) -> None:
    result = screen_block(
        _block(_noise(100, 8, 81), sample_rate_hz=sample_rate_hz),
        _profile(),
        CPU,
        _amplitude_options(),
    )

    assert result.diagnostics["detectors"]["stalta"]["status"] == "skipped"
    assert result.diagnostics["detectors"]["stalta"]["reason"] == "frequency_band_unavailable"
    assert result.diagnostics["detectors"]["energy"]["processed_bands"] == 0
    assert result.diagnostics["detectors"]["energy"]["skipped_bands"]


def test_energy_partially_clips_a_band_at_nyquist_and_continues() -> None:
    profile = _profile(
        stalta_freq_low_hz=0.5,
        stalta_freq_high_hz=10.0,
        energy_bands_hz=((0.5, 5.0), (12.0, 35.0)),
    )

    result = screen_block(
        _block(_noise(100, 8, 82), sample_rate_hz=4.0),
        profile,
        CPU,
        _amplitude_options(),
    )

    energy = result.diagnostics["detectors"]["energy"]
    assert energy["processed_bands"] == 1
    assert energy["effective_bands_hz"] == [(0.5, 2.0)]
    assert energy["skipped_bands"] == [1]


@pytest.mark.parametrize("scale", (1e-7, 1e-8))
def test_detection_validity_is_invariant_to_small_physical_units(scale: float) -> None:
    data = _noise(500, 16, 83)
    data[300:311, 5:11] += _pulse(11, 8.0)[:, None]
    profile = _profile()
    options = _amplitude_options()

    reference = screen_block(_block(data), profile, CPU, options)
    scaled = screen_block(_block(data * scale), profile, CPU, options)

    def target_evidence(result: object) -> set[tuple[str, int, int]]:
        return {
            (candidate.detector, candidate.channel_start, candidate.channel_stop)
            for candidate in result.candidates
            if 10_250 <= candidate.peak_sample <= 10_350
        }

    assert target_evidence(scaled) == target_evidence(reference)
    assert bool(scaled.events) == bool(reference.events)


@pytest.mark.parametrize(
    "options",
    (
        ScreeningOptions(spatial_window_m=0.0),
        ScreeningOptions(spatial_overlap=1.0),
        ScreeningOptions(off_ratio=0.0),
        ScreeningOptions(release_gap_seconds=-0.1),
        ScreeningOptions(short_sta_seconds=1.0, short_lta_seconds=1.0),
        ScreeningOptions(coincidence_tolerance_seconds=-0.01),
    ),
)
def test_invalid_options_are_rejected_before_processing(options: ScreeningOptions) -> None:
    with pytest.raises(ValueError, match="screening option"):
        screen_block(_block(_noise(100, 4, 8)), _profile(), CPU, options)


def test_screening_coordinates_are_absolute_and_ids_are_deterministic() -> None:
    data = _noise(500, 12, 9)
    data[300:311, 3:9] += _pulse(11, 8.0)[:, None]
    block = _block(data, sample_offset=20_000)
    options = _amplitude_options()

    first = screen_block(block, _profile(), CPU, options)
    second = screen_block(block, _profile(), CPU, options)

    assert min(candidate.start_sample for candidate in first.candidates) >= 20_000
    assert first.events == second.events
    assert first.diagnostics["algorithm_version"] == "screening-v2"
