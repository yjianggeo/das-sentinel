from __future__ import annotations

import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate
from das_sentinel.fusion import fuse_candidates


@pytest.fixture
def profile() -> ProcessingProfile:
    return ProcessingProfile(
        fusion_normal_score=0.60,
        fusion_strong_score=0.85,
        merge_gap_seconds=1.0,
    )


def candidate(
    detector: str,
    start_sample: int,
    end_sample: int,
    score: float,
    *,
    peak_sample: int | None = None,
    channel_start: int = 3,
    channel_stop: int = 8,
    quality_flags: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        detector=detector,
        start_sample=start_sample,
        end_sample=end_sample,
        peak_sample=(start_sample + end_sample) // 2 if peak_sample is None else peak_sample,
        channel_start=channel_start,
        channel_stop=channel_stop,
        score=score,
        quality_flags=quality_flags,
    )


def test_two_overlapping_normal_candidates_form_event(profile):
    events = fuse_candidates(
        [candidate("stalta", 100, 180, 0.65), candidate("energy", 120, 190, 0.64)],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert len(events) == 1
    assert {e.detector for e in events[0].evidence} == {"stalta", "energy"}


def test_single_weak_candidate_stays_out_of_review_queue(profile):
    assert fuse_candidates([candidate("energy", 100, 180, 0.45)], profile, 100.0, "site-a", "v1") == []


def test_event_id_is_deterministic(profile):
    first = fuse_candidates([candidate("stalta", 100, 180, 0.95)], profile, 100.0, "s", "v1")[0]
    second = fuse_candidates([candidate("stalta", 100, 180, 0.95)], profile, 100.0, "s", "v1")[0]

    assert first.event_id == second.event_id


def test_source_fingerprint_namespaces_event_id_without_breaking_legacy_calls(profile):
    evidence = [candidate("stalta", 100, 180, 0.95)]

    legacy = fuse_candidates(evidence, profile, 100.0, "s", "v1")[0]
    first = fuse_candidates(
        evidence, profile, 100.0, "s", "v1", source_fingerprint="source-a"
    )[0]
    repeated = fuse_candidates(
        evidence, profile, 100.0, "s", "v1", source_fingerprint="source-a"
    )[0]
    second = fuse_candidates(
        evidence, profile, 100.0, "s", "v1", source_fingerprint="source-b"
    )[0]

    assert first.event_id == repeated.event_id
    assert first.event_id != second.event_id
    assert legacy.event_id not in {first.event_id, second.event_id}


def test_candidates_merge_across_offsets_when_gap_is_within_sample_rate_window(profile):
    events = fuse_candidates(
        [candidate("stalta", 10_000, 10_050, 0.70), candidate("energy", 10_150, 10_210, 0.70)],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert len(events) == 1
    assert (events[0].start_sample, events[0].end_sample) == (10_000, 10_210)


def test_candidates_outside_sample_rate_converted_gap_do_not_merge(profile):
    events = fuse_candidates(
        [candidate("stalta", 100, 150, 0.70), candidate("energy", 251, 300, 0.70)],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert events == []


def test_fractional_merge_gap_does_not_round_up_to_an_extra_sample(profile):
    fractional_gap_profile = profile.model_copy(update={"merge_gap_seconds": 0.006})

    events = fuse_candidates(
        [candidate("stalta", 100, 150, 0.70), candidate("energy", 151, 200, 0.70)],
        fractional_gap_profile,
        100.0,
        "site-a",
        "v1",
    )

    assert events == []


def test_fractional_merge_gap_allows_zero_sample_gap(profile):
    fractional_gap_profile = profile.model_copy(update={"merge_gap_seconds": 0.006})

    events = fuse_candidates(
        [candidate("stalta", 100, 150, 0.70), candidate("energy", 150, 200, 0.70)],
        fractional_gap_profile,
        100.0,
        "site-a",
        "v1",
    )

    assert len(events) == 1


def test_rejects_negative_merge_gap(profile):
    invalid_profile = profile.model_copy(update={"merge_gap_seconds": -0.01})

    with pytest.raises(ValueError, match="merge_gap_seconds must be nonnegative"):
        fuse_candidates([], invalid_profile, 100.0, "site-a", "v1")


def test_spatially_disjoint_candidates_do_not_merge(profile):
    events = fuse_candidates(
        [
            candidate("stalta", 100, 180, 0.70, channel_start=0, channel_stop=3),
            candidate("energy", 120, 190, 0.70, channel_start=4, channel_stop=8),
        ],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert events == []


def test_spatially_adjacent_candidates_merge(profile):
    events = fuse_candidates(
        [
            candidate("stalta", 100, 180, 0.70, channel_start=0, channel_stop=3),
            candidate("energy", 120, 190, 0.70, channel_start=3, channel_stop=8),
        ],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert len(events) == 1
    assert (events[0].channel_start, events[0].channel_stop) == (0, 8)


def test_multiple_candidates_from_one_detector_family_do_not_satisfy_normal_gate(profile):
    events = fuse_candidates(
        [candidate("energy", 100, 180, 0.70), candidate("energy", 120, 190, 0.70)],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert events == []


def test_exact_normal_threshold_requires_two_families(profile):
    event = fuse_candidates(
        [candidate("stalta", 100, 180, 0.60), candidate("energy", 120, 190, 0.60)],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert len(event) == 1


def test_quality_flag_vetoes_otherwise_strong_single_candidate(profile):
    events = fuse_candidates(
        [
            candidate(
                "coherence",
                100,
                180,
                0.85,
                quality_flags=("zero_slowness_common_mode",),
            )
        ],
        profile,
        100.0,
        "site-a",
        "v1",
    )

    assert events == []


def test_exact_strong_threshold_allows_one_quality_valid_candidate(profile):
    events = fuse_candidates([candidate("stalta", 100, 180, 0.85)], profile, 100.0, "site-a", "v1")

    assert len(events) == 1


def test_input_order_does_not_change_event_or_evidence_order(profile):
    candidates = [
        candidate("energy", 120, 190, 0.64, peak_sample=150),
        candidate("stalta", 100, 180, 0.65, peak_sample=140),
    ]

    forward = fuse_candidates(candidates, profile, 100.0, "site-a", "v1")
    reversed_input = fuse_candidates(list(reversed(candidates)), profile, 100.0, "site-a", "v1")

    assert forward == reversed_input
    assert [e.detector for e in forward[0].evidence] == ["energy", "stalta"]


def test_event_uses_highest_score_peak_and_all_evidence(profile):
    low = candidate("stalta", 100, 180, 0.65, peak_sample=130)
    high = candidate("energy", 120, 190, 0.90, peak_sample=165)

    event = fuse_candidates([low, high], profile, 100.0, "site-a", "v1")[0]

    assert event.peak_sample == 165
    assert event.evidence == (high, low)


@pytest.mark.parametrize("sample_rate_hz", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_nonpositive_or_nonfinite_sample_rate(profile, sample_rate_hz):
    with pytest.raises(ValueError, match="sample_rate_hz must be positive and finite"):
        fuse_candidates([], profile, sample_rate_hz, "site-a", "v1")
