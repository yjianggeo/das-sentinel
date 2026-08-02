from __future__ import annotations

import math
from collections.abc import Iterable
from uuid import NAMESPACE_URL, uuid5

from das_sentinel.config import ProcessingProfile
from das_sentinel.domain import Candidate, EventRecord


def _candidate_key(candidate: Candidate) -> tuple[object, ...]:
    return (
        candidate.start_sample,
        candidate.end_sample,
        candidate.channel_start,
        candidate.channel_stop,
        candidate.detector,
        candidate.peak_sample,
        candidate.score,
        tuple(sorted(candidate.metrics.items())),
        candidate.quality_flags,
    )


def _evidence_key(candidate: Candidate) -> tuple[object, ...]:
    return (candidate.detector, candidate.peak_sample, *_candidate_key(candidate))


def _same_event(left: Candidate, right: Candidate, merge_gap_samples: int) -> bool:
    temporal_overlap = max(left.start_sample, right.start_sample) <= min(
        left.end_sample, right.end_sample
    ) + merge_gap_samples
    channel_overlap_or_adjacent = (
        left.channel_start <= right.channel_stop and right.channel_start <= left.channel_stop
    )
    return temporal_overlap and channel_overlap_or_adjacent


def _detector_family(detector: str) -> str:
    return detector.split(":", 1)[0]


def _passes_gate(evidence: tuple[Candidate, ...], profile: ProcessingProfile) -> bool:
    valid_evidence = tuple(candidate for candidate in evidence if not candidate.quality_flags)
    if any(candidate.score >= profile.fusion_strong_score for candidate in valid_evidence):
        return True
    normal_families = {
        _detector_family(candidate.detector)
        for candidate in valid_evidence
        if candidate.score >= profile.fusion_normal_score
    }
    return len(normal_families) >= 2


def _event_from_evidence(
    evidence: tuple[Candidate, ...],
    site_id: str,
    pipeline_version: str,
    source_fingerprint: str | None,
) -> EventRecord:
    ordered_evidence = tuple(sorted(evidence, key=_evidence_key))
    peak_candidate = min(
        ordered_evidence,
        key=lambda candidate: (-candidate.score, candidate.peak_sample, _evidence_key(candidate)),
    )
    start_sample = min(candidate.start_sample for candidate in ordered_evidence)
    end_sample = max(candidate.end_sample for candidate in ordered_evidence)
    channel_start = min(candidate.channel_start for candidate in ordered_evidence)
    channel_stop = max(candidate.channel_stop for candidate in ordered_evidence)
    event_name = f"{site_id}|{pipeline_version}|{peak_candidate.peak_sample}|{channel_start}|{channel_stop}"
    if source_fingerprint is not None:
        event_name = f"{event_name}|source_fingerprint={source_fingerprint}"
    return EventRecord(
        event_id=str(uuid5(NAMESPACE_URL, event_name)),
        start_sample=start_sample,
        end_sample=end_sample,
        peak_sample=peak_candidate.peak_sample,
        channel_start=channel_start,
        channel_stop=channel_stop,
        evidence=ordered_evidence,
    )


def fuse_candidates(
    candidates: Iterable[Candidate],
    profile: ProcessingProfile,
    sample_rate_hz: float,
    site_id: str,
    pipeline_version: str,
    *,
    source_fingerprint: str | None = None,
) -> list[EventRecord]:
    """Merge compatible detector evidence into stable, reviewable event records."""
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if profile.merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds must be nonnegative")

    merge_gap_samples = math.floor(profile.merge_gap_seconds * sample_rate_hz)
    remaining = sorted(candidates, key=_candidate_key)
    groups: list[tuple[Candidate, ...]] = []
    while remaining:
        group = [remaining.pop(0)]
        index = 0
        while index < len(remaining):
            candidate = remaining[index]
            if any(_same_event(candidate, member, merge_gap_samples) for member in group):
                group.append(remaining.pop(index))
                index = 0
            else:
                index += 1
        groups.append(tuple(group))

    events = [
        _event_from_evidence(
            group, site_id, pipeline_version, source_fingerprint
        )
        for group in groups
        if _passes_gate(group, profile)
    ]
    return sorted(
        events,
        key=lambda event: (
            event.start_sample,
            event.end_sample,
            event.channel_start,
            event.channel_stop,
            event.event_id,
        ),
    )
