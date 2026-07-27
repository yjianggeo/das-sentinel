from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from das_sentinel.domain import Label


@dataclass(frozen=True)
class RuleResult:
    scores: dict[Label, float]
    reasons: dict[Label, tuple[str, ...]]


def _number(features: Mapping[str, object], key: str) -> float:
    value = features.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def score_rules(features: Mapping[str, object], site_profile: Mapping[str, object]) -> RuleResult:
    """Return bounded physical-rule evidence; absent evidence never invents a class."""
    scores = {label: 0.0 for label in (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)}
    reasons = {label: () for label in scores}
    duration = _number(features, "duration_s")
    continuity = _number(features, "continuity")
    coherence = _number(features, "coherence")
    velocity = _number(features, "apparent_velocity_mps")
    peak_ratio = _number(features, "peak_median_ratio")
    zero_slowness = _number(features, "zero_slowness_flag")
    bounds = site_profile.get("traffic_velocity_mps", (2.0, 60.0))
    low, high = (float(bounds[0]), float(bounds[1])) if isinstance(bounds, tuple | list) and len(bounds) == 2 else (2.0, 60.0)
    if low <= velocity <= high and velocity > 0:
        scores[Label.TRAFFIC] += 0.5
        reasons[Label.TRAFFIC] += ("trajectory_speed_in_range",)
    if continuity >= 0.5:
        scores[Label.TRAFFIC] += 0.25
        reasons[Label.TRAFFIC] += ("spatially_continuous",)
    if coherence >= 0.5:
        scores[Label.TRAFFIC] += 0.25
        reasons[Label.TRAFFIC] += ("coherent_propagation",)
    if 0 < duration <= 5.0:
        scores[Label.BLAST] += 0.5
        reasons[Label.BLAST] += ("short_duration_impulse",)
    if peak_ratio >= 5.0:
        scores[Label.BLAST] += 0.5
        reasons[Label.BLAST] += ("impulsive_peak",)
    if duration >= 15.0:
        scores[Label.EARTHQUAKE] += 0.5
        reasons[Label.EARTHQUAKE] += ("sustained_duration",)
    if coherence >= 0.6 and not zero_slowness:
        scores[Label.EARTHQUAKE] += 0.5
        reasons[Label.EARTHQUAKE] += ("coherent_non_common_mode",)
    if zero_slowness:
        scores[Label.NOISE] = 1.0
        reasons[Label.NOISE] = ("zero_slowness_common_mode",)
    return RuleResult(scores, reasons)
