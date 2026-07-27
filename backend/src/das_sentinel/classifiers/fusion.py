from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from das_sentinel.domain import Label

_LABELS = (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)


@dataclass(frozen=True)
class ClassificationResult:
    label: Label
    decision_score: float
    margin: float
    scores: dict[Label, float]
    reasons: tuple[str, ...] = ()


def fuse_classification(
    rule_scores: Mapping[Label, float],
    model_probabilities: Mapping[Label, float] | None,
    quality_flags: tuple[str, ...],
) -> ClassificationResult:
    """Fuse rule and optional calibrated-model evidence, with mandatory abstention."""
    flags = set(quality_flags)
    base = {label: min(1.0, max(0.0, float(rule_scores.get(label, 0.0)))) for label in _LABELS}
    if model_probabilities is not None:
        scores = {label: round(0.35 * base[label] + 0.65 * min(1.0, max(0.0, float(model_probabilities.get(label, 0.0)))), 12) for label in _LABELS}
    else:
        scores = base
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
    label, decision_score = ordered[0]
    margin = round(decision_score - ordered[1][1], 12)
    if "zero_slowness_common_mode" in flags:
        return ClassificationResult(Label.NOISE, decision_score, margin, scores, ("zero_slowness_common_mode",))
    if flags & {"bad_channel_majority", "edge_only", "edge_only_event"}:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("quality_evidence_insufficient",))
    if decision_score < 0.60:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("decision_score_below_threshold",))
    if margin < 0.15:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("top_two_margin_below_threshold",))
    return ClassificationResult(label, decision_score, margin, scores)
