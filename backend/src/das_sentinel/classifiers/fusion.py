from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from das_sentinel.domain import Label

_LABELS = (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)


@dataclass(frozen=True)
class ClassificationResult:
    label: Label
    decision_score: float
    margin: float
    scores: dict[Label, float]
    reasons: tuple[str, ...] = ()


def _clean_evidence(value: object) -> tuple[float, bool]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not isfinite(number):
        return 0.0, True
    return min(1.0, max(0.0, number)), False


def fuse_classification(
    rule_scores: Mapping[Label, float],
    model_probabilities: Mapping[Label, float] | None,
    quality_flags: tuple[str, ...],
) -> ClassificationResult:
    """Fuse rule and optional calibrated-model evidence, with mandatory abstention."""
    flags = set(quality_flags)
    nonfinite = False
    base = {}
    for label in _LABELS:
        value, invalid = _clean_evidence(rule_scores.get(label, 0.0))
        base[label] = value
        nonfinite |= invalid
    if model_probabilities is not None:
        model_scores = {}
        for label in _LABELS:
            value, invalid = _clean_evidence(model_probabilities.get(label, 0.0))
            model_scores[label] = value
            nonfinite |= invalid
        scores = {label: round(0.35 * base[label] + 0.65 * model_scores[label], 12) for label in _LABELS}
    else:
        scores = base
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
    label, decision_score = ordered[0]
    margin = round(decision_score - ordered[1][1], 12)
    if nonfinite:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("nonfinite_classification_evidence",))
    if "zero_slowness_common_mode" in flags:
        return ClassificationResult(Label.NOISE, decision_score, margin, scores, ("zero_slowness_common_mode",))
    if flags & {"bad_channel_majority", "edge_only", "edge_only_event"}:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("quality_evidence_insufficient",))
    if decision_score < 0.60:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("decision_score_below_threshold",))
    if margin < 0.15:
        return ClassificationResult(Label.UNKNOWN, decision_score, margin, scores, ("top_two_margin_below_threshold",))
    return ClassificationResult(label, decision_score, margin, scores)
