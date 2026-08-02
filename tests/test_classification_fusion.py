import pytest
from das_sentinel.classifiers.fusion import fuse_classification
from das_sentinel.domain import Label


def test_fusion_abstains_when_top_scores_are_close():
    result = fuse_classification(
        rule_scores={Label.EARTHQUAKE: 0.62, Label.BLAST: 0.55},
        model_probabilities=None,
        quality_flags=(),
    )

    assert result.label == Label.UNKNOWN
    assert result.decision_score == 0.62
    assert result.margin == 0.07


def test_fusion_uses_weighted_rules_and_model_when_available():
    result = fuse_classification(
        rule_scores={Label.TRAFFIC: 0.8, Label.BLAST: 0.1},
        model_probabilities={Label.TRAFFIC: 0.7, Label.BLAST: 0.2},
        quality_flags=(),
    )

    assert result.label == Label.TRAFFIC
    assert result.decision_score == 0.735
    assert result.margin == 0.57


def test_fusion_quality_flags_override_high_scores():
    result = fuse_classification(
        rule_scores={Label.EARTHQUAKE: 0.9},
        model_probabilities=None,
        quality_flags=("zero_slowness_common_mode",),
    )

    assert result.label == Label.NOISE


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_fusion_abstains_for_nonfinite_rule_evidence(value):
    result = fuse_classification({Label.EARTHQUAKE: value}, None, ())

    assert result.label == Label.UNKNOWN
    assert result.reasons == ("nonfinite_classification_evidence",)
    assert all(0.0 <= score <= 1.0 for score in result.scores.values())


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_fusion_abstains_for_nonfinite_model_evidence(value):
    result = fuse_classification({Label.EARTHQUAKE: 0.9}, {Label.EARTHQUAKE: value}, ())

    assert result.label == Label.UNKNOWN
    assert result.reasons == ("nonfinite_classification_evidence",)
    assert all(0.0 <= score <= 1.0 for score in result.scores.values())
