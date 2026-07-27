from das_sentinel.classifiers.rules import score_rules
from das_sentinel.domain import Label


def test_rule_classifier_explains_traffic():
    features = {
        "duration_s": 10.0,
        "apparent_velocity_mps": 20.0,
        "continuity": 0.9,
        "coherence": 0.8,
        "zero_slowness_flag": 0.0,
    }

    result = score_rules(features, {"traffic_velocity_mps": (2.0, 60.0)})

    assert max(result.scores, key=result.scores.get) == Label.TRAFFIC
    assert "trajectory_speed_in_range" in result.reasons[Label.TRAFFIC]
    assert all(0.0 <= score <= 1.0 for score in result.scores.values())


def test_rules_return_stable_zero_scores_when_evidence_is_missing():
    result = score_rules({}, {})

    assert tuple(result.scores) == (Label.EARTHQUAKE, Label.BLAST, Label.TRAFFIC, Label.NOISE)
    assert all(score == 0.0 for score in result.scores.values())
