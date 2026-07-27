from dataclasses import dataclass

import pytest
from das_sentinel.classifiers.similarity import FeatureVectorizer, SimilarityIndex


@dataclass(frozen=True)
class FeatureRow:
    event_id: str
    features: dict[str, float | str]


@pytest.fixture
def feature_rows():
    return [
        FeatureRow(f"event-{index}", {"duration_s": float(index), "rms": float(index % 2), "site_id": "site-a"})
        for index in range(8)
    ]


def test_similarity_returns_cluster_and_reviewed_neighbors(feature_rows):
    index = SimilarityIndex().fit(feature_rows)

    result = index.query(feature_rows[0].features, limit=3)

    assert result.cluster_id is not None
    assert result.neighbors[0].event_id == feature_rows[0].event_id
    assert result.neighbors[0].distance == pytest.approx(0.0)


def test_vectorizer_rejects_unapproved_or_string_features():
    vectorizer = FeatureVectorizer()

    with pytest.raises(ValueError, match="approved numeric feature"):
        vectorizer.transform_one({"source_path": "Z:/raw/file.h5"})
    with pytest.raises(TypeError, match="must be numeric"):
        vectorizer.transform_one({"duration_s": "one"})


def test_vectorizer_zero_fills_missing_and_nonfinite_values():
    vector = FeatureVectorizer().transform_one({"duration_s": float("nan")})

    assert vector[0] == 0.0
    assert vector.shape == (len(FeatureVectorizer().feature_names),)
