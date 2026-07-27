from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from das_sentinel.features import FEATURE_NAMES

_CONTEXT_FEATURES = frozenset({"site_id", "cable_type", "device_config"})
_NUMERIC_FEATURES = tuple(sorted(set(FEATURE_NAMES) - _CONTEXT_FEATURES))


class FeatureVectorizer:
    """Convert the approved event-measurement schema into stable numeric vectors."""

    feature_schema_version = "1"

    def __init__(self, feature_names: Iterable[str] | None = None) -> None:
        names = tuple(feature_names) if feature_names is not None else _NUMERIC_FEATURES
        if not names or tuple(sorted(names)) != names or not set(names).issubset(_NUMERIC_FEATURES):
            raise ValueError("feature schema must be a sorted approved numeric feature subset")
        self.feature_names = names

    def transform_one(self, features: Mapping[str, Any]) -> np.ndarray:
        unknown = set(features) - set(self.feature_names) - _CONTEXT_FEATURES
        if unknown:
            raise ValueError(f"unapproved numeric feature: {min(unknown)}")
        values: list[float] = []
        for name in self.feature_names:
            value = features.get(name, 0.0)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise TypeError(f"feature {name} must be numeric")
            numeric = float(value)
            values.append(numeric if np.isfinite(numeric) else 0.0)
        return np.asarray(values, dtype=np.float64)

    def transform(self, rows: Iterable[Mapping[str, Any]]) -> np.ndarray:
        vectors = [self.transform_one(row) for row in rows]
        if not vectors:
            raise ValueError("at least one feature row is required")
        return np.vstack(vectors)


@dataclass(frozen=True)
class SimilarNeighbor:
    event_id: str
    distance: float


@dataclass(frozen=True)
class SimilarityResult:
    cluster_id: int
    neighbors: tuple[SimilarNeighbor, ...]


class SimilarityIndex:
    """Scaled HDBSCAN clusters plus nearest reviewed events; clusters are not labels."""

    def __init__(self) -> None:
        self._vectorizer = FeatureVectorizer()
        self._scaler = StandardScaler()
        self._rows: tuple[Any, ...] = ()
        self._scaled = np.empty((0, len(self._vectorizer.feature_names)))
        self._cluster_ids = np.empty(0, dtype=int)

    def fit(self, rows: Iterable[Any]) -> SimilarityIndex:
        self._rows = tuple(rows)
        if not self._rows:
            raise ValueError("at least one reviewed event is required")
        vectors = self._vectorizer.transform(row.features for row in self._rows)
        self._scaled = self._scaler.fit_transform(vectors)
        if len(self._rows) == 1:
            self._cluster_ids = np.array([-1], dtype=int)
        else:
            clusterer = HDBSCAN(min_cluster_size=min(5, len(self._rows)), allow_single_cluster=True, copy=False)
            self._cluster_ids = clusterer.fit_predict(self._scaled)
        return self

    def query(self, features: Mapping[str, Any], limit: int = 5) -> SimilarityResult:
        if not self._rows:
            raise ValueError("fit the similarity index before querying")
        if limit < 1:
            raise ValueError("limit must be positive")
        query = self._scaler.transform(self._vectorizer.transform_one(features).reshape(1, -1))
        distances = pairwise_distances(query, self._scaled, metric="euclidean")[0]
        indices = np.argsort(distances, kind="stable")[:limit]
        neighbors = tuple(SimilarNeighbor(str(self._rows[index].event_id), float(distances[index])) for index in indices)
        return SimilarityResult(int(self._cluster_ids[indices[0]]), neighbors)
