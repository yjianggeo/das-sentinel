from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.model_selection import StratifiedGroupKFold


def make_group_folds(
    labels: Sequence[Any],
    groups: Sequence[Any],
    *,
    n_splits: int = 5,
    seed: int = 20260726,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split labels by whole groups, requiring every class on both sides.

    Keep valid historical SGKF partitions. Its greedy stratification can miss
    feasible class coverage, so otherwise solve a bounded group-count problem;
    neither path examines features or prediction scores.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    if (labels.ndim != 1 or groups.ndim != 1 or not len(labels)
            or len(labels) != len(groups) or type(n_splits) is not int or n_splits < 2):
        raise ValueError("labels and groups must be matching nonempty vectors; n_splits must be at least two")
    _, label_ids = np.unique(labels, return_inverse=True)
    _, group_ids = np.unique(groups, return_inverse=True)
    counts = np.zeros((int(group_ids.max()) + 1, int(label_ids.max()) + 1), dtype=int)
    np.add.at(counts, (group_ids, label_ids), 1)
    if ((counts > 0).sum(axis=0) < n_splits).any():
        raise ValueError("each class must occur in at least n_splits distinct groups")

    initial = list(StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    ).split(np.zeros((len(labels), 1)), labels, groups))
    expected = set(label_ids)
    if all(set(label_ids[train]) == set(label_ids[test]) == expected
           for train, test in initial):
        return initial

    assignment = _balanced_group_assignment(counts, n_splits, seed)
    row_folds = assignment[group_ids]
    splits = [(np.flatnonzero(row_folds != fold), np.flatnonzero(row_folds == fold))
              for fold in range(n_splits)]
    if any(set(label_ids[train]) != expected or set(label_ids[test]) != expected
           or set(group_ids[train]) & set(group_ids[test]) for train, test in splits):
        raise ValueError("no group partition with complete class coverage was found")
    return splits


def _balanced_group_assignment(counts: np.ndarray, folds: int, seed: int) -> np.ndarray:
    group_count, class_count = counts.shape
    assigned = group_count * folds
    deviations = folds * class_count
    # Binary group/fold assignments plus absolute deviations from each class's
    # target fraction. Normalizing prevents the majority class dominating cost.
    variables = assigned + deviations
    matrix = lil_matrix((group_count + 3 * deviations, variables), dtype=float)
    lower = np.full(matrix.shape[0], -np.inf)
    upper = np.full(matrix.shape[0], np.inf)
    for group in range(group_count):
        matrix[group, group * folds:(group + 1) * folds] = 1
        lower[group] = upper[group] = 1
    normalized = counts / counts.sum(axis=0)
    for fold in range(folds):
        columns = np.arange(group_count) * folds + fold
        for label in range(class_count):
            offset = fold * class_count + label
            row = group_count + 3 * offset
            matrix[row, columns] = counts[:, label] > 0
            lower[row] = 1
            for sign, deviation_row in ((1, row + 1), (-1, row + 2)):
                matrix[deviation_row, columns] = sign * normalized[:, label]
                matrix[deviation_row, assigned + offset] = -1
                upper[deviation_row] = sign / folds
    variable_lower = np.zeros(variables)
    variable_upper = np.concatenate([np.ones(assigned), np.full(deviations, np.inf)])
    rare = np.flatnonzero((counts > 0).sum(axis=0) == folds)
    if len(rare):
        # With exactly K groups carrying a class, each must be in a separate
        # fold. Canonical labels remove fold-permutation symmetry, not solutions.
        for fold, group in enumerate(np.flatnonzero(counts[:, rare[0]])):
            variable_lower[group * folds + fold] = 1
    else:
        variable_lower[int(np.argmax(counts.sum(axis=1))) * folds] = 1
    cost = np.concatenate([
        np.random.default_rng(seed).uniform(0, 1e-7, assigned), np.ones(deviations)
    ])
    result = milp(
        cost,
        integrality=np.concatenate([np.ones(assigned), np.zeros(deviations)]),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsc(), lower, upper),
        options={"time_limit": 10.0, "node_limit": 10000, "mip_rel_gap": 1e-6},
    )
    if not result.success or result.x is None:
        raise ValueError("no group partition with complete class coverage was found within the search limit")
    allocation = result.x[:assigned].reshape(group_count, folds)
    if not np.allclose(allocation, np.round(allocation), atol=1e-6):
        raise ValueError("group coverage search returned a nonintegral partition")
    return allocation.argmax(axis=1)
