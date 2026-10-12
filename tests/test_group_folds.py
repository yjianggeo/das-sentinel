import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold


def _data(distributions):
    labels, groups = [], []
    for label, counts in zip(("earthquake", "blast", "traffic", "noise"), distributions, strict=True):
        for group, count in enumerate(counts):
            labels.extend([label] * count)
            groups.extend([f"day-{group}"] * count)
    return np.asarray(labels), np.asarray(groups)


def _check_folds(labels, groups, folds, n_splits):
    assert len(folds) == n_splits
    seen = []
    for train, test in folds:
        assert set(labels[train]) == set(labels[test]) == set(labels)
        assert not set(groups[train]) & set(groups[test])
        assert sorted(np.concatenate([train, test])) == list(range(len(labels)))
        seen.extend(test)
    assert sorted(seen) == list(range(len(labels)))


def test_feasible_rare_class_groups_are_covered_without_splitting_a_day():
    from das_sentinel.classifiers.group_folds import make_group_folds

    labels, groups = _data([
        [6, 6, 2, 1, 4, 5, 0, 6],
        [0, 2, 0, 12, 4, 3, 1, 0],
        [0, 6, 2, 3, 7, 7, 4, 1],
        [17, 2, 3, 0, 3, 0, 4, 1],
    ])
    initial = list(StratifiedGroupKFold(5, shuffle=True, random_state=20260726).split(
        np.zeros((len(labels), 1)), labels, groups
    ))
    assert any(set(labels[test]) != set(labels) for _, test in initial)

    first = make_group_folds(labels, groups)
    repeated = make_group_folds(labels, groups)
    reversed_rows = make_group_folds(labels[::-1], groups[::-1])

    _check_folds(labels, groups, first, 5)
    assert [set(groups[test]) for _, test in first] == [set(groups[test]) for _, test in repeated]
    assert [set(groups[test]) for _, test in first] == [set(groups[::-1][test]) for _, test in reversed_rows]


def test_an_already_valid_seeded_partition_is_preserved():
    from das_sentinel.classifiers.group_folds import make_group_folds

    labels, groups = _data([[4] * 5] * 4)
    expected = list(StratifiedGroupKFold(5, shuffle=True, random_state=20260726).split(
        np.zeros((len(labels), 1)), labels, groups
    ))

    actual = make_group_folds(labels, groups)

    for (train, test), (expected_train, expected_test) in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(train, expected_train)
        np.testing.assert_array_equal(test, expected_test)


def test_one_class_with_too_few_groups_cannot_be_repaired():
    from das_sentinel.classifiers.group_folds import make_group_folds

    labels, groups = _data([[4] * 5, [10, 10, 0, 0, 0], [4] * 5, [4] * 5])

    with pytest.raises(ValueError, match="class.*groups"):
        make_group_folds(labels, groups)


def test_group_counts_are_necessary_but_not_sufficient_for_class_coverage():
    from das_sentinel.classifiers.group_folds import make_group_folds

    # A-B, A-C and B-C must each land on opposite sides; a triangle cannot be
    # divided that way even though each class appears in two groups.
    with pytest.raises(ValueError, match="coverage"):
        make_group_folds(["a", "b", "a", "c", "b", "c"],
                         ["ab", "ab", "ac", "ac", "bc", "bc"], n_splits=2)


@pytest.mark.parametrize("labels, groups, n_splits", [
    ([], [], 5), (["a", "b"], ["one"], 2),
    ([["a"], ["b"]], ["one", "two"], 2),
    (["a", "b"], ["one", "two"], 1),
])
def test_invalid_split_inputs_are_rejected(labels, groups, n_splits):
    from das_sentinel.classifiers.group_folds import make_group_folds

    with pytest.raises(ValueError):
        make_group_folds(labels, groups, n_splits=n_splits)
