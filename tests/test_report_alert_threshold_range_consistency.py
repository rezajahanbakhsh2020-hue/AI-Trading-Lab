from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_range_consistency import (
    build_consistency_summary,
    calculate_consistency_ratio,
    calculate_consistent_snapshot_count,
    calculate_inconsistency_ratio,
    calculate_inconsistent_snapshot_count,
    find_inconsistent_positions,
)


def test_consistent_snapshot_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_consistent_snapshot_count(
        history,
        3,
        6,
    ) == 3


def test_inconsistent_snapshot_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_inconsistent_snapshot_count(
        history,
        3,
        6,
    ) == 2


def test_consistency_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_consistency_ratio(
        history,
        3,
        6,
    ) == pytest.approx(3 / 5)


def test_inconsistency_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_inconsistency_ratio(
        history,
        3,
        6,
    ) == pytest.approx(2 / 5)


def test_find_inconsistent_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert find_inconsistent_positions(
        history,
        3,
        6,
    ) == [0, 4]


def test_all_snapshots_consistent():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_consistent_snapshot_count(
        history,
        3,
        6,
    ) == 3

    assert calculate_inconsistent_snapshot_count(
        history,
        3,
        6,
    ) == 0

    assert calculate_consistency_ratio(
        history,
        3,
        6,
    ) == 1.0

    assert calculate_inconsistency_ratio(
        history,
        3,
        6,
    ) == 0.0

    assert find_inconsistent_positions(
        history,
        3,
        6,
    ) == []


def test_all_snapshots_inconsistent():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 8},
    ]

    assert calculate_consistent_snapshot_count(
        history,
        3,
        6,
    ) == 0

    assert calculate_inconsistent_snapshot_count(
        history,
        3,
        6,
    ) == 4

    assert calculate_consistency_ratio(
        history,
        3,
        6,
    ) == 0.0

    assert calculate_inconsistency_ratio(
        history,
        3,
        6,
    ) == 1.0

    assert find_inconsistent_positions(
        history,
        3,
        6,
    ) == [0, 1, 2, 3]


def test_empty_history():
    assert calculate_consistent_snapshot_count(
        [],
        3,
        6,
    ) == 0

    assert calculate_inconsistent_snapshot_count(
        [],
        3,
        6,
    ) == 0

    assert calculate_consistency_ratio(
        [],
        3,
        6,
    ) == 0.0

    assert calculate_inconsistency_ratio(
        [],
        3,
        6,
    ) == 1.0

    assert find_inconsistent_positions(
        [],
        3,
        6,
    ) == []


def test_zero_width_range():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert calculate_consistent_snapshot_count(
        history,
        3,
        3,
    ) == 2

    assert calculate_inconsistent_snapshot_count(
        history,
        3,
        3,
    ) == 1

    assert calculate_consistency_ratio(
        history,
        3,
        3,
    ) == pytest.approx(2 / 3)

    assert find_inconsistent_positions(
        history,
        3,
        3,
    ) == [1]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert build_consistency_summary(
        history,
        3,
        6,
    ) == {
        "snapshot_count": 5,
        "lower_bound": 3,
        "upper_bound": 6,
        "consistent_snapshot_count": 3,
        "inconsistent_snapshot_count": 2,
        "consistency_ratio": pytest.approx(3 / 5),
        "inconsistency_ratio": pytest.approx(2 / 5),
        "inconsistent_positions": [0, 4],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_consistency_ratio(
            {},
            3,
            6,
        )


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_consistency_ratio(
            [{"alert_count": 3}, "invalid"],
            3,
            6,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"value": 3}],
            3,
            6,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"alert_count": 3.5}],
            3,
            6,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"alert_count": True}],
            3,
            6,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"alert_count": -1}],
            3,
            6,
        )


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            3.5,
            6,
        )


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            3,
            6.5,
        )


def test_boolean_lower_bound():
    with pytest.raises(TypeError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            True,
            6,
        )


def test_boolean_upper_bound():
    with pytest.raises(TypeError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            3,
            False,
        )


def test_negative_lower_bound():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            -1,
            6,
        )


def test_negative_upper_bound():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            3,
            -1,
        )


def test_lower_bound_greater_than_upper_bound():
    with pytest.raises(ValueError):
        calculate_consistency_ratio(
            [{"alert_count": 4}],
            6,
            3,
        )
