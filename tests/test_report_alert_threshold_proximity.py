from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_proximity import (
    build_threshold_proximity_summary,
    calculate_distance,
    calculate_proximity_ratio,
    count_within_tolerance,
    find_outside_tolerance_positions,
    find_within_tolerance_positions,
    is_within_tolerance,
)


def test_calculate_distance():
    assert calculate_distance(7, 4) == 3
    assert calculate_distance(2, 4) == 2
    assert calculate_distance(4, 4) == 0


def test_is_within_tolerance():
    assert is_within_tolerance(5, 4, 1) is True
    assert is_within_tolerance(3, 4, 1) is True
    assert is_within_tolerance(6, 4, 1) is False


def test_zero_tolerance():
    assert is_within_tolerance(4, 4, 0) is True
    assert is_within_tolerance(5, 4, 0) is False


def test_count_within_tolerance():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    assert count_within_tolerance(
        history,
        4,
        1,
    ) == 3


def test_calculate_proximity_ratio():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    assert calculate_proximity_ratio(
        history,
        4,
        1,
    ) == pytest.approx(0.6)


def test_find_within_tolerance_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    assert find_within_tolerance_positions(
        history,
        4,
        1,
    ) == [1, 2, 3]


def test_find_outside_tolerance_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    assert find_outside_tolerance_positions(
        history,
        4,
        1,
    ) == [0, 4]


def test_empty_history():
    assert count_within_tolerance([], 4, 1) == 0
    assert calculate_proximity_ratio([], 4, 1) == 0.0
    assert find_within_tolerance_positions([], 4, 1) == []
    assert find_outside_tolerance_positions([], 4, 1) == []

    assert build_threshold_proximity_summary(
        [],
        4,
        1,
    ) == {
        "snapshot_count": 0,
        "threshold": 4,
        "tolerance": 1,
        "within_tolerance_count": 0,
        "proximity_ratio": 0.0,
        "within_tolerance_positions": [],
        "outside_tolerance_positions": [],
    }


def test_zero_threshold():
    history = [
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 3},
    ]

    assert count_within_tolerance(
        history,
        0,
        1,
    ) == 2

    assert find_within_tolerance_positions(
        history,
        0,
        1,
    ) == [0, 1]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    result = build_threshold_proximity_summary(
        history,
        4,
        1,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "tolerance": 1,
        "within_tolerance_count": 3,
        "proximity_ratio": pytest.approx(0.6),
        "within_tolerance_positions": [1, 2, 3],
        "outside_tolerance_positions": [0, 4],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_within_tolerance({}, 4, 1)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_within_tolerance(
            [{"alert_count": 4}, "invalid"],
            4,
            1,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_within_tolerance(
            [{"value": 4}],
            4,
            1,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_within_tolerance(
            [{"alert_count": 2.5}],
            4,
            1,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_within_tolerance(
            [{"alert_count": True}],
            4,
            1,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_within_tolerance(
            [{"alert_count": -1}],
            4,
            1,
        )


def test_invalid_threshold_type():
    with pytest.raises(TypeError):
        count_within_tolerance(
            [{"alert_count": 4}],
            4.5,
            1,
        )


def test_boolean_threshold():
    with pytest.raises(TypeError):
        count_within_tolerance(
            [{"alert_count": 4}],
            True,
            1,
        )


def test_negative_threshold():
    with pytest.raises(ValueError):
        count_within_tolerance(
            [{"alert_count": 4}],
            -1,
            1,
        )


def test_invalid_tolerance_type():
    with pytest.raises(TypeError):
        count_within_tolerance(
            [{"alert_count": 4}],
            4,
            1.5,
        )


def test_boolean_tolerance():
    with pytest.raises(TypeError):
        count_within_tolerance(
            [{"alert_count": 4}],
            4,
            True,
        )


def test_negative_tolerance():
    with pytest.raises(ValueError):
        count_within_tolerance(
            [{"alert_count": 4}],
            4,
            -1,
        )


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_distance(4.5, 4)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_distance(True, 4)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        calculate_distance(-1, 4)
