from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_distance import (
    build_threshold_distance_summary,
    calculate_average_distance,
    calculate_distance,
    calculate_total_distance,
    find_closest_positions,
    find_farthest_positions,
)


def test_calculate_distance_above_threshold():
    assert calculate_distance(7, 4) == 3


def test_calculate_distance_below_threshold():
    assert calculate_distance(2, 4) == 2


def test_calculate_distance_at_threshold():
    assert calculate_distance(4, 4) == 0


def test_calculate_total_distance():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert calculate_total_distance(
        history,
        4,
    ) == 8


def test_calculate_average_distance():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert calculate_average_distance(
        history,
        4,
    ) == pytest.approx(2.0)


def test_find_closest_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert find_closest_positions(
        history,
        4,
    ) == [2]


def test_find_multiple_closest_positions():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert find_closest_positions(
        history,
        4,
    ) == [2]


def test_find_farthest_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert find_farthest_positions(
        history,
        4,
    ) == [0, 2]


def test_find_multiple_farthest_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    assert find_farthest_positions(
        history,
        4,
    ) == [0, 1]


def test_empty_history():
    assert calculate_total_distance([], 4) == 0
    assert calculate_average_distance([], 4) == 0.0
    assert find_closest_positions([], 4) == []
    assert find_farthest_positions([], 4) == []

    assert build_threshold_distance_summary([], 4) == {
        "snapshot_count": 0,
        "threshold": 4,
        "total_distance": 0,
        "average_distance": 0.0,
        "closest_positions": [],
        "farthest_positions": [],
    }


def test_zero_threshold():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_total_distance(
        history,
        0,
    ) == 7

    assert find_closest_positions(
        history,
        0,
    ) == [0]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    result = build_threshold_distance_summary(
        history,
        4,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "total_distance": 8,
        "average_distance": pytest.approx(1.6),
        "closest_positions": [4],
        "farthest_positions": [0, 2],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_total_distance({}, 4)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_total_distance(
            [{"alert_count": 5}, "invalid"],
            4,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_total_distance(
            [{"value": 5}],
            4,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_total_distance(
            [{"alert_count": 2.5}],
            4,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_total_distance(
            [{"alert_count": True}],
            4,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_total_distance(
            [{"alert_count": -1}],
            4,
        )


def test_invalid_threshold_type():
    with pytest.raises(TypeError):
        calculate_total_distance(
            [{"alert_count": 5}],
            4.5,
        )


def test_boolean_threshold():
    with pytest.raises(TypeError):
        calculate_total_distance(
            [{"alert_count": 5}],
            True,
        )


def test_negative_threshold():
    with pytest.raises(ValueError):
        calculate_total_distance(
            [{"alert_count": 5}],
            -1,
        )


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_distance(4.5, 3)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_distance(True, 3)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        calculate_distance(-1, 3)
