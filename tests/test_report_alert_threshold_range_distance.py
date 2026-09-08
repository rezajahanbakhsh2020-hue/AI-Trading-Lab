from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_range_distance import (
    build_range_distance_summary,
    calculate_mean_range_distance,
    calculate_range_distance,
    calculate_total_range_distance,
    find_farthest_range_positions,
)


def test_calculate_range_distance_below():
    assert calculate_range_distance(1, 3, 6) == 2


def test_calculate_range_distance_inside():
    assert calculate_range_distance(3, 3, 6) == 0
    assert calculate_range_distance(4, 3, 6) == 0
    assert calculate_range_distance(6, 3, 6) == 0


def test_calculate_range_distance_above():
    assert calculate_range_distance(9, 3, 6) == 3


def test_calculate_total_range_distance():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_total_range_distance(
        history,
        3,
        6,
    ) == 5


def test_calculate_mean_range_distance():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert calculate_mean_range_distance(
        history,
        3,
        6,
    ) == pytest.approx(1.0)


def test_find_farthest_range_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert find_farthest_range_positions(
        history,
        3,
        6,
    ) == [4]


def test_farthest_positions_include_ties():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 8},
        {"alert_count": 9},
    ]

    assert find_farthest_range_positions(
        history,
        3,
        7,
    ) == [0, 3]


def test_empty_history():
    assert calculate_total_range_distance(
        [],
        3,
        6,
    ) == 0

    assert calculate_mean_range_distance(
        [],
        3,
        6,
    ) == 0.0

    assert find_farthest_range_positions(
        [],
        3,
        6,
    ) == []

    assert build_range_distance_summary(
        [],
        3,
        6,
    ) == {
        "snapshot_count": 0,
        "lower_bound": 3,
        "upper_bound": 6,
        "total_distance": 0,
        "mean_distance": 0.0,
        "maximum_distance": 0,
        "farthest_positions": [],
    }


def test_zero_width_range():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    assert calculate_total_range_distance(
        history,
        4,
        4,
    ) == 2

    assert find_farthest_range_positions(
        history,
        4,
        4,
    ) == [0, 2]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 9},
    ]

    assert build_range_distance_summary(
        history,
        3,
        6,
    ) == {
        "snapshot_count": 5,
        "lower_bound": 3,
        "upper_bound": 6,
        "total_distance": 5,
        "mean_distance": 1.0,
        "maximum_distance": 3,
        "farthest_positions": [4],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_total_range_distance(
            {},
            3,
            6,
        )


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_total_range_distance(
            [{"alert_count": 3}, "invalid"],
            3,
            6,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"value": 3}],
            3,
            6,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"alert_count": 3.5}],
            3,
            6,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"alert_count": True}],
            3,
            6,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"alert_count": -1}],
            3,
            6,
        )


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            3.5,
            6,
        )


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            3,
            6.5,
        )


def test_boolean_lower_bound():
    with pytest.raises(TypeError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            True,
            6,
        )


def test_boolean_upper_bound():
    with pytest.raises(TypeError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            3,
            False,
        )


def test_negative_lower_bound():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            -1,
            6,
        )


def test_negative_upper_bound():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            3,
            -1,
        )


def test_lower_bound_greater_than_upper_bound():
    with pytest.raises(ValueError):
        calculate_total_range_distance(
            [{"alert_count": 4}],
            6,
            3,
        )


def test_direct_calculation_invalid_alert_count():
    with pytest.raises(TypeError):
        calculate_range_distance(
            4.5,
            3,
            6,
        )


def test_direct_calculation_boolean_alert_count():
    with pytest.raises(TypeError):
        calculate_range_distance(
            True,
            3,
            6,
        )


def test_direct_calculation_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_range_distance(
            -1,
            3,
            6,
        )
