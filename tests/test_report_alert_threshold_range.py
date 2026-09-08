from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_range import (
    build_threshold_range_summary,
    calculate_range_coverage,
    calculate_range_width,
    count_alerts_in_range,
    find_alert_range_positions,
    is_alert_count_in_range,
)


def test_is_alert_count_in_range():
    assert is_alert_count_in_range(4, 2, 6) is True
    assert is_alert_count_in_range(2, 2, 6) is True
    assert is_alert_count_in_range(6, 2, 6) is True


def test_is_alert_count_in_range_outside():
    assert is_alert_count_in_range(1, 2, 6) is False
    assert is_alert_count_in_range(7, 2, 6) is False


def test_calculate_range_width():
    assert calculate_range_width(2, 6) == 4
    assert calculate_range_width(4, 4) == 0


def test_count_alerts_in_range():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert count_alerts_in_range(
        history,
        2,
        6,
    ) == 3


def test_find_alert_range_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert find_alert_range_positions(
        history,
        2,
        6,
    ) == [1, 2, 3]


def test_calculate_range_coverage():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert calculate_range_coverage(
        history,
        2,
        6,
    ) == pytest.approx(3 / 5)


def test_empty_history():
    assert count_alerts_in_range([], 2, 6) == 0
    assert find_alert_range_positions([], 2, 6) == []
    assert calculate_range_coverage([], 2, 6) == 0.0


def test_single_point_range():
    history = [
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    assert count_alerts_in_range(
        history,
        3,
        3,
    ) == 2

    assert find_alert_range_positions(
        history,
        3,
        3,
    ) == [0, 2]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    assert build_threshold_range_summary(
        history,
        2,
        6,
    ) == {
        "snapshot_count": 5,
        "lower_bound": 2,
        "upper_bound": 6,
        "range_width": 4,
        "in_range_count": 3,
        "out_of_range_count": 2,
        "coverage": pytest.approx(3 / 5),
        "in_range_positions": [1, 2, 3],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        count_alerts_in_range({}, 2, 6)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        count_alerts_in_range(
            [{"alert_count": 4}, "invalid"],
            2,
            6,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"value": 4}],
            2,
            6,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"alert_count": 2.5}],
            2,
            6,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"alert_count": True}],
            2,
            6,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"alert_count": -1}],
            2,
            6,
        )


def test_invalid_lower_bound_type():
    with pytest.raises(TypeError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            2.5,
            6,
        )


def test_invalid_upper_bound_type():
    with pytest.raises(TypeError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            2,
            6.5,
        )


def test_boolean_lower_bound():
    with pytest.raises(TypeError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            True,
            6,
        )


def test_boolean_upper_bound():
    with pytest.raises(TypeError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            2,
            False,
        )


def test_negative_lower_bound():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            -1,
            6,
        )


def test_negative_upper_bound():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            2,
            -1,
        )


def test_lower_bound_greater_than_upper_bound():
    with pytest.raises(ValueError):
        count_alerts_in_range(
            [{"alert_count": 4}],
            6,
            2,
        )


def test_direct_calculation_invalid_alert_count():
    with pytest.raises(TypeError):
        is_alert_count_in_range(
            4.5,
            2,
            6,
        )


def test_direct_calculation_boolean_alert_count():
    with pytest.raises(TypeError):
        is_alert_count_in_range(
            True,
            2,
            6,
        )


def test_direct_calculation_negative_alert_count():
    with pytest.raises(ValueError):
        is_alert_count_in_range(
            -1,
            2,
            6,
        )
