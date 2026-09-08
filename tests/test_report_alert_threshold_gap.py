from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_gap import (
    build_threshold_gap_summary,
    calculate_absolute_gap,
    calculate_average_absolute_gap,
    calculate_average_gap,
    calculate_gap,
    find_above_threshold_positions,
    find_below_threshold_positions,
)


def test_calculate_gap_above_threshold():
    assert calculate_gap(7, 4) == 3


def test_calculate_gap_at_threshold():
    assert calculate_gap(4, 4) == 0


def test_calculate_gap_below_threshold():
    assert calculate_gap(2, 4) == -2


def test_calculate_absolute_gap():
    assert calculate_absolute_gap(7, 4) == 3
    assert calculate_absolute_gap(2, 4) == 2
    assert calculate_absolute_gap(4, 4) == 0


def test_calculate_average_gap():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_average_gap(
        history,
        4,
    ) == pytest.approx(0.0)


def test_calculate_average_absolute_gap():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 8},
    ]

    assert calculate_average_absolute_gap(
        history,
        4,
    ) == pytest.approx(2.0)


def test_find_above_threshold_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 7},
    ]

    assert find_above_threshold_positions(
        history,
        4,
    ) == [1, 3]


def test_find_below_threshold_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 4},
        {"alert_count": 2},
    ]

    assert find_below_threshold_positions(
        history,
        4,
    ) == [0, 3]


def test_equal_threshold_is_in_neither_position_list():
    history = [
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    assert find_above_threshold_positions(
        history,
        4,
    ) == [1]

    assert find_below_threshold_positions(
        history,
        4,
    ) == [2]


def test_empty_history():
    assert calculate_average_gap([], 4) == 0.0
    assert calculate_average_absolute_gap([], 4) == 0.0
    assert find_above_threshold_positions([], 4) == []
    assert find_below_threshold_positions([], 4) == []

    assert build_threshold_gap_summary([], 4) == {
        "snapshot_count": 0,
        "threshold": 4,
        "average_gap": 0.0,
        "average_absolute_gap": 0.0,
        "above_threshold_positions": [],
        "below_threshold_positions": [],
    }


def test_zero_threshold():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_average_gap(
        history,
        0,
    ) == pytest.approx(7 / 3)

    assert find_above_threshold_positions(
        history,
        0,
    ) == [1, 2]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    result = build_threshold_gap_summary(
        history,
        4,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "average_gap": pytest.approx(0.0),
        "average_absolute_gap": pytest.approx(1.6),
        "above_threshold_positions": [1, 2],
        "below_threshold_positions": [0, 3],
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_average_gap({}, 4)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_average_gap(
            [{"alert_count": 5}, "invalid"],
            4,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_average_gap(
            [{"value": 5}],
            4,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_average_gap(
            [{"alert_count": 2.5}],
            4,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_average_gap(
            [{"alert_count": True}],
            4,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_average_gap(
            [{"alert_count": -1}],
            4,
        )


def test_invalid_threshold_type():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        calculate_average_gap(history, 4.5)


def test_boolean_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        calculate_average_gap(history, True)


def test_negative_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(ValueError):
        calculate_average_gap(history, -1)


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_gap(4.5, 3)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_gap(True, 3)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        calculate_gap(-1, 3)
