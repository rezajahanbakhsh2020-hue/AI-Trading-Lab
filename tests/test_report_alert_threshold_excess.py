from __future__ import annotations

import pytest

from src.evaluation.report_alert_threshold_excess import (
    build_threshold_excess_summary,
    calculate_average_excess,
    calculate_excess,
    calculate_max_excess,
    calculate_total_excess,
    find_excess_positions,
)


def test_calculate_excess_above_threshold():
    assert calculate_excess(7, 4) == 3


def test_calculate_excess_at_threshold():
    assert calculate_excess(4, 4) == 0


def test_calculate_excess_below_threshold():
    assert calculate_excess(2, 4) == 0


def test_calculate_total_excess():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert calculate_total_excess(history, 4) == 4


def test_calculate_average_excess():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert calculate_average_excess(
        history,
        4,
    ) == pytest.approx(1.0)


def test_find_excess_positions():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 7},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert find_excess_positions(
        history,
        4,
    ) == [1, 2, 4]


def test_calculate_max_excess():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 9},
        {"alert_count": 4},
    ]

    assert calculate_max_excess(
        history,
        4,
    ) == 5


def test_no_excess():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert calculate_total_excess(history, 4) == 0
    assert calculate_average_excess(history, 4) == 0.0
    assert find_excess_positions(history, 4) == []
    assert calculate_max_excess(history, 4) == 0


def test_every_observation_has_excess():
    history = [
        {"alert_count": 5},
        {"alert_count": 6},
        {"alert_count": 8},
    ]

    assert calculate_total_excess(history, 4) == 7
    assert calculate_average_excess(
        history,
        4,
    ) == pytest.approx(7 / 3)
    assert find_excess_positions(
        history,
        4,
    ) == [0, 1, 2]
    assert calculate_max_excess(history, 4) == 4


def test_empty_history():
    assert calculate_total_excess([], 4) == 0
    assert calculate_average_excess([], 4) == 0.0
    assert find_excess_positions([], 4) == []
    assert calculate_max_excess([], 4) == 0

    assert build_threshold_excess_summary([], 4) == {
        "snapshot_count": 0,
        "threshold": 4,
        "total_excess": 0,
        "average_excess": 0.0,
        "excess_positions": [],
        "max_excess": 0,
    }


def test_zero_threshold():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    assert calculate_total_excess(history, 0) == 7
    assert find_excess_positions(history, 0) == [1, 2]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 8},
        {"alert_count": 2},
        {"alert_count": 6},
    ]

    result = build_threshold_excess_summary(
        history,
        4,
    )

    assert result == {
        "snapshot_count": 5,
        "threshold": 4,
        "total_excess": 7,
        "average_excess": pytest.approx(1.4),
        "excess_positions": [1, 2, 4],
        "max_excess": 4,
    }


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_total_excess({}, 4)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_total_excess(
            [{"alert_count": 5}, "invalid"],
            4,
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_total_excess(
            [{"value": 5}],
            4,
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_total_excess(
            [{"alert_count": 2.5}],
            4,
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_total_excess(
            [{"alert_count": True}],
            4,
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_total_excess(
            [{"alert_count": -1}],
            4,
        )


def test_invalid_threshold_type():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        calculate_total_excess(history, 4.5)


def test_boolean_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(TypeError):
        calculate_total_excess(history, True)


def test_negative_threshold():
    history = [{"alert_count": 5}]

    with pytest.raises(ValueError):
        calculate_total_excess(history, -1)


def test_invalid_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_excess(4.5, 3)


def test_boolean_alert_count_for_direct_calculation():
    with pytest.raises(TypeError):
        calculate_excess(True, 3)


def test_negative_alert_count_for_direct_calculation():
    with pytest.raises(ValueError):
        calculate_excess(-1, 3)
