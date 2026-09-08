from __future__ import annotations

import pytest

from src.evaluation.report_alert_occupancy_duration import (
    build_alert_occupancy_duration_summary,
    calculate_active_durations,
    calculate_average_active_duration,
    calculate_longest_active_duration,
)


def test_calculate_active_durations():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    assert calculate_active_durations(history) == [2, 2, 1]


def test_calculate_average_active_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    assert calculate_average_active_duration(history) == pytest.approx(
        5 / 3
    )


def test_calculate_longest_active_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    assert calculate_longest_active_duration(history) == 2


def test_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_active_durations(history) == [3]
    assert calculate_average_active_duration(history) == 3.0
    assert calculate_longest_active_duration(history) == 3


def test_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_active_durations(history) == []
    assert calculate_average_active_duration(history) == 0.0
    assert calculate_longest_active_duration(history) == 0


def test_single_active_snapshot():
    history = [{"alert_count": 5}]

    assert calculate_active_durations(history) == [1]
    assert calculate_average_active_duration(history) == 1.0
    assert calculate_longest_active_duration(history) == 1


def test_active_period_at_start():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_active_durations(history) == [2]


def test_active_period_at_end():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_active_durations(history) == [3]


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 5},
    ]

    result = build_alert_occupancy_duration_summary(history)

    assert result == {
        "snapshot_count": 7,
        "active_period_count": 3,
        "active_durations": [2, 2, 1],
        "average_active_duration": pytest.approx(5 / 3),
        "longest_active_duration": 2,
    }


def test_build_summary_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_occupancy_duration_summary(history)

    assert result == {
        "snapshot_count": 2,
        "active_period_count": 0,
        "active_durations": [],
        "average_active_duration": 0.0,
        "longest_active_duration": 0,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_active_durations([])

    with pytest.raises(ValueError):
        build_alert_occupancy_duration_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_active_durations({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_active_durations(
            [{"alert_count": 1}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_active_durations(
            [{"value": 1}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_active_durations(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_active_durations(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_active_durations(
            [{"alert_count": -1}]
        )
