from __future__ import annotations

import pytest

from src.evaluation.report_alert_inactive_duration import (
    build_alert_inactive_duration_summary,
    calculate_average_inactive_duration,
    calculate_inactive_durations,
    calculate_longest_inactive_duration,
)


def test_calculate_inactive_durations():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_inactive_durations(history) == [2, 1, 3]


def test_calculate_average_inactive_duration():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_average_inactive_duration(history) == pytest.approx(
        2.0
    )


def test_calculate_longest_inactive_duration():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_longest_inactive_duration(history) == 3


def test_all_inactive():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_inactive_durations(history) == [3]
    assert calculate_average_inactive_duration(history) == 3.0
    assert calculate_longest_inactive_duration(history) == 3


def test_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_inactive_durations(history) == []
    assert calculate_average_inactive_duration(history) == 0.0
    assert calculate_longest_inactive_duration(history) == 0


def test_single_inactive_snapshot():
    history = [{"alert_count": 0}]

    assert calculate_inactive_durations(history) == [1]
    assert calculate_average_inactive_duration(history) == 1.0
    assert calculate_longest_inactive_duration(history) == 1


def test_inactive_period_at_start():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_inactive_durations(history) == [2]


def test_inactive_period_at_end():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_inactive_durations(history) == [3]


def test_build_summary():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_inactive_duration_summary(history)

    assert result == {
        "snapshot_count": 8,
        "inactive_period_count": 3,
        "inactive_durations": [2, 1, 3],
        "average_inactive_duration": pytest.approx(2.0),
        "longest_inactive_duration": 3,
    }


def test_build_summary_all_active():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    result = build_alert_inactive_duration_summary(history)

    assert result == {
        "snapshot_count": 2,
        "inactive_period_count": 0,
        "inactive_durations": [],
        "average_inactive_duration": 0.0,
        "longest_inactive_duration": 0,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_inactive_durations([])

    with pytest.raises(ValueError):
        build_alert_inactive_duration_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_inactive_durations({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_inactive_durations(
            [{"alert_count": 0}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_inactive_durations(
            [{"value": 0}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_inactive_durations(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_inactive_durations(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_inactive_durations(
            [{"alert_count": -1}]
        )
