from __future__ import annotations

import pytest

from src.evaluation.report_alert_duration import (
    build_alert_duration_summary,
    calculate_alert_durations,
    calculate_average_alert_duration,
    calculate_longest_alert_duration,
    calculate_total_alert_duration,
)


def test_calculate_alert_durations():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
        {"alert_count": 1},
    ]

    assert calculate_alert_durations(history) == [2, 2, 1]


def test_calculate_alert_durations_empty():
    assert calculate_alert_durations([]) == []


def test_calculate_alert_durations_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_durations(history) == []


def test_calculate_alert_durations_all_alerts():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_durations(history) == [3]


def test_calculate_alert_durations_single_alert():
    assert calculate_alert_durations(
        [{"alert_count": 1}]
    ) == [1]


def test_calculate_alert_durations_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_durations({})


def test_calculate_alert_durations_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_durations(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_durations_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_durations(
            [{"value": 1}]
        )


def test_calculate_alert_durations_rejects_non_integer_count():
    with pytest.raises(ValueError):
        calculate_alert_durations(
            [{"alert_count": 1.5}]
        )


def test_calculate_alert_durations_rejects_boolean_count():
    with pytest.raises(ValueError):
        calculate_alert_durations(
            [{"alert_count": True}]
        )


def test_calculate_alert_durations_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_durations(
            [{"alert_count": -1}]
        )


def test_calculate_total_alert_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    assert calculate_total_alert_duration(history) == 3


def test_calculate_total_alert_duration_empty():
    assert calculate_total_alert_duration([]) == 0


def test_calculate_longest_alert_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 5},
        {"alert_count": 0},
    ]

    assert calculate_longest_alert_duration(history) == 3


def test_calculate_longest_alert_duration_no_alerts():
    assert calculate_longest_alert_duration(
        [{"alert_count": 0}]
    ) == 0


def test_calculate_average_alert_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_average_alert_duration(history) == 1.5


def test_calculate_average_alert_duration_no_alerts():
    assert calculate_average_alert_duration(
        [{"alert_count": 0}]
    ) == 0.0


def test_build_alert_duration_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    result = build_alert_duration_summary(history)

    assert result == {
        "snapshot_count": 7,
        "alert_period_count": 3,
        "durations": [2, 1, 2],
        "total_alert_duration": 5,
        "longest_alert_duration": 2,
        "average_alert_duration": 5 / 3,
    }


def test_build_alert_duration_summary_empty():
    result = build_alert_duration_summary([])

    assert result == {
        "snapshot_count": 0,
        "alert_period_count": 0,
        "durations": [],
        "total_alert_duration": 0,
        "longest_alert_duration": 0,
        "average_alert_duration": 0.0,
    }
