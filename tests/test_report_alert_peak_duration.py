from __future__ import annotations

import pytest

from src.evaluation.report_alert_peak_duration import (
    build_alert_peak_duration_summary,
    calculate_average_peak_duration,
    calculate_longest_peak_duration,
    calculate_peak_alert_count,
    calculate_peak_durations,
)


def test_calculate_peak_alert_count():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 1},
    ]

    assert calculate_peak_alert_count(history) == 5


def test_calculate_peak_durations():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_peak_durations(history) == [2, 3, 1]


def test_calculate_longest_peak_duration():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_longest_peak_duration(history) == 3


def test_calculate_average_peak_duration():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    assert calculate_average_peak_duration(history) == pytest.approx(
        2.0
    )


def test_peak_at_start():
    history = [
        {"alert_count": 7},
        {"alert_count": 7},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    assert calculate_peak_durations(history) == [2]


def test_peak_at_end():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 7},
    ]

    assert calculate_peak_durations(history) == [2]


def test_peak_occurs_everywhere():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert calculate_peak_durations(history) == [3]
    assert calculate_longest_peak_duration(history) == 3
    assert calculate_average_peak_duration(history) == 3.0


def test_peak_occurs_once():
    history = [
        {"alert_count": 1},
        {"alert_count": 8},
        {"alert_count": 2},
    ]

    assert calculate_peak_durations(history) == [1]
    assert calculate_longest_peak_duration(history) == 1
    assert calculate_average_peak_duration(history) == 1.0


def test_all_zero_history():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_peak_alert_count(history) == 0
    assert calculate_peak_durations(history) == [3]
    assert calculate_longest_peak_duration(history) == 3
    assert calculate_average_peak_duration(history) == 3.0


def test_single_snapshot():
    history = [
        {"alert_count": 6},
    ]

    assert calculate_peak_durations(history) == [1]
    assert calculate_longest_peak_duration(history) == 1
    assert calculate_average_peak_duration(history) == 1.0


def test_build_summary():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 1},
        {"alert_count": 5},
    ]

    result = build_alert_peak_duration_summary(history)

    assert result == {
        "snapshot_count": 8,
        "peak_alert_count": 5,
        "peak_period_count": 3,
        "peak_durations": [2, 3, 1],
        "average_peak_duration": pytest.approx(2.0),
        "longest_peak_duration": 3,
    }


def test_build_summary_single_snapshot():
    history = [
        {"alert_count": 3},
    ]

    result = build_alert_peak_duration_summary(history)

    assert result == {
        "snapshot_count": 1,
        "peak_alert_count": 3,
        "peak_period_count": 1,
        "peak_durations": [1],
        "average_peak_duration": 1.0,
        "longest_peak_duration": 1,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_peak_durations([])

    with pytest.raises(ValueError):
        build_alert_peak_duration_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_peak_durations({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_peak_durations(
            [{"alert_count": 5}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_durations(
            [{"value": 5}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_durations(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_durations(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_durations(
            [{"alert_count": -1}]
        )
