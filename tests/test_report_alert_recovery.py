from __future__ import annotations

import pytest

from src.evaluation.report_alert_recovery import (
    build_alert_recovery_summary,
    calculate_average_recovery_duration,
    calculate_fastest_recovery_duration,
    calculate_recovery_durations,
    calculate_slowest_recovery_duration,
    calculate_total_recovery_duration,
    count_recovered_alert_periods,
)


def test_calculate_recovery_durations():
    history = [
        {"alert_count": 2},
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 3},
        {"alert_count": 2},
        {"alert_count": 1},
        {"alert_count": 0},
    ]

    assert calculate_recovery_durations(history) == [2, 4]


def test_calculate_recovery_durations_empty():
    assert calculate_recovery_durations([]) == []


def test_calculate_recovery_durations_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_recovery_durations(history) == []


def test_calculate_recovery_durations_single_snapshot_recovery():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
    ]

    assert calculate_recovery_durations(history) == [1]


def test_calculate_recovery_durations_unresolved_alert():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_recovery_durations(history) == []


def test_calculate_recovery_durations_mixed_completed_and_unresolved():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_recovery_durations(history) == [1]


def test_calculate_recovery_durations_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_recovery_durations({})


def test_calculate_recovery_durations_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_recovery_durations(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_recovery_durations_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_recovery_durations(
            [{"value": 1}]
        )


def test_calculate_recovery_durations_rejects_non_integer_count():
    with pytest.raises(ValueError):
        calculate_recovery_durations(
            [{"alert_count": 1.5}]
        )


def test_calculate_recovery_durations_rejects_boolean_count():
    with pytest.raises(ValueError):
        calculate_recovery_durations(
            [{"alert_count": True}]
        )


def test_calculate_recovery_durations_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_recovery_durations(
            [{"alert_count": -1}]
        )


def test_calculate_total_recovery_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    assert calculate_total_recovery_duration(history) == 3


def test_calculate_total_recovery_duration_no_recovery():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_total_recovery_duration(history) == 0


def test_calculate_average_recovery_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    assert calculate_average_recovery_duration(history) == 1.5


def test_calculate_average_recovery_duration_no_recovery():
    assert calculate_average_recovery_duration(
        [{"alert_count": 1}]
    ) == 0.0


def test_calculate_fastest_recovery_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
    ]

    assert calculate_fastest_recovery_duration(history) == 1


def test_calculate_fastest_recovery_duration_no_recovery():
    assert calculate_fastest_recovery_duration(
        [{"alert_count": 1}]
    ) == 0


def test_calculate_slowest_recovery_duration():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    assert calculate_slowest_recovery_duration(history) == 3


def test_calculate_slowest_recovery_duration_no_recovery():
    assert calculate_slowest_recovery_duration(
        [{"alert_count": 1}]
    ) == 0


def test_count_recovered_alert_periods():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
    ]

    assert count_recovered_alert_periods(history) == 2


def test_count_recovered_alert_periods_empty():
    assert count_recovered_alert_periods([]) == 0


def test_build_alert_recovery_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 0},
        {"alert_count": 4},
    ]

    result = build_alert_recovery_summary(history)

    assert result == {
        "snapshot_count": 6,
        "recovered_period_count": 2,
        "recovery_durations": [1, 3],
        "total_recovery_duration": 4,
        "average_recovery_duration": 2.0,
        "fastest_recovery_duration": 1,
        "slowest_recovery_duration": 3,
    }


def test_build_alert_recovery_summary_empty():
    result = build_alert_recovery_summary([])

    assert result == {
        "snapshot_count": 0,
        "recovered_period_count": 0,
        "recovery_durations": [],
        "total_recovery_duration": 0,
        "average_recovery_duration": 0.0,
        "fastest_recovery_duration": 0,
        "slowest_recovery_duration": 0,
    }
