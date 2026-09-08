from __future__ import annotations

import pytest

from src.evaluation.report_alert_frequency import (
    build_alert_frequency_summary,
    calculate_alert_event_frequency,
    calculate_alert_frequency,
    count_alert_events,
)


def test_count_alert_events():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert count_alert_events(history) == 6


def test_count_alert_events_empty():
    assert count_alert_events([]) == 0


def test_calculate_alert_frequency():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    assert calculate_alert_frequency(history) == 6 / 5


def test_calculate_alert_frequency_empty():
    with pytest.raises(ValueError):
        calculate_alert_frequency([])


def test_calculate_alert_event_frequency():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    assert calculate_alert_event_frequency(history) == 3.0


def test_calculate_alert_event_frequency_no_alerts():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_event_frequency(history) == 0.0


def test_build_alert_frequency_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 3},
    ]

    result = build_alert_frequency_summary(history)

    assert result == {
        "snapshot_count": 5,
        "total_alert_events": 6,
        "alert_frequency": 6 / 5,
        "alerts_per_snapshot": 6 / 5,
    }


def test_build_alert_frequency_summary_single_snapshot():
    history = [
        {"alert_count": 4},
    ]

    result = build_alert_frequency_summary(history)

    assert result == {
        "snapshot_count": 1,
        "total_alert_events": 4,
        "alert_frequency": 4.0,
        "alerts_per_snapshot": 4.0,
    }


def test_build_alert_frequency_summary_empty():
    with pytest.raises(ValueError):
        build_alert_frequency_summary([])


def test_rejects_invalid_history_type():
    with pytest.raises(TypeError):
        count_alert_events({})


def test_rejects_invalid_history_item():
    with pytest.raises(TypeError):
        count_alert_events(
            [{"alert_count": 1}, "invalid"]
        )


def test_rejects_missing_alert_count():
    with pytest.raises(ValueError):
        count_alert_events(
            [{"value": 1}]
        )


def test_rejects_non_integer_alert_count():
    with pytest.raises(ValueError):
        count_alert_events(
            [{"alert_count": 1.5}]
        )


def test_rejects_boolean_alert_count():
    with pytest.raises(ValueError):
        count_alert_events(
            [{"alert_count": True}]
        )


def test_rejects_negative_alert_count():
    with pytest.raises(ValueError):
        count_alert_events(
            [{"alert_count": -1}]
        )
