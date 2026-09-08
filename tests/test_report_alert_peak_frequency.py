from __future__ import annotations

import pytest

from src.evaluation.report_alert_peak_frequency import (
    build_alert_peak_frequency_summary,
    calculate_peak_alert_count,
    calculate_peak_frequency,
    count_peak_occurrences,
)


def test_calculate_peak_alert_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 2},
    ]

    assert calculate_peak_alert_count(history) == 5


def test_count_peak_occurrences():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 2},
    ]

    assert count_peak_occurrences(history) == 2


def test_calculate_peak_frequency():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 2},
    ]

    assert calculate_peak_frequency(history) == pytest.approx(2 / 5)


def test_peak_occurs_once():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert count_peak_occurrences(history) == 1
    assert calculate_peak_frequency(history) == pytest.approx(1 / 4)


def test_peak_occurs_every_snapshot():
    history = [
        {"alert_count": 4},
        {"alert_count": 4},
        {"alert_count": 4},
    ]

    assert calculate_peak_alert_count(history) == 4
    assert count_peak_occurrences(history) == 3
    assert calculate_peak_frequency(history) == 1.0


def test_all_zero_history():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_peak_alert_count(history) == 0
    assert count_peak_occurrences(history) == 3
    assert calculate_peak_frequency(history) == 1.0


def test_single_snapshot():
    history = [
        {"alert_count": 6},
    ]

    assert calculate_peak_alert_count(history) == 6
    assert count_peak_occurrences(history) == 1
    assert calculate_peak_frequency(history) == 1.0


def test_build_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 5},
        {"alert_count": 2},
    ]

    result = build_alert_peak_frequency_summary(history)

    assert result == {
        "snapshot_count": 5,
        "peak_alert_count": 5,
        "peak_occurrences": 2,
        "peak_frequency": pytest.approx(2 / 5),
    }


def test_build_summary_single_snapshot():
    history = [
        {"alert_count": 3},
    ]

    result = build_alert_peak_frequency_summary(history)

    assert result == {
        "snapshot_count": 1,
        "peak_alert_count": 3,
        "peak_occurrences": 1,
        "peak_frequency": 1.0,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_peak_alert_count([])

    with pytest.raises(ValueError):
        count_peak_occurrences([])

    with pytest.raises(ValueError):
        calculate_peak_frequency([])

    with pytest.raises(ValueError):
        build_alert_peak_frequency_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_peak_alert_count({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_peak_alert_count(
            [{"alert_count": 1}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_alert_count(
            [{"value": 1}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_alert_count(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_alert_count(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_peak_alert_count(
            [{"alert_count": -1}]
        )
