from __future__ import annotations

import pytest

from src.evaluation.report_alert_peak import (
    build_alert_peak_summary,
    calculate_peak_alert_count,
    calculate_peak_alert_excess,
)


def test_calculate_peak_alert_count():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 2},
        {"alert_count": 7},
        {"alert_count": 3},
    ]

    assert calculate_peak_alert_count(history) == 7


def test_calculate_peak_alert_count_all_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_peak_alert_count(history) == 0


def test_calculate_peak_alert_count_single_snapshot():
    history = [
        {"alert_count": 5},
    ]

    assert calculate_peak_alert_count(history) == 5


def test_calculate_peak_alert_excess():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 2},
        {"alert_count": 7},
    ]

    assert calculate_peak_alert_excess(history, 5) == 2


def test_calculate_peak_alert_excess_at_threshold():
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    assert calculate_peak_alert_excess(history, 5) == 0


def test_calculate_peak_alert_excess_below_threshold():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    assert calculate_peak_alert_excess(history, 5) == 0


def test_build_alert_peak_summary_without_threshold():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 2},
    ]

    result = build_alert_peak_summary(history)

    assert result == {
        "snapshot_count": 4,
        "peak_alert_count": 7,
    }


def test_build_alert_peak_summary_with_threshold():
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 7},
        {"alert_count": 2},
    ]

    result = build_alert_peak_summary(history, threshold=5)

    assert result == {
        "snapshot_count": 4,
        "peak_alert_count": 7,
        "threshold": 5,
        "peak_alert_excess": 2,
    }


def test_build_alert_peak_summary_threshold_not_exceeded():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = build_alert_peak_summary(history, threshold=5)

    assert result == {
        "snapshot_count": 3,
        "peak_alert_count": 3,
        "threshold": 5,
        "peak_alert_excess": 0,
    }


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_peak_alert_count([])


def test_build_summary_empty_history():
    with pytest.raises(ValueError):
        build_alert_peak_summary([])


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


def test_invalid_threshold_type():
    history = [{"alert_count": 3}]

    with pytest.raises(ValueError):
        calculate_peak_alert_excess(history, 2.5)


def test_boolean_threshold():
    history = [{"alert_count": 3}]

    with pytest.raises(ValueError):
        calculate_peak_alert_excess(history, True)


def test_negative_threshold():
    history = [{"alert_count": 3}]

    with pytest.raises(ValueError):
        calculate_peak_alert_excess(history, -1)
