from __future__ import annotations

import pytest

from src.evaluation.report_alert_trend import (
    build_alert_trend_summary,
    calculate_alert_count_trend,
    calculate_alert_rate,
)


def test_calculate_alert_count_trend_increasing():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    result = calculate_alert_count_trend(history)

    assert result == {
        "first": 1.0,
        "last": 4.0,
        "change": 3.0,
        "direction": "increasing",
    }


def test_calculate_alert_count_trend_decreasing():
    history = [
        {"alert_count": 4},
        {"alert_count": 2},
        {"alert_count": 1},
    ]

    result = calculate_alert_count_trend(history)

    assert result == {
        "first": 4.0,
        "last": 1.0,
        "change": -3.0,
        "direction": "decreasing",
    }


def test_calculate_alert_count_trend_stable():
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 2},
    ]

    result = calculate_alert_count_trend(history)

    assert result == {
        "first": 2.0,
        "last": 2.0,
        "change": 0.0,
        "direction": "stable",
    }


def test_calculate_alert_count_trend_single_snapshot():
    result = calculate_alert_count_trend(
        [{"alert_count": 3}]
    )

    assert result == {
        "first": 3.0,
        "last": 3.0,
        "change": 0.0,
        "direction": "stable",
    }


def test_calculate_alert_count_trend_empty():
    with pytest.raises(ValueError):
        calculate_alert_count_trend([])


def test_calculate_alert_count_trend_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_count_trend({})


def test_calculate_alert_count_trend_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_count_trend(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_count_trend_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_count_trend(
            [{"level": "low"}]
        )


def test_calculate_alert_count_trend_rejects_non_numeric_count():
    with pytest.raises(ValueError):
        calculate_alert_count_trend(
            [{"alert_count": "invalid"}]
        )


def test_calculate_alert_count_trend_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_count_trend(
            [{"alert_count": -1}]
        )


def test_calculate_alert_rate():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    assert calculate_alert_rate(history) == 2.0


def test_calculate_alert_rate_zero():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_rate(history) == 0.0


def test_calculate_alert_rate_empty():
    with pytest.raises(ValueError):
        calculate_alert_rate([])


def test_calculate_alert_rate_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_rate({})


def test_calculate_alert_rate_rejects_invalid_count():
    with pytest.raises(ValueError):
        calculate_alert_rate(
            [{"alert_count": "invalid"}]
        )


def test_calculate_alert_rate_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_rate(
            [{"alert_count": -1}]
        )


def test_build_alert_trend_summary():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    result = build_alert_trend_summary(history)

    assert result == {
        "snapshot_count": 3,
        "trend": {
            "first": 1.0,
            "last": 4.0,
            "change": 3.0,
            "direction": "increasing",
        },
        "average_alert_rate": 2.3333333333,
    }


def test_build_alert_trend_summary_single_snapshot():
    history = [
        {"alert_count": 2},
    ]

    result = build_alert_trend_summary(history)

    assert result == {
        "snapshot_count": 1,
        "trend": {
            "first": 2.0,
            "last": 2.0,
            "change": 0.0,
            "direction": "stable",
        },
        "average_alert_rate": 2.0,
    }
