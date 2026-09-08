from __future__ import annotations

import pytest

from src.evaluation.report_alert_statistics import (
    build_alert_statistics_summary,
    calculate_alert_free_ratio,
    calculate_alert_statistics,
    calculate_zero_alert_snapshots,
)


def test_calculate_alert_statistics():
    history = [
        {"alert_count": 1},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = calculate_alert_statistics(history)

    assert result == {
        "count": 3.0,
        "minimum": 1.0,
        "maximum": 3.0,
        "mean": 2.0,
        "range": 2.0,
    }


def test_calculate_alert_statistics_single_snapshot():
    result = calculate_alert_statistics(
        [{"alert_count": 4}]
    )

    assert result == {
        "count": 1.0,
        "minimum": 4.0,
        "maximum": 4.0,
        "mean": 4.0,
        "range": 0.0,
    }


def test_calculate_alert_statistics_empty():
    with pytest.raises(ValueError):
        calculate_alert_statistics([])


def test_calculate_alert_statistics_rejects_invalid_history():
    with pytest.raises(TypeError):
        calculate_alert_statistics({})


def test_calculate_alert_statistics_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_alert_statistics(
            [{"alert_count": 1}, "invalid"]
        )


def test_calculate_alert_statistics_rejects_missing_count():
    with pytest.raises(ValueError):
        calculate_alert_statistics(
            [{"level": "low"}]
        )


def test_calculate_alert_statistics_rejects_non_numeric_count():
    with pytest.raises(ValueError):
        calculate_alert_statistics(
            [{"alert_count": "invalid"}]
        )


def test_calculate_alert_statistics_rejects_negative_count():
    with pytest.raises(ValueError):
        calculate_alert_statistics(
            [{"alert_count": -1}]
        )


def test_calculate_zero_alert_snapshots():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 1},
    ]

    assert calculate_zero_alert_snapshots(history) == 2


def test_calculate_zero_alert_snapshots_empty():
    assert calculate_zero_alert_snapshots([]) == 0


def test_calculate_zero_alert_snapshots_rejects_invalid_item():
    with pytest.raises(TypeError):
        calculate_zero_alert_snapshots(
            [{"alert_count": 0}, "invalid"]
        )


def test_calculate_zero_alert_snapshots_rejects_invalid_count():
    with pytest.raises(ValueError):
        calculate_zero_alert_snapshots(
            [{"alert_count": "invalid"}]
        )


def test_calculate_alert_free_ratio():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 0},
        {"alert_count": 1},
    ]

    assert calculate_alert_free_ratio(history) == 0.5


def test_calculate_alert_free_ratio_all_clear():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_alert_free_ratio(history) == 1.0


def test_calculate_alert_free_ratio_none_clear():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
    ]

    assert calculate_alert_free_ratio(history) == 0.0


def test_calculate_alert_free_ratio_empty():
    with pytest.raises(ValueError):
        calculate_alert_free_ratio([])


def test_build_alert_statistics_summary():
    history = [
        {"alert_count": 0},
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 0},
    ]

    result = build_alert_statistics_summary(history)

    assert result == {
        "statistics": {
            "count": 4.0,
            "minimum": 0.0,
            "maximum": 4.0,
            "mean": 1.5,
            "range": 4.0,
        },
        "zero_alert_snapshots": 2,
        "alert_free_ratio": 0.5,
    }
