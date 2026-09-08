from __future__ import annotations

import pytest

from src.evaluation.report_alert_baseline import (
    build_alert_baseline_summary,
    calculate_absolute_deviations,
    calculate_alert_deviations,
    calculate_baseline_alert_count,
    calculate_baseline_ratio,
    calculate_mean_absolute_deviation,
)


def test_calculate_baseline_alert_count():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_baseline_alert_count(history) == 4.0


def test_calculate_alert_deviations():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_alert_deviations(history) == [
        -2.0,
        0.0,
        2.0,
    ]


def test_calculate_absolute_deviations():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_absolute_deviations(history) == [
        2.0,
        0.0,
        2.0,
    ]


def test_calculate_mean_absolute_deviation():
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    assert calculate_mean_absolute_deviation(history) == pytest.approx(
        1.5
    )


def test_calculate_baseline_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    assert calculate_baseline_ratio(history) == pytest.approx(1.5)


def test_baseline_with_equal_values():
    history = [
        {"alert_count": 5},
        {"alert_count": 5},
        {"alert_count": 5},
    ]

    assert calculate_baseline_alert_count(history) == 5.0
    assert calculate_alert_deviations(history) == [
        0.0,
        0.0,
        0.0,
    ]
    assert calculate_mean_absolute_deviation(history) == 0.0
    assert calculate_baseline_ratio(history) == 1.0


def test_all_zero_history():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    assert calculate_baseline_alert_count(history) == 0.0
    assert calculate_baseline_ratio(history) == 0.0


def test_single_snapshot():
    history = [
        {"alert_count": 7},
    ]

    assert calculate_baseline_alert_count(history) == 7.0
    assert calculate_alert_deviations(history) == [0.0]
    assert calculate_absolute_deviations(history) == [0.0]
    assert calculate_mean_absolute_deviation(history) == 0.0
    assert calculate_baseline_ratio(history) == 1.0


def test_build_summary():
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    result = build_alert_baseline_summary(history)

    assert result == {
        "snapshot_count": 3,
        "baseline_alert_count": 4.0,
        "peak_alert_count": 6,
        "minimum_alert_count": 2,
        "alert_deviations": [-2.0, 0.0, 2.0],
        "mean_absolute_deviation": 4 / 3,
        "peak_to_baseline_ratio": 1.5,
    }


def test_build_summary_with_zero_baseline():
    history = [
        {"alert_count": 0},
        {"alert_count": 0},
    ]

    result = build_alert_baseline_summary(history)

    assert result["baseline_alert_count"] == 0.0
    assert result["peak_alert_count"] == 0
    assert result["minimum_alert_count"] == 0
    assert result["mean_absolute_deviation"] == 0.0
    assert result["peak_to_baseline_ratio"] == 0.0


def test_empty_history():
    with pytest.raises(ValueError):
        calculate_baseline_alert_count([])

    with pytest.raises(ValueError):
        build_alert_baseline_summary([])


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_baseline_alert_count({})


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_baseline_alert_count(
            [{"alert_count": 5}, "invalid"]
        )


def test_missing_alert_count():
    with pytest.raises(ValueError):
        calculate_baseline_alert_count(
            [{"value": 5}]
        )


def test_non_integer_alert_count():
    with pytest.raises(ValueError):
        calculate_baseline_alert_count(
            [{"alert_count": 1.5}]
        )


def test_boolean_alert_count():
    with pytest.raises(ValueError):
        calculate_baseline_alert_count(
            [{"alert_count": True}]
        )


def test_negative_alert_count():
    with pytest.raises(ValueError):
        calculate_baseline_alert_count(
            [{"alert_count": -1}]
        )
