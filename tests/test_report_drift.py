from __future__ import annotations

import pytest

from src.evaluation.report_drift import (
    build_report_drift_summary,
    calculate_metric_drift,
    calculate_relative_metric_drift,
    find_high_drift_metrics,
)


def test_calculate_metric_drift():
    reports = [
        {"sharpe_ratio": 1.0},
        {"sharpe_ratio": 1.2},
        {"sharpe_ratio": 1.5},
    ]

    result = calculate_metric_drift(
        reports,
        "sharpe_ratio",
    )

    assert result == 0.5


def test_calculate_metric_drift_uses_absolute_change():
    reports = [
        {"value": 2.0},
        {"value": 1.0},
    ]

    result = calculate_metric_drift(
        reports,
        "value",
    )

    assert result == 1.0


def test_calculate_metric_drift_single_report():
    result = calculate_metric_drift(
        [{"value": 2.0}],
        "value",
    )

    assert result == 0.0


def test_calculate_metric_drift_rejects_empty_reports():
    with pytest.raises(ValueError):
        calculate_metric_drift(
            [],
            "value",
        )


def test_calculate_metric_drift_rejects_invalid_report():
    with pytest.raises(TypeError):
        calculate_metric_drift(
            [{"value": 1.0}, "invalid"],
            "value",
        )


def test_calculate_metric_drift_rejects_missing_metric():
    with pytest.raises(ValueError):
        calculate_metric_drift(
            [{"other": 1.0}],
            "value",
        )


def test_calculate_relative_metric_drift():
    reports = [
        {"value": 100.0},
        {"value": 120.0},
    ]

    result = calculate_relative_metric_drift(
        reports,
        "value",
    )

    assert result == 0.2


def test_calculate_relative_metric_drift_with_decrease():
    reports = [
        {"value": 100.0},
        {"value": 80.0},
    ]

    result = calculate_relative_metric_drift(
        reports,
        "value",
    )

    assert result == 0.2


def test_calculate_relative_metric_drift_zero_to_zero():
    reports = [
        {"value": 0.0},
        {"value": 0.0},
    ]

    result = calculate_relative_metric_drift(
        reports,
        "value",
    )

    assert result == 0.0


def test_calculate_relative_metric_drift_rejects_zero_baseline():
    reports = [
        {"value": 0.0},
        {"value": 1.0},
    ]

    with pytest.raises(ValueError):
        calculate_relative_metric_drift(
            reports,
            "value",
        )


def test_find_high_drift_metrics():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.50,
        },
        {
            "sharpe_ratio": 1.5,
            "win_rate": 0.52,
        },
    ]

    result = find_high_drift_metrics(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
        maximum_drift=0.1,
    )

    assert result == ["sharpe_ratio"]


def test_find_high_drift_metrics_at_threshold():
    reports = [
        {"value": 1.0},
        {"value": 1.2},
    ]

    result = find_high_drift_metrics(
        reports,
        ["value"],
        maximum_drift=0.2,
    )

    assert result == []


def test_find_high_drift_metrics_rejects_negative_threshold():
    with pytest.raises(ValueError):
        find_high_drift_metrics(
            [{"value": 1.0}, {"value": 2.0}],
            ["value"],
            maximum_drift=-0.1,
        )


def test_find_high_drift_metrics_rejects_invalid_threshold():
    with pytest.raises(TypeError):
        find_high_drift_metrics(
            [{"value": 1.0}, {"value": 2.0}],
            ["value"],
            maximum_drift="1.0",
        )


def test_build_report_drift_summary():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.50,
        },
        {
            "sharpe_ratio": 1.5,
            "win_rate": 0.52,
        },
    ]

    result = build_report_drift_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
        maximum_drift=0.1,
    )

    assert result == {
        "report_count": 2,
        "drift": {
            "sharpe_ratio": 0.5,
            "win_rate": 0.02,
        },
        "high_drift_metrics": ["sharpe_ratio"],
        "stable": False,
    }


def test_build_report_drift_summary_stable():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.50,
        },
        {
            "sharpe_ratio": 1.05,
            "win_rate": 0.51,
        },
    ]

    result = build_report_drift_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
        maximum_drift=0.1,
    )

    assert result == {
        "report_count": 2,
        "drift": {
            "sharpe_ratio": 0.05,
            "win_rate": 0.01,
        },
        "high_drift_metrics": [],
        "stable": True,
    }
