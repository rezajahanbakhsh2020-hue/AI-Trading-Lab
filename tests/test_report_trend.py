from __future__ import annotations

import pytest

from src.evaluation.report_trend import (
    calculate_metric_changes,
    calculate_metric_trend,
    find_improving_trends,
)


@pytest.fixture
def reports():
    return [
        {"period": 1, "sharpe_ratio": 1.0},
        {"period": 2, "sharpe_ratio": 1.2},
        {"period": 3, "sharpe_ratio": 1.5},
    ]


def test_calculate_metric_trend_increasing(reports):
    result = calculate_metric_trend(
        reports,
        "sharpe_ratio",
    )

    assert result == {
        "first": 1.0,
        "last": 1.5,
        "change": 0.5,
        "direction": "increasing",
    }


def test_calculate_metric_trend_decreasing():
    reports = [
        {"value": 3.0},
        {"value": 2.0},
        {"value": 1.0},
    ]

    result = calculate_metric_trend(
        reports,
        "value",
    )

    assert result["first"] == 3.0
    assert result["last"] == 1.0
    assert result["change"] == -2.0
    assert result["direction"] == "decreasing"


def test_calculate_metric_trend_stable():
    reports = [
        {"value": 1.5},
        {"value": 1.5},
        {"value": 1.5},
    ]

    result = calculate_metric_trend(
        reports,
        "value",
    )

    assert result == {
        "first": 1.5,
        "last": 1.5,
        "change": 0.0,
        "direction": "stable",
    }


def test_calculate_metric_trend_single_report():
    reports = [{"value": 2.0}]

    result = calculate_metric_trend(
        reports,
        "value",
    )

    assert result == {
        "first": 2.0,
        "last": 2.0,
        "change": 0.0,
        "direction": "stable",
    }


def test_calculate_metric_trend_rejects_empty_reports():
    with pytest.raises(ValueError):
        calculate_metric_trend(
            [],
            "value",
        )


def test_calculate_metric_trend_rejects_invalid_reports():
    with pytest.raises(TypeError):
        calculate_metric_trend(
            ("invalid",),
            "value",
        )


def test_calculate_metric_trend_rejects_invalid_report():
    with pytest.raises(TypeError):
        calculate_metric_trend(
            [{"value": 1.0}, "invalid"],
            "value",
        )


def test_calculate_metric_trend_rejects_invalid_metric():
    with pytest.raises(TypeError):
        calculate_metric_trend(
            [{"value": 1.0}],
            123,
        )


def test_calculate_metric_trend_rejects_missing_metric():
    with pytest.raises(ValueError):
        calculate_metric_trend(
            [{"other": 1.0}],
            "value",
        )


def test_calculate_metric_trend_rejects_non_numeric_metric():
    with pytest.raises(ValueError):
        calculate_metric_trend(
            [{"value": "1.0"}],
            "value",
        )


def test_calculate_metric_changes(reports):
    result = calculate_metric_changes(
        reports,
        "sharpe_ratio",
    )

    assert result == [0.2, 0.3]


def test_calculate_metric_changes_single_report():
    result = calculate_metric_changes(
        [{"value": 2.0}],
        "value",
    )

    assert result == []


def test_calculate_metric_changes_empty_reports():
    result = calculate_metric_changes(
        [],
        "value",
    )

    assert result == []


def test_calculate_metric_changes_rejects_invalid_reports():
    with pytest.raises(TypeError):
        calculate_metric_changes(
            ("invalid",),
            "value",
        )


def test_calculate_metric_changes_rejects_missing_metric():
    with pytest.raises(ValueError):
        calculate_metric_changes(
            [{"other": 1.0}],
            "value",
        )


def test_find_improving_trends(reports):
    result = find_improving_trends(
        reports,
        ["sharpe_ratio"],
    )

    assert result == ["sharpe_ratio"]


def test_find_improving_trends_multiple_metrics():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.7,
            "max_drawdown": -0.1,
        },
        {
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
            "max_drawdown": -0.2,
        },
    ]

    result = find_improving_trends(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ],
    )

    assert result == [
        "sharpe_ratio",
        "max_drawdown",
    ]


def test_find_improving_trends_returns_empty_when_no_metric_improves():
    reports = [
        {"value": 2.0},
        {"value": 1.0},
    ]

    result = find_improving_trends(
        reports,
        ["value"],
    )

    assert result == []


def test_find_improving_trends_rejects_invalid_metrics(
    reports,
):
    with pytest.raises(TypeError):
        find_improving_trends(
            reports,
            "sharpe_ratio",
        )
