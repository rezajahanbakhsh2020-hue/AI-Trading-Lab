from __future__ import annotations

import pytest

from src.evaluation.report_trend_summary import (
    build_report_trend_summary,
    count_trend_directions,
    summarize_report_trends,
)


def test_summarize_report_trends():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.60,
        },
        {
            "sharpe_ratio": 1.5,
            "win_rate": 0.60,
        },
        {
            "sharpe_ratio": 1.8,
            "win_rate": 0.60,
        },
    ]

    result = summarize_report_trends(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "sharpe_ratio": {
            "first": 1.0,
            "last": 1.8,
            "change": 0.8,
            "direction": "increasing",
        },
        "win_rate": {
            "first": 0.6,
            "last": 0.6,
            "change": 0.0,
            "direction": "stable",
        },
    }


def test_summarize_report_trends_decreasing():
    reports = [
        {"value": 3.0},
        {"value": 2.0},
        {"value": 1.0},
    ]

    result = summarize_report_trends(
        reports,
        ["value"],
    )

    assert result == {
        "value": {
            "first": 3.0,
            "last": 1.0,
            "change": -2.0,
            "direction": "decreasing",
        },
    }


def test_summarize_report_trends_empty_metrics():
    result = summarize_report_trends(
        [{"value": 1.0}],
        [],
    )

    assert result == {}


def test_summarize_report_trends_rejects_invalid_reports():
    with pytest.raises(TypeError):
        summarize_report_trends(
            (),
            ["value"],
        )


def test_summarize_report_trends_rejects_invalid_metrics():
    with pytest.raises(TypeError):
        summarize_report_trends(
            [{"value": 1.0}],
            "value",
        )


def test_summarize_report_trends_rejects_invalid_metric_name():
    with pytest.raises(TypeError):
        summarize_report_trends(
            [{"value": 1.0}],
            [123],
        )


def test_count_trend_directions():
    trends = {
        "sharpe_ratio": {
            "direction": "increasing",
        },
        "win_rate": {
            "direction": "stable",
        },
        "total_return": {
            "direction": "decreasing",
        },
        "sortino_ratio": {
            "direction": "increasing",
        },
    }

    result = count_trend_directions(trends)

    assert result == {
        "increasing": 2,
        "decreasing": 1,
        "stable": 1,
    }


def test_count_trend_directions_empty():
    assert count_trend_directions({}) == {
        "increasing": 0,
        "decreasing": 0,
        "stable": 0,
    }


def test_count_trend_directions_rejects_invalid_trends():
    with pytest.raises(TypeError):
        count_trend_directions([])


def test_count_trend_directions_rejects_invalid_trend():
    with pytest.raises(TypeError):
        count_trend_directions(
            {
                "sharpe_ratio": "invalid",
            }
        )


def test_count_trend_directions_rejects_invalid_direction():
    with pytest.raises(ValueError):
        count_trend_directions(
            {
                "sharpe_ratio": {
                    "direction": "unknown",
                },
            }
        )


def test_build_report_trend_summary():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.60,
        },
        {
            "sharpe_ratio": 1.5,
            "win_rate": 0.60,
        },
        {
            "sharpe_ratio": 1.8,
            "win_rate": 0.60,
        },
    ]

    result = build_report_trend_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "report_count": 3,
        "metric_count": 2,
        "trends": {
            "sharpe_ratio": {
                "first": 1.0,
                "last": 1.8,
                "change": 0.8,
                "direction": "increasing",
            },
            "win_rate": {
                "first": 0.6,
                "last": 0.6,
                "change": 0.0,
                "direction": "stable",
            },
        },
        "direction_counts": {
            "increasing": 1,
            "decreasing": 0,
            "stable": 1,
        },
    }
