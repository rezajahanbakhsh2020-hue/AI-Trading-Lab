from __future__ import annotations

import pytest

from src.evaluation.report_alert_summary import (
    build_report_alert_overview,
    count_alerts_by_type,
    summarize_report_alerts,
)


def test_summarize_report_alerts():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.60,
        "max_drawdown": -0.30,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "max_drawdown": -0.20,
    }

    result = summarize_report_alerts(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 2,
        "has_alerts": True,
        "alerts": [
            "sharpe_ratio: below threshold",
            "max_drawdown: below threshold",
        ],
    }


def test_summarize_report_alerts_without_alerts():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.60,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = summarize_report_alerts(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 0,
        "has_alerts": False,
        "alerts": [],
    }


def test_summarize_report_alerts_detects_missing_value():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = summarize_report_alerts(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 1,
        "has_alerts": True,
        "alerts": [
            "win_rate: invalid or missing value",
        ],
    }


def test_summarize_report_alerts_rejects_invalid_report():
    with pytest.raises(TypeError):
        summarize_report_alerts(
            [],
            {"sharpe_ratio": 1.0},
        )


def test_summarize_report_alerts_rejects_invalid_thresholds():
    with pytest.raises(TypeError):
        summarize_report_alerts(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_count_alerts_by_type():
    alerts = [
        "sharpe_ratio: below threshold",
        "win_rate: invalid or missing value",
        "max_drawdown: below threshold",
        "sortino_ratio: invalid or missing value",
    ]

    result = count_alerts_by_type(alerts)

    assert result == {
        "invalid_or_missing": 2,
        "below_threshold": 2,
    }


def test_count_alerts_by_type_empty():
    assert count_alerts_by_type([]) == {
        "invalid_or_missing": 0,
        "below_threshold": 0,
    }


def test_count_alerts_by_type_ignores_unknown_alert_type():
    alerts = [
        "unknown: something else",
    ]

    assert count_alerts_by_type(alerts) == {
        "invalid_or_missing": 0,
        "below_threshold": 0,
    }


def test_count_alerts_by_type_rejects_invalid_alerts():
    with pytest.raises(TypeError):
        count_alerts_by_type("invalid")


def test_count_alerts_by_type_rejects_non_string_alert():
    with pytest.raises(TypeError):
        count_alerts_by_type(
            [
                "sharpe_ratio: below threshold",
                123,
            ]
        )


def test_build_report_alert_overview():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": "invalid",
        "max_drawdown": -0.10,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "max_drawdown": -0.20,
    }

    result = build_report_alert_overview(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 2,
        "has_alerts": True,
        "alerts": [
            "sharpe_ratio: below threshold",
            "win_rate: invalid or missing value",
        ],
        "alert_types": {
            "invalid_or_missing": 1,
            "below_threshold": 1,
        },
    }


def test_build_report_alert_overview_without_alerts():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    result = build_report_alert_overview(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 0,
        "has_alerts": False,
        "alerts": [],
        "alert_types": {
            "invalid_or_missing": 0,
            "below_threshold": 0,
        },
    }
