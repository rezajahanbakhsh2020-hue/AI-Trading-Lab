from __future__ import annotations

import pytest

from src.evaluation.report_alerts import (
    build_report_alert_summary,
    count_report_alerts,
    find_report_alerts,
    has_report_alerts,
)


def test_find_report_alerts_returns_failed_metrics():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.65,
        "total_return": 0.20,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "total_return": 0.10,
    }

    result = find_report_alerts(
        report,
        thresholds,
    )

    assert result == [
        "sharpe_ratio: below threshold",
    ]


def test_find_report_alerts_detects_missing_metric():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = find_report_alerts(
        report,
        thresholds,
    )

    assert result == [
        "win_rate: invalid or missing value",
    ]


def test_find_report_alerts_detects_non_numeric_metric():
    report = {
        "sharpe_ratio": "1.5",
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    result = find_report_alerts(
        report,
        thresholds,
    )

    assert result == [
        "sharpe_ratio: invalid or missing value",
    ]


def test_find_report_alerts_all_pass():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    assert find_report_alerts(
        report,
        thresholds,
    ) == []


def test_find_report_alerts_equal_threshold_is_not_alert():
    report = {
        "sharpe_ratio": 1.0,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    assert find_report_alerts(
        report,
        thresholds,
    ) == []


def test_find_report_alerts_empty_thresholds():
    assert find_report_alerts(
        {"sharpe_ratio": 1.5},
        {},
    ) == []


def test_find_report_alerts_rejects_invalid_report():
    with pytest.raises(TypeError):
        find_report_alerts(
            [],
            {"sharpe_ratio": 1.0},
        )


def test_find_report_alerts_rejects_invalid_thresholds():
    with pytest.raises(TypeError):
        find_report_alerts(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_find_report_alerts_rejects_invalid_threshold():
    with pytest.raises(TypeError):
        find_report_alerts(
            {"sharpe_ratio": 1.5},
            {"sharpe_ratio": "1.0"},
        )


def test_has_report_alerts_returns_true():
    report = {
        "sharpe_ratio": 0.8,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    assert has_report_alerts(
        report,
        thresholds,
    )


def test_has_report_alerts_returns_false():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    assert not has_report_alerts(
        report,
        thresholds,
    )


def test_count_report_alerts():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.4,
        "total_return": 0.20,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "total_return": 0.10,
    }

    assert count_report_alerts(
        report,
        thresholds,
    ) == 2


def test_count_report_alerts_zero():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    assert count_report_alerts(
        report,
        thresholds,
    ) == 0


def test_build_report_alert_summary():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = build_report_alert_summary(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 1,
        "has_alerts": True,
        "alerts": [
            "sharpe_ratio: below threshold",
        ],
    }


def test_build_report_alert_summary_without_alerts():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = build_report_alert_summary(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 0,
        "has_alerts": False,
        "alerts": [],
    }
