from __future__ import annotations

import pytest

from src.evaluation.report_alert_policy import (
    build_alert_policy_summary,
    evaluate_alert_policy,
    is_alert_policy_passed,
)


def test_evaluate_alert_policy_passes_without_alerts():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.60,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = evaluate_alert_policy(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 0,
        "alerts": [],
        "maximum_alerts": 0,
        "passed": True,
    }


def test_evaluate_alert_policy_fails_with_alerts():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.60,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = evaluate_alert_policy(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 1,
        "alerts": [
            "sharpe_ratio: below threshold",
        ],
        "maximum_alerts": 0,
        "passed": False,
    }


def test_evaluate_alert_policy_allows_configured_alerts():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.60,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = evaluate_alert_policy(
        report,
        thresholds,
        maximum_alerts=1,
    )

    assert result["alert_count"] == 1
    assert result["maximum_alerts"] == 1
    assert result["passed"] is True


def test_evaluate_alert_policy_fails_above_allowed_alerts():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.40,
        "profit_factor": 0.8,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "profit_factor": 1.0,
    }

    result = evaluate_alert_policy(
        report,
        thresholds,
        maximum_alerts=1,
    )

    assert result["alert_count"] == 3
    assert result["passed"] is False


def test_evaluate_alert_policy_detects_missing_metric():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = evaluate_alert_policy(
        report,
        thresholds,
    )

    assert result["alert_count"] == 1
    assert result["alerts"] == [
        "win_rate: invalid or missing value",
    ]
    assert result["passed"] is False


def test_evaluate_alert_policy_detects_non_numeric_metric():
    report = {
        "sharpe_ratio": "invalid",
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    result = evaluate_alert_policy(
        report,
        thresholds,
    )

    assert result["alerts"] == [
        "sharpe_ratio: invalid or missing value",
    ]


def test_evaluate_alert_policy_empty_thresholds():
    result = evaluate_alert_policy(
        {"sharpe_ratio": 1.5},
        {},
    )

    assert result == {
        "alert_count": 0,
        "alerts": [],
        "maximum_alerts": 0,
        "passed": True,
    }


def test_evaluate_alert_policy_rejects_invalid_report():
    with pytest.raises(TypeError):
        evaluate_alert_policy(
            [],
            {"sharpe_ratio": 1.0},
        )


def test_evaluate_alert_policy_rejects_invalid_thresholds():
    with pytest.raises(TypeError):
        evaluate_alert_policy(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_evaluate_alert_policy_rejects_invalid_metric_name():
    with pytest.raises(TypeError):
        evaluate_alert_policy(
            {"sharpe_ratio": 1.5},
            {123: 1.0},
        )


def test_evaluate_alert_policy_rejects_invalid_threshold():
    with pytest.raises(TypeError):
        evaluate_alert_policy(
            {"sharpe_ratio": 1.5},
            {"sharpe_ratio": "1.0"},
        )


def test_evaluate_alert_policy_rejects_invalid_maximum_alerts():
    with pytest.raises(TypeError):
        evaluate_alert_policy(
            {"sharpe_ratio": 1.5},
            {"sharpe_ratio": 1.0},
            maximum_alerts="0",
        )


def test_evaluate_alert_policy_rejects_negative_maximum_alerts():
    with pytest.raises(ValueError):
        evaluate_alert_policy(
            {"sharpe_ratio": 1.5},
            {"sharpe_ratio": 1.0},
            maximum_alerts=-1,
        )


def test_is_alert_policy_passed():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    assert is_alert_policy_passed(
        report,
        thresholds,
    ) is True


def test_is_alert_policy_not_passed():
    report = {
        "sharpe_ratio": 0.8,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    assert is_alert_policy_passed(
        report,
        thresholds,
    ) is False


def test_build_alert_policy_summary():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.60,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = build_alert_policy_summary(
        report,
        thresholds,
    )

    assert result == {
        "passed": False,
        "alert_count": 1,
        "maximum_alerts": 0,
        "alerts": [
            "sharpe_ratio: below threshold",
        ],
    }
