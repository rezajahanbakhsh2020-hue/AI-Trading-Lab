from __future__ import annotations

import pytest

from src.evaluation.report_alert_levels import (
    build_alert_level_summary,
    calculate_alert_level,
    classify_alert_level,
)


def test_classify_alert_level_none():
    assert classify_alert_level(0) == "none"


def test_classify_alert_level_low():
    assert classify_alert_level(1) == "low"
    assert classify_alert_level(2) == "low"


def test_classify_alert_level_medium():
    assert classify_alert_level(3) == "medium"
    assert classify_alert_level(5) == "medium"


def test_classify_alert_level_high():
    assert classify_alert_level(6) == "high"
    assert classify_alert_level(10) == "high"


def test_classify_alert_level_rejects_invalid_type():
    with pytest.raises(TypeError):
        classify_alert_level("2")


def test_classify_alert_level_rejects_boolean():
    with pytest.raises(TypeError):
        classify_alert_level(True)


def test_classify_alert_level_rejects_negative_count():
    with pytest.raises(ValueError):
        classify_alert_level(-1)


def test_calculate_alert_level_none():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.60,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    assert calculate_alert_level(
        report,
        thresholds,
    ) == "none"


def test_calculate_alert_level_low():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.60,
        "max_drawdown": -0.10,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "max_drawdown": -0.20,
    }

    assert calculate_alert_level(
        report,
        thresholds,
    ) == "low"


def test_calculate_alert_level_medium():
    report = {
        "metric_1": 0.0,
        "metric_2": 0.0,
        "metric_3": 0.0,
        "metric_4": 0.0,
        "metric_5": 0.0,
        "metric_6": 1.0,
    }

    thresholds = {
        "metric_1": 1.0,
        "metric_2": 1.0,
        "metric_3": 1.0,
        "metric_4": 1.0,
        "metric_5": 1.0,
        "metric_6": 1.0,
    }

    assert calculate_alert_level(
        report,
        thresholds,
    ) == "medium"


def test_calculate_alert_level_high():
    report = {
        "metric_1": 0.0,
        "metric_2": 0.0,
        "metric_3": 0.0,
        "metric_4": 0.0,
        "metric_5": 0.0,
        "metric_6": 0.0,
    }

    thresholds = {
        "metric_1": 1.0,
        "metric_2": 1.0,
        "metric_3": 1.0,
        "metric_4": 1.0,
        "metric_5": 1.0,
        "metric_6": 1.0,
    }

    assert calculate_alert_level(
        report,
        thresholds,
    ) == "high"


def test_calculate_alert_level_missing_metric():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    assert calculate_alert_level(
        report,
        thresholds,
    ) == "low"


def test_calculate_alert_level_invalid_report():
    with pytest.raises(TypeError):
        calculate_alert_level(
            [],
            {"sharpe_ratio": 1.0},
        )


def test_calculate_alert_level_invalid_thresholds():
    with pytest.raises(TypeError):
        calculate_alert_level(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_calculate_alert_level_invalid_metric_name():
    with pytest.raises(TypeError):
        calculate_alert_level(
            {"sharpe_ratio": 1.5},
            {123: 1.0},
        )


def test_calculate_alert_level_invalid_threshold():
    with pytest.raises(TypeError):
        calculate_alert_level(
            {"sharpe_ratio": 1.5},
            {"sharpe_ratio": "1.0"},
        )


def test_build_alert_level_summary():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.40,
        "max_drawdown": -0.30,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "max_drawdown": -0.20,
    }

    result = build_alert_level_summary(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 3,
        "level": "medium",
        "has_alerts": True,
    }


def test_build_alert_level_summary_without_alerts():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    result = build_alert_level_summary(
        report,
        thresholds,
    )

    assert result == {
        "alert_count": 0,
        "level": "none",
        "has_alerts": False,
    }
