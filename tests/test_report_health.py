from __future__ import annotations

import pytest

from src.evaluation.report_health import (
    build_report_health_summary,
    calculate_report_health_score,
    find_invalid_report_metrics,
    is_report_healthy,
)


def test_calculate_report_health_score_complete():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
        "max_drawdown": -0.1,
    }

    result = calculate_report_health_score(
        report,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ],
    )

    assert result == 1.0


def test_calculate_report_health_score_partial():
    report = {
        "sharpe_ratio": 1.5,
    }

    result = calculate_report_health_score(
        report,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
            "total_return",
        ],
    )

    assert result == 0.25


def test_calculate_report_health_score_invalid_values():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": "0.6",
        "max_drawdown": None,
    }

    result = calculate_report_health_score(
        report,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ],
    )

    assert result == round(1 / 3, 10)


def test_calculate_report_health_score_empty_required_metrics():
    with pytest.raises(ValueError):
        calculate_report_health_score(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_calculate_report_health_score_rejects_invalid_report():
    with pytest.raises(TypeError):
        calculate_report_health_score(
            [],
            ["sharpe_ratio"],
        )


def test_calculate_report_health_score_rejects_invalid_metrics():
    with pytest.raises(TypeError):
        calculate_report_health_score(
            {"sharpe_ratio": 1.5},
            "sharpe_ratio",
        )


def test_find_invalid_report_metrics():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = find_invalid_report_metrics(
        report,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ],
    )

    assert result == ["max_drawdown"]


def test_find_invalid_report_metrics_detects_non_numeric():
    report = {
        "sharpe_ratio": "1.5",
        "win_rate": 0.6,
    }

    result = find_invalid_report_metrics(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == ["sharpe_ratio"]


def test_find_invalid_report_metrics_returns_empty():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = find_invalid_report_metrics(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == []


def test_find_invalid_report_metrics_rejects_invalid_report():
    with pytest.raises(TypeError):
        find_invalid_report_metrics(
            [],
            ["sharpe_ratio"],
        )


def test_is_report_healthy():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    assert is_report_healthy(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )


def test_is_report_healthy_returns_false():
    report = {
        "sharpe_ratio": 1.5,
    }

    assert not is_report_healthy(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )


def test_build_report_health_summary():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": "0.6",
    }

    result = build_report_health_summary(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "score": 0.5,
        "healthy": False,
        "invalid_metrics": ["win_rate"],
    }


def test_build_report_health_summary_healthy():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = build_report_health_summary(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "score": 1.0,
        "healthy": True,
        "invalid_metrics": [],
    }
