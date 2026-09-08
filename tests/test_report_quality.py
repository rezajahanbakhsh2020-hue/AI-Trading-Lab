from __future__ import annotations

import pytest

from src.evaluation.report_quality import (
    build_report_quality_summary,
    calculate_report_quality_score,
    find_missing_report_metrics,
    is_report_complete,
)


def test_calculate_report_quality_score_complete_report():
    report = {
        "total_return": 0.25,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 1.8,
    }

    result = calculate_report_quality_score(
        report,
        [
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
        ],
    )

    assert result == 1.0


def test_calculate_report_quality_score_partial_report():
    report = {
        "total_return": 0.25,
        "sharpe_ratio": 1.5,
    }

    result = calculate_report_quality_score(
        report,
        [
            "total_return",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
        ],
    )

    assert result == 0.5


def test_calculate_report_quality_score_empty_report():
    result = calculate_report_quality_score(
        {},
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == 0.0


def test_calculate_report_quality_score_default_metrics():
    report = {
        "total_return": 0.25,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.5,
    }

    result = calculate_report_quality_score(report)

    assert result == round(3 / 7, 10)


def test_calculate_report_quality_score_ignores_non_numeric_values():
    report = {
        "sharpe_ratio": "1.5",
        "win_rate": 0.6,
    }

    result = calculate_report_quality_score(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == 0.5


def test_calculate_report_quality_score_rejects_empty_metrics():
    with pytest.raises(ValueError):
        calculate_report_quality_score(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_calculate_report_quality_score_rejects_invalid_report():
    with pytest.raises(TypeError):
        calculate_report_quality_score(
            [],
            ["sharpe_ratio"],
        )


def test_calculate_report_quality_score_rejects_invalid_metrics():
    with pytest.raises(TypeError):
        calculate_report_quality_score(
            {"sharpe_ratio": 1.5},
            "sharpe_ratio",
        )


def test_find_missing_report_metrics():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = find_missing_report_metrics(
        report,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ],
    )

    assert result == ["max_drawdown"]


def test_find_missing_report_metrics_detects_non_numeric():
    report = {
        "sharpe_ratio": "1.5",
        "win_rate": 0.6,
    }

    result = find_missing_report_metrics(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == ["sharpe_ratio"]


def test_find_missing_report_metrics_returns_empty_when_complete():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = find_missing_report_metrics(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == []


def test_find_missing_report_metrics_rejects_invalid_report():
    with pytest.raises(TypeError):
        find_missing_report_metrics(
            [],
            ["sharpe_ratio"],
        )


def test_is_report_complete():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    assert is_report_complete(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )


def test_is_report_complete_returns_false():
    report = {
        "sharpe_ratio": 1.5,
    }

    assert not is_report_complete(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )


def test_build_report_quality_summary():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = build_report_quality_summary(
        report,
        [
            "sharpe_ratio",
            "win_rate",
            "max_drawdown",
        ],
    )

    assert result == {
        "score": round(2 / 3, 10),
        "complete": False,
        "missing_metrics": ["max_drawdown"],
    }


def test_build_report_quality_summary_complete():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.6,
    }

    result = build_report_quality_summary(
        report,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "score": 1.0,
        "complete": True,
        "missing_metrics": [],
    }


def test_build_report_quality_summary_rejects_invalid_report():
    with pytest.raises(TypeError):
        build_report_quality_summary(
            [],
            ["sharpe_ratio"],
        )
