from __future__ import annotations

import pytest

from src.evaluation.report_thresholds import (
    build_threshold_summary,
    evaluate_report_thresholds,
    find_failed_thresholds,
    is_report_within_thresholds,
)


def test_evaluate_report_thresholds_all_pass():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
        "total_return": 0.20,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
        "total_return": 0.10,
    }

    result = evaluate_report_thresholds(
        report,
        thresholds,
    )

    assert result == {
        "sharpe_ratio": True,
        "win_rate": True,
        "total_return": True,
    }


def test_evaluate_report_thresholds_some_fail():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = evaluate_report_thresholds(
        report,
        thresholds,
    )

    assert result == {
        "sharpe_ratio": False,
        "win_rate": True,
    }


def test_evaluate_report_thresholds_missing_metric_fails():
    report = {
        "sharpe_ratio": 1.5,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = evaluate_report_thresholds(
        report,
        thresholds,
    )

    assert result == {
        "sharpe_ratio": True,
        "win_rate": False,
    }


def test_evaluate_report_thresholds_non_numeric_metric_fails():
    report = {
        "sharpe_ratio": "1.5",
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    result = evaluate_report_thresholds(
        report,
        thresholds,
    )

    assert result == {
        "sharpe_ratio": False,
    }


def test_evaluate_report_thresholds_equal_value_passes():
    report = {
        "sharpe_ratio": 1.0,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
    }

    result = evaluate_report_thresholds(
        report,
        thresholds,
    )

    assert result == {
        "sharpe_ratio": True,
    }


def test_evaluate_report_thresholds_empty_thresholds():
    result = evaluate_report_thresholds(
        {"sharpe_ratio": 1.5},
        {},
    )

    assert result == {}


def test_evaluate_report_thresholds_rejects_invalid_report():
    with pytest.raises(TypeError):
        evaluate_report_thresholds(
            [],
            {"sharpe_ratio": 1.0},
        )


def test_evaluate_report_thresholds_rejects_invalid_thresholds():
    with pytest.raises(TypeError):
        evaluate_report_thresholds(
            {"sharpe_ratio": 1.5},
            [],
        )


def test_evaluate_report_thresholds_rejects_invalid_threshold_value():
    with pytest.raises(TypeError):
        evaluate_report_thresholds(
            {"sharpe_ratio": 1.5},
            {"sharpe_ratio": "1.0"},
        )


def test_find_failed_thresholds():
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

    result = find_failed_thresholds(
        report,
        thresholds,
    )

    assert result == ["sharpe_ratio"]


def test_find_failed_thresholds_returns_empty_when_all_pass():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    assert find_failed_thresholds(
        report,
        thresholds,
    ) == []


def test_is_report_within_thresholds():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    assert is_report_within_thresholds(
        report,
        thresholds,
    )


def test_is_report_within_thresholds_returns_false():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    assert not is_report_within_thresholds(
        report,
        thresholds,
    )


def test_is_report_within_thresholds_empty_thresholds():
    assert is_report_within_thresholds(
        {"sharpe_ratio": 1.5},
        {},
    )


def test_build_threshold_summary():
    report = {
        "sharpe_ratio": 0.8,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = build_threshold_summary(
        report,
        thresholds,
    )

    assert result == {
        "thresholds": {
            "sharpe_ratio": 1.0,
            "win_rate": 0.50,
        },
        "results": {
            "sharpe_ratio": False,
            "win_rate": True,
        },
        "failed_metrics": ["sharpe_ratio"],
        "passed": False,
    }


def test_build_threshold_summary_all_pass():
    report = {
        "sharpe_ratio": 1.5,
        "win_rate": 0.65,
    }

    thresholds = {
        "sharpe_ratio": 1.0,
        "win_rate": 0.50,
    }

    result = build_threshold_summary(
        report,
        thresholds,
    )

    assert result == {
        "thresholds": {
            "sharpe_ratio": 1.0,
            "win_rate": 0.50,
        },
        "results": {
            "sharpe_ratio": True,
            "win_rate": True,
        },
        "failed_metrics": [],
        "passed": True,
    }
