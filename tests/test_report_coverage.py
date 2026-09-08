from __future__ import annotations

import pytest

from src.evaluation.report_coverage import (
    build_report_coverage_summary,
    calculate_report_coverage,
    find_low_coverage_metrics,
    is_report_set_complete,
)


def test_calculate_report_coverage():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
        },
        {
            "sharpe_ratio": 1.4,
            "win_rate": 0.7,
        },
        {
            "sharpe_ratio": 1.5,
        },
    ]

    result = calculate_report_coverage(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "sharpe_ratio": 1.0,
        "win_rate": round(2 / 3, 10),
    }


def test_calculate_report_coverage_ignores_non_numeric_values():
    reports = [
        {
            "sharpe_ratio": 1.2,
        },
        {
            "sharpe_ratio": "1.4",
        },
        {
            "sharpe_ratio": None,
        },
    ]

    result = calculate_report_coverage(
        reports,
        ["sharpe_ratio"],
    )

    assert result == {
        "sharpe_ratio": round(1 / 3, 10),
    }


def test_calculate_report_coverage_empty_reports():
    result = calculate_report_coverage(
        [],
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "sharpe_ratio": 0.0,
        "win_rate": 0.0,
    }


def test_calculate_report_coverage_rejects_invalid_reports():
    with pytest.raises(TypeError):
        calculate_report_coverage(
            [{"sharpe_ratio": 1.2}, "invalid"],
            ["sharpe_ratio"],
        )


def test_calculate_report_coverage_rejects_invalid_metrics():
    with pytest.raises(TypeError):
        calculate_report_coverage(
            [{"sharpe_ratio": 1.2}],
            "sharpe_ratio",
        )


def test_find_low_coverage_metrics():
    reports = [
        {"sharpe_ratio": 1.2, "win_rate": 0.6},
        {"sharpe_ratio": 1.4},
        {"sharpe_ratio": 1.5},
    ]

    result = find_low_coverage_metrics(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
        minimum_coverage=0.8,
    )

    assert result == ["win_rate"]


def test_find_low_coverage_metrics_accepts_full_coverage():
    reports = [
        {"sharpe_ratio": 1.2},
        {"sharpe_ratio": 1.4},
    ]

    result = find_low_coverage_metrics(
        reports,
        ["sharpe_ratio"],
        minimum_coverage=1.0,
    )

    assert result == []


def test_find_low_coverage_metrics_rejects_invalid_threshold():
    reports = [{"sharpe_ratio": 1.2}]

    with pytest.raises(ValueError):
        find_low_coverage_metrics(
            reports,
            ["sharpe_ratio"],
            minimum_coverage=1.5,
        )


def test_find_low_coverage_metrics_rejects_non_numeric_threshold():
    reports = [{"sharpe_ratio": 1.2}]

    with pytest.raises(TypeError):
        find_low_coverage_metrics(
            reports,
            ["sharpe_ratio"],
            minimum_coverage="1.0",
        )


def test_is_report_set_complete():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
        },
        {
            "sharpe_ratio": 1.4,
            "win_rate": 0.7,
        },
    ]

    assert is_report_set_complete(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )


def test_is_report_set_complete_returns_false():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
        },
        {
            "sharpe_ratio": 1.4,
        },
    ]

    assert not is_report_set_complete(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )


def test_is_report_set_complete_empty_reports():
    assert not is_report_set_complete(
        [],
        ["sharpe_ratio"],
    )


def test_build_report_coverage_summary():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
        },
        {
            "sharpe_ratio": 1.4,
        },
        {
            "sharpe_ratio": 1.5,
        },
    ]

    result = build_report_coverage_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
        minimum_coverage=0.8,
    )

    assert result == {
        "report_count": 3,
        "coverage": {
            "sharpe_ratio": 1.0,
            "win_rate": round(1 / 3, 10),
        },
        "low_coverage_metrics": ["win_rate"],
        "complete": False,
    }


def test_build_report_coverage_summary_complete():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
        },
        {
            "sharpe_ratio": 1.4,
            "win_rate": 0.7,
        },
    ]

    result = build_report_coverage_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "report_count": 2,
        "coverage": {
            "sharpe_ratio": 1.0,
            "win_rate": 1.0,
        },
        "low_coverage_metrics": [],
        "complete": True,
    }
