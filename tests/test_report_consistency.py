from __future__ import annotations

import pytest

from src.evaluation.report_consistency import (
    build_report_consistency_summary,
    calculate_report_metric_coverage,
    find_common_report_metrics,
    find_inconsistent_report_metrics,
)


def test_find_common_report_metrics():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
            "total_return": 0.2,
        },
        {
            "sharpe_ratio": 1.4,
            "win_rate": 0.7,
            "max_drawdown": -0.1,
        },
        {
            "sharpe_ratio": 1.5,
            "win_rate": 0.65,
        },
    ]

    result = find_common_report_metrics(reports)

    assert result == [
        "sharpe_ratio",
        "win_rate",
    ]


def test_find_common_report_metrics_single_report():
    reports = [
        {
            "sharpe_ratio": 1.2,
            "win_rate": 0.6,
        },
    ]

    result = find_common_report_metrics(reports)

    assert result == [
        "sharpe_ratio",
        "win_rate",
    ]


def test_find_common_report_metrics_empty():
    assert find_common_report_metrics([]) == []


def test_find_common_report_metrics_rejects_invalid_reports():
    with pytest.raises(TypeError):
        find_common_report_metrics(
            [
                {"sharpe_ratio": 1.2},
                "invalid",
            ]
        )


def test_find_inconsistent_report_metrics():
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
            "win_rate": 0.65,
        },
    ]

    result = find_inconsistent_report_metrics(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == ["win_rate"]


def test_find_inconsistent_report_metrics_ignores_absent_metric():
    reports = [
        {"sharpe_ratio": 1.2},
        {"sharpe_ratio": 1.4},
    ]

    result = find_inconsistent_report_metrics(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == []


def test_find_inconsistent_report_metrics_rejects_invalid_metrics():
    with pytest.raises(TypeError):
        find_inconsistent_report_metrics(
            [{"sharpe_ratio": 1.2}],
            "sharpe_ratio",
        )


def test_calculate_report_metric_coverage():
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
            "win_rate": 0.65,
        },
    ]

    result = calculate_report_metric_coverage(
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


def test_calculate_report_metric_coverage_empty_reports():
    result = calculate_report_metric_coverage(
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


def test_calculate_report_metric_coverage_rejects_invalid_report():
    with pytest.raises(TypeError):
        calculate_report_metric_coverage(
            [{"sharpe_ratio": 1.2}, "invalid"],
            ["sharpe_ratio"],
        )


def test_build_report_consistency_summary():
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
            "win_rate": 0.65,
        },
    ]

    result = build_report_consistency_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "report_count": 3,
        "common_metrics": ["sharpe_ratio"],
        "inconsistent_metrics": ["win_rate"],
        "coverage": {
            "sharpe_ratio": 1.0,
            "win_rate": round(2 / 3, 10),
        },
        "consistent": False,
    }


def test_build_report_consistency_summary_complete():
    reports = [
        {"sharpe_ratio": 1.2, "win_rate": 0.6},
        {"sharpe_ratio": 1.4, "win_rate": 0.7},
    ]

    result = build_report_consistency_summary(
        reports,
        [
            "sharpe_ratio",
            "win_rate",
        ],
    )

    assert result == {
        "report_count": 2,
        "common_metrics": [
            "sharpe_ratio",
            "win_rate",
        ],
        "inconsistent_metrics": [],
        "coverage": {
            "sharpe_ratio": 1.0,
            "win_rate": 1.0,
        },
        "consistent": True,
    }
