from __future__ import annotations

import pytest

from src.evaluation.report_aggregation import (
    aggregate_report_metric,
    aggregate_report_metrics,
    calculate_weighted_report_metric,
)


@pytest.fixture
def reports():
    return [
        {
            "strategy": "A",
            "sharpe_ratio": 1.0,
            "weight": 1.0,
        },
        {
            "strategy": "B",
            "sharpe_ratio": 2.0,
            "weight": 2.0,
        },
        {
            "strategy": "C",
            "sharpe_ratio": 1.5,
            "weight": 1.0,
        },
    ]


def test_aggregate_report_metric(reports):
    result = aggregate_report_metric(
        reports,
        "sharpe_ratio",
    )

    assert result["count"] == 3.0
    assert result["sum"] == 4.5
    assert result["mean"] == 1.5


def test_aggregate_report_metric_returns_float_values(
    reports,
):
    result = aggregate_report_metric(
        reports,
        "sharpe_ratio",
    )

    assert all(
        isinstance(value, float)
        for value in result.values()
    )


def test_aggregate_report_metric_rejects_invalid_reports():
    with pytest.raises(TypeError):
        aggregate_report_metric(
            ("invalid",),
            "sharpe_ratio",
        )


def test_aggregate_report_metric_rejects_invalid_report():
    with pytest.raises(TypeError):
        aggregate_report_metric(
            [{"sharpe_ratio": 1.0}, "invalid"],
            "sharpe_ratio",
        )


def test_aggregate_report_metric_rejects_invalid_metric(
    reports,
):
    with pytest.raises(TypeError):
        aggregate_report_metric(
            reports,
            123,
        )


def test_aggregate_report_metric_rejects_missing_metric():
    with pytest.raises(ValueError):
        aggregate_report_metric(
            [{"strategy": "A"}],
            "sharpe_ratio",
        )


def test_aggregate_report_metric_rejects_non_numeric_metric():
    with pytest.raises(ValueError):
        aggregate_report_metric(
            [{"sharpe_ratio": "1.5"}],
            "sharpe_ratio",
        )


def test_aggregate_report_metric_rejects_empty_reports():
    with pytest.raises(ValueError):
        aggregate_report_metric(
            [],
            "sharpe_ratio",
        )


def test_aggregate_report_metrics(reports):
    result = aggregate_report_metrics(
        reports,
        ["sharpe_ratio"],
    )

    assert result == {
        "sharpe_ratio": {
            "count": 3.0,
            "sum": 4.5,
            "mean": 1.5,
        }
    }


def test_aggregate_report_metrics_supports_multiple_metrics():
    reports = [
        {
            "sharpe_ratio": 1.0,
            "win_rate": 0.5,
        },
        {
            "sharpe_ratio": 2.0,
            "win_rate": 0.7,
        },
    ]

    result = aggregate_report_metrics(
        reports,
        ["sharpe_ratio", "win_rate"],
    )

    assert result["sharpe_ratio"]["mean"] == 1.5
    assert result["win_rate"]["mean"] == 0.6


def test_aggregate_report_metrics_rejects_invalid_reports():
    with pytest.raises(TypeError):
        aggregate_report_metrics(
            ("invalid",),
            ["sharpe_ratio"],
        )


def test_aggregate_report_metrics_rejects_invalid_metrics(
    reports,
):
    with pytest.raises(TypeError):
        aggregate_report_metrics(
            reports,
            "sharpe_ratio",
        )


def test_calculate_weighted_report_metric(reports):
    result = calculate_weighted_report_metric(
        reports,
        "sharpe_ratio",
        "weight",
    )

    assert result == pytest.approx(1.625)


def test_calculate_weighted_report_metric_with_equal_weights():
    reports = [
        {"metric": 1.0, "weight": 1.0},
        {"metric": 3.0, "weight": 1.0},
    ]

    result = calculate_weighted_report_metric(
        reports,
        "metric",
        "weight",
    )

    assert result == 2.0


def test_calculate_weighted_report_metric_rejects_invalid_metric(
    reports,
):
    with pytest.raises(TypeError):
        calculate_weighted_report_metric(
            reports,
            123,
            "weight",
        )


def test_calculate_weighted_report_metric_rejects_invalid_weight_field(
    reports,
):
    with pytest.raises(TypeError):
        calculate_weighted_report_metric(
            reports,
            "sharpe_ratio",
            123,
        )


def test_calculate_weighted_report_metric_rejects_missing_metric():
    with pytest.raises(ValueError):
        calculate_weighted_report_metric(
            [{"weight": 1.0}],
            "sharpe_ratio",
            "weight",
        )


def test_calculate_weighted_report_metric_rejects_missing_weight():
    with pytest.raises(ValueError):
        calculate_weighted_report_metric(
            [{"sharpe_ratio": 1.0}],
            "sharpe_ratio",
            "weight",
        )


def test_calculate_weighted_report_metric_rejects_negative_weight():
    with pytest.raises(ValueError):
        calculate_weighted_report_metric(
            [
                {
                    "sharpe_ratio": 1.0,
                    "weight": -1.0,
                }
            ],
            "sharpe_ratio",
            "weight",
        )


def test_calculate_weighted_report_metric_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        calculate_weighted_report_metric(
            [
                {
                    "sharpe_ratio": 1.0,
                    "weight": 0.0,
                }
            ],
            "sharpe_ratio",
            "weight",
        )
