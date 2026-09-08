import pytest

from src.evaluation.report_metrics_summary import (
    build_metrics_summary,
    calculate_metric_average,
    calculate_metric_count,
    summarize_report_metrics,
)


@pytest.fixture
def report():
    return {
        "total_return": 0.20,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "calmar_ratio": 1.2,
        "win_rate": 0.60,
        "profit_factor": 2.5,
        "strategy": "momentum",
        "status": "pass",
    }


def test_summarize_report_metrics(report):
    result = summarize_report_metrics(report)

    assert result == {
        "total_return": 0.20,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "calmar_ratio": 1.2,
        "win_rate": 0.60,
        "profit_factor": 2.5,
    }


def test_summarize_report_metrics_returns_floats(
    report,
):
    result = summarize_report_metrics(report)

    assert all(
        isinstance(value, float)
        for value in result.values()
    )


def test_summarize_report_metrics_ignores_unknown_fields():
    report = {
        "sharpe_ratio": 2,
        "strategy": "A",
        "custom_metric": 10,
    }

    result = summarize_report_metrics(report)

    assert result == {
        "sharpe_ratio": 2.0,
    }


def test_summarize_report_metrics_ignores_non_numeric_values():
    report = {
        "sharpe_ratio": "1.5",
        "win_rate": True,
        "profit_factor": None,
    }

    result = summarize_report_metrics(report)

    assert result == {}


def test_summarize_report_metrics_empty_report():
    assert summarize_report_metrics({}) == {}


def test_summarize_report_metrics_rejects_invalid_report():
    with pytest.raises(TypeError):
        summarize_report_metrics(["invalid"])


def test_calculate_metric_count(report):
    assert calculate_metric_count(report) == 7


def test_calculate_metric_count_empty_report():
    assert calculate_metric_count({}) == 0


def test_calculate_metric_average(report):
    expected = (
        0.20
        - 0.10
        + 1.5
        + 2.0
        + 1.2
        + 0.60
        + 2.5
    ) / 7

    assert calculate_metric_average(report) == pytest.approx(
        expected
    )


def test_calculate_metric_average_rejects_empty_report():
    with pytest.raises(ValueError):
        calculate_metric_average({})


def test_calculate_metric_average_rejects_invalid_report():
    with pytest.raises(TypeError):
        calculate_metric_average(["invalid"])


def test_build_metrics_summary(report):
    result = build_metrics_summary(report)

    assert result["metrics"] == {
        "total_return": 0.20,
        "max_drawdown": -0.10,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "calmar_ratio": 1.2,
        "win_rate": 0.60,
        "profit_factor": 2.5,
    }

    assert result["count"] == 7

    assert result["average"] == pytest.approx(
        calculate_metric_average(report)
    )


def test_build_metrics_summary_empty_report():
    result = build_metrics_summary({})

    assert result == {
        "metrics": {},
        "count": 0,
        "average": 0.0,
    }


def test_build_metrics_summary_rejects_invalid_report():
    with pytest.raises(TypeError):
        build_metrics_summary(["invalid"])
