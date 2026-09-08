import pytest

from src.evaluation.report_statistics import (
    calculate_metric_statistics,
    calculate_report_statistics,
    find_best_report,
)


@pytest.fixture
def reports():
    return [
        {
            "total_return": 0.10,
            "sharpe_ratio": 1.0,
        },
        {
            "total_return": 0.20,
            "sharpe_ratio": 1.5,
        },
        {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
        },
    ]


def test_calculate_metric_statistics(reports):
    result = calculate_metric_statistics(
        reports,
        "total_return",
    )

    assert result["count"] == 3.0
    assert result["minimum"] == pytest.approx(0.10)
    assert result["maximum"] == pytest.approx(0.20)
    assert result["mean"] == pytest.approx(0.15)
    assert result["range"] == pytest.approx(0.10)


def test_calculate_metric_statistics_ignores_missing_values():
    reports = [
        {"sharpe_ratio": 1.0},
        {"other_metric": 2.0},
        {"sharpe_ratio": 1.5},
    ]

    result = calculate_metric_statistics(
        reports,
        "sharpe_ratio",
    )

    assert result["count"] == 2.0
    assert result["mean"] == pytest.approx(1.25)


def test_calculate_metric_statistics_rejects_non_numeric_values():
    reports = [
        {"sharpe_ratio": "1.5"},
    ]

    with pytest.raises(ValueError):
        calculate_metric_statistics(
            reports,
            "sharpe_ratio",
        )


def test_calculate_metric_statistics_rejects_empty_reports():
    with pytest.raises(ValueError):
        calculate_metric_statistics(
            [],
            "sharpe_ratio",
        )


def test_calculate_metric_statistics_rejects_invalid_reports():
    with pytest.raises(TypeError):
        calculate_metric_statistics(
            [{"sharpe_ratio": 1.0}, "invalid"],
            "sharpe_ratio",
        )


def test_calculate_metric_statistics_rejects_invalid_reports_argument():
    with pytest.raises(TypeError):
        calculate_metric_statistics(
            ("invalid",),
            "sharpe_ratio",
        )


def test_calculate_metric_statistics_rejects_invalid_metric(
    reports,
):
    with pytest.raises(TypeError):
        calculate_metric_statistics(
            reports,
            123,
        )


def test_calculate_report_statistics(reports):
    result = calculate_report_statistics(
        reports,
        [
            "total_return",
            "sharpe_ratio",
        ],
    )

    assert set(result) == {
        "total_return",
        "sharpe_ratio",
    }

    assert result["total_return"]["mean"] == pytest.approx(0.15)
    assert result["sharpe_ratio"]["mean"] == pytest.approx(1.2333333333)


def test_calculate_report_statistics_rejects_invalid_metrics(
    reports,
):
    with pytest.raises(TypeError):
        calculate_report_statistics(
            reports,
            "sharpe_ratio",
        )


def test_calculate_report_statistics_rejects_invalid_reports():
    with pytest.raises(TypeError):
        calculate_report_statistics(
            ("invalid",),
            ["sharpe_ratio"],
        )


def test_find_best_report(reports):
    result = find_best_report(
        reports,
        "sharpe_ratio",
    )

    assert result == {
        "total_return": 0.20,
        "sharpe_ratio": 1.5,
    }


def test_find_best_report_uses_highest_value():
    reports = [
        {"metric": -1.0},
        {"metric": -0.5},
        {"metric": -2.0},
    ]

    result = find_best_report(
        reports,
        "metric",
    )

    assert result["metric"] == -0.5


def test_find_best_report_rejects_empty_reports():
    with pytest.raises(ValueError):
        find_best_report(
            [],
            "sharpe_ratio",
        )


def test_find_best_report_rejects_missing_metric():
    with pytest.raises(ValueError):
        find_best_report(
            [{"other_metric": 1.0}],
            "sharpe_ratio",
        )


def test_find_best_report_rejects_non_numeric_metric():
    with pytest.raises(ValueError):
        find_best_report(
            [{"sharpe_ratio": "1.5"}],
            "sharpe_ratio",
        )


def test_find_best_report_rejects_invalid_reports_argument():
    with pytest.raises(TypeError):
        find_best_report(
            ("invalid",),
            "sharpe_ratio",
        )


def test_find_best_report_rejects_invalid_metric(
    reports,
):
    with pytest.raises(TypeError):
        find_best_report(
            reports,
            123,
        )
