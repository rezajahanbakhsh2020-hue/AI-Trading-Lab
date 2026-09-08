import pytest

from src.evaluation.report_comparison import (
    calculate_report_differences,
    compare_reports,
    find_degraded_metrics,
    find_improved_metrics,
)


@pytest.fixture
def first_report():
    return {
        "total_return": 0.10,
        "max_drawdown": -0.08,
        "sharpe_ratio": 1.20,
        "strategy": "XAU/USD",
    }


@pytest.fixture
def second_report():
    return {
        "total_return": 0.15,
        "max_drawdown": -0.05,
        "sharpe_ratio": 1.50,
        "strategy": "XAU/USD",
    }


def test_compare_reports_returns_common_fields(
    first_report,
    second_report,
):
    result = compare_reports(
        first_report,
        second_report,
    )

    assert result["total_return"] == {
        "first": 0.10,
        "second": 0.15,
        "difference": pytest.approx(0.05),
    }

    assert result["sharpe_ratio"] == {
        "first": 1.20,
        "second": 1.50,
        "difference": pytest.approx(0.30),
    }


def test_compare_reports_handles_non_numeric_common_field(
    first_report,
    second_report,
):
    result = compare_reports(
        first_report,
        second_report,
    )

    assert result["strategy"] == {
        "first": "XAU/USD",
        "second": "XAU/USD",
    }

    assert "difference" not in result["strategy"]


def test_compare_reports_ignores_fields_not_common():
    first = {
        "total_return": 0.10,
        "old_metric": 1.0,
    }

    second = {
        "total_return": 0.20,
        "new_metric": 2.0,
    }

    result = compare_reports(first, second)

    assert set(result) == {"total_return"}


def test_compare_reports_rejects_invalid_first_report(
    second_report,
):
    with pytest.raises(TypeError):
        compare_reports(
            ["invalid"],
            second_report,
        )


def test_compare_reports_rejects_invalid_second_report(
    first_report,
):
    with pytest.raises(TypeError):
        compare_reports(
            first_report,
            ["invalid"],
        )


def test_calculate_report_differences(
    first_report,
    second_report,
):
    result = calculate_report_differences(
        first_report,
        second_report,
    )

    assert result["total_return"] == pytest.approx(0.05)
    assert result["max_drawdown"] == pytest.approx(0.03)
    assert result["sharpe_ratio"] == pytest.approx(0.30)


def test_calculate_report_differences_ignores_non_numeric_fields(
    first_report,
    second_report,
):
    result = calculate_report_differences(
        first_report,
        second_report,
    )

    assert "strategy" not in result


def test_calculate_report_differences_ignores_missing_fields():
    first = {
        "total_return": 0.10,
        "only_first": 1.0,
    }

    second = {
        "total_return": 0.20,
        "only_second": 2.0,
    }

    result = calculate_report_differences(first, second)

    assert result == {
        "total_return": pytest.approx(0.10),
    }


def test_calculate_report_differences_rejects_invalid_first_report():
    with pytest.raises(TypeError):
        calculate_report_differences(
            ["invalid"],
            {"metric": 1.0},
        )


def test_calculate_report_differences_rejects_invalid_second_report():
    with pytest.raises(TypeError):
        calculate_report_differences(
            {"metric": 1.0},
            ["invalid"],
        )


def test_find_improved_metrics(
    first_report,
    second_report,
):
    result = find_improved_metrics(
        first_report,
        second_report,
    )

    assert set(result) == {
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
    }


def test_find_degraded_metrics():
    first = {
        "total_return": 0.20,
        "sharpe_ratio": 1.50,
    }

    second = {
        "total_return": 0.10,
        "sharpe_ratio": 1.20,
    }

    result = find_degraded_metrics(first, second)

    assert set(result) == {
        "total_return",
        "sharpe_ratio",
    }


def test_find_improved_metrics_returns_empty_when_no_improvement():
    first = {
        "total_return": 0.20,
    }

    second = {
        "total_return": 0.20,
    }

    assert find_improved_metrics(first, second) == []


def test_find_degraded_metrics_returns_empty_when_no_degradation():
    first = {
        "total_return": 0.20,
    }

    second = {
        "total_return": 0.20,
    }

    assert find_degraded_metrics(first, second) == []
