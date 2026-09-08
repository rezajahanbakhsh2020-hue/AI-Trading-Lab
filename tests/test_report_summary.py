import pytest

from src.evaluation.report_summary import (
    build_report_overview,
    build_report_summary,
    calculate_report_score,
)


@pytest.fixture
def sample_report():
    return {
        "total_return": 0.15,
        "max_drawdown": -0.06,
        "sharpe_ratio": 1.42,
        "sortino_ratio": 1.8,
        "calmar_ratio": 2.1,
        "win_rate": 0.65,
        "profit_factor": 1.9,
        "strategy": "XAU/USD",
    }


def test_build_report_summary_returns_main_metrics(sample_report):
    result = build_report_summary(sample_report)

    assert result == {
        "total_return": 0.15,
        "max_drawdown": -0.06,
        "sharpe_ratio": 1.42,
        "sortino_ratio": 1.8,
        "calmar_ratio": 2.1,
        "win_rate": 0.65,
        "profit_factor": 1.9,
    }


def test_build_report_summary_ignores_unrelated_fields():
    report = {
        "total_return": 0.10,
        "strategy": "XAU/USD",
        "custom_value": 123,
    }

    result = build_report_summary(report)

    assert result == {
        "total_return": 0.10,
    }


def test_build_report_summary_supports_partial_report():
    report = {
        "sharpe_ratio": 1.5,
    }

    assert build_report_summary(report) == {
        "sharpe_ratio": 1.5,
    }


def test_build_report_summary_supports_empty_report():
    assert build_report_summary({}) == {}


def test_build_report_summary_rejects_non_mapping():
    with pytest.raises(TypeError):
        build_report_summary(["invalid"])


def test_calculate_report_score(sample_report):
    result = calculate_report_score(sample_report)

    expected = (
        0.15
        + 1.42
        + 1.8
        + 2.1
        - 0.06
    )

    assert result == pytest.approx(expected)


def test_calculate_report_score_supports_partial_report():
    report = {
        "total_return": 0.20,
        "sharpe_ratio": 1.5,
    }

    assert calculate_report_score(report) == pytest.approx(1.7)


def test_calculate_report_score_returns_zero_for_empty_report():
    assert calculate_report_score({}) == 0.0


def test_calculate_report_score_ignores_non_numeric_values():
    report = {
        "total_return": "0.20",
        "sharpe_ratio": 1.5,
    }

    assert calculate_report_score(report) == pytest.approx(1.5)


def test_calculate_report_score_rejects_non_mapping():
    with pytest.raises(TypeError):
        calculate_report_score(["invalid"])


def test_build_report_overview(sample_report):
    result = build_report_overview(sample_report)

    assert result["summary"] == {
        "total_return": 0.15,
        "max_drawdown": -0.06,
        "sharpe_ratio": 1.42,
        "sortino_ratio": 1.8,
        "calmar_ratio": 2.1,
        "win_rate": 0.65,
        "profit_factor": 1.9,
    }

    assert result["score"] == pytest.approx(
        0.15 + 1.42 + 1.8 + 2.1 - 0.06
    )


def test_build_report_overview_rejects_non_mapping():
    with pytest.raises(TypeError):
        build_report_overview(["invalid"])
