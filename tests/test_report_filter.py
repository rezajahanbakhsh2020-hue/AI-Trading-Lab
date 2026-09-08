import pytest

from src.evaluation.report_filter import (
    filter_acceptable_reports,
    filter_reports,
    filter_reports_by_status,
)


@pytest.fixture
def reports():
    return [
        {
            "strategy": "A",
            "sharpe_ratio": 1.0,
            "status": "fail",
            "acceptable": False,
        },
        {
            "strategy": "B",
            "sharpe_ratio": 1.8,
            "status": "pass",
            "acceptable": True,
        },
        {
            "strategy": "C",
            "sharpe_ratio": 1.5,
            "status": "pass",
            "acceptable": True,
        },
    ]


def test_filter_reports_by_minimum(reports):
    result = filter_reports(
        reports,
        "sharpe_ratio",
        minimum=1.5,
    )

    assert [report["strategy"] for report in result] == [
        "B",
        "C",
    ]


def test_filter_reports_by_maximum(reports):
    result = filter_reports(
        reports,
        "sharpe_ratio",
        maximum=1.5,
    )

    assert [report["strategy"] for report in result] == [
        "A",
        "C",
    ]


def test_filter_reports_by_range(reports):
    result = filter_reports(
        reports,
        "sharpe_ratio",
        minimum=1.2,
        maximum=1.6,
    )

    assert [report["strategy"] for report in result] == [
        "C",
    ]


def test_filter_reports_without_bounds(reports):
    result = filter_reports(
        reports,
        "sharpe_ratio",
    )

    assert result == reports


def test_filter_reports_returns_empty_when_no_match(
    reports,
):
    result = filter_reports(
        reports,
        "sharpe_ratio",
        minimum=2.0,
    )

    assert result == []


def test_filter_reports_rejects_invalid_reports_argument():
    with pytest.raises(TypeError):
        filter_reports(
            ("invalid",),
            "sharpe_ratio",
        )


def test_filter_reports_rejects_invalid_report():
    with pytest.raises(TypeError):
        filter_reports(
            [{"sharpe_ratio": 1.0}, "invalid"],
            "sharpe_ratio",
        )


def test_filter_reports_rejects_invalid_metric(
    reports,
):
    with pytest.raises(TypeError):
        filter_reports(
            reports,
            123,
        )


def test_filter_reports_rejects_missing_metric():
    with pytest.raises(ValueError):
        filter_reports(
            [{"strategy": "A"}],
            "sharpe_ratio",
        )


def test_filter_reports_rejects_non_numeric_metric():
    with pytest.raises(ValueError):
        filter_reports(
            [{"sharpe_ratio": "1.5"}],
            "sharpe_ratio",
        )


def test_filter_reports_rejects_invalid_minimum(
    reports,
):
    with pytest.raises(TypeError):
        filter_reports(
            reports,
            "sharpe_ratio",
            minimum="1.0",
        )


def test_filter_reports_rejects_invalid_maximum(
    reports,
):
    with pytest.raises(TypeError):
        filter_reports(
            reports,
            "sharpe_ratio",
            maximum="2.0",
        )


def test_filter_reports_rejects_invalid_range(
    reports,
):
    with pytest.raises(ValueError):
        filter_reports(
            reports,
            "sharpe_ratio",
            minimum=2.0,
            maximum=1.0,
        )


def test_filter_reports_by_status(reports):
    result = filter_reports_by_status(
        reports,
        "pass",
    )

    assert [report["strategy"] for report in result] == [
        "B",
        "C",
    ]


def test_filter_reports_by_status_returns_empty(
    reports,
):
    result = filter_reports_by_status(
        reports,
        "incomplete",
    )

    assert result == []


def test_filter_reports_by_status_rejects_invalid_reports():
    with pytest.raises(TypeError):
        filter_reports_by_status(
            ("invalid",),
            "pass",
        )


def test_filter_reports_by_status_rejects_invalid_status(
    reports,
):
    with pytest.raises(TypeError):
        filter_reports_by_status(
            reports,
            123,
        )


def test_filter_acceptable_reports(reports):
    result = filter_acceptable_reports(reports)

    assert [report["strategy"] for report in result] == [
        "B",
        "C",
    ]


def test_filter_acceptable_reports_returns_empty():
    reports = [
        {
            "strategy": "A",
            "acceptable": False,
        }
    ]

    assert filter_acceptable_reports(reports) == []


def test_filter_acceptable_reports_rejects_invalid_reports():
    with pytest.raises(TypeError):
        filter_acceptable_reports(
            ("invalid",),
        )
