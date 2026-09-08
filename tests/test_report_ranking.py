import pytest

from src.evaluation.report_ranking import (
    assign_report_ranks,
    get_top_reports,
    rank_reports,
)


@pytest.fixture
def reports():
    return [
        {
            "strategy": "A",
            "sharpe_ratio": 1.2,
        },
        {
            "strategy": "B",
            "sharpe_ratio": 1.8,
        },
        {
            "strategy": "C",
            "sharpe_ratio": 1.5,
        },
    ]


def test_rank_reports_descending(reports):
    result = rank_reports(
        reports,
        "sharpe_ratio",
    )

    assert [report["strategy"] for report in result] == [
        "B",
        "C",
        "A",
    ]


def test_rank_reports_ascending(reports):
    result = rank_reports(
        reports,
        "sharpe_ratio",
        descending=False,
    )

    assert [report["strategy"] for report in result] == [
        "A",
        "C",
        "B",
    ]


def test_rank_reports_preserves_report_contents(reports):
    result = rank_reports(
        reports,
        "sharpe_ratio",
    )

    assert result[0] == {
        "strategy": "B",
        "sharpe_ratio": 1.8,
    }


def test_rank_reports_returns_empty_for_empty_list():
    assert rank_reports(
        [],
        "sharpe_ratio",
    ) == []


def test_rank_reports_rejects_invalid_reports_argument():
    with pytest.raises(TypeError):
        rank_reports(
            ("invalid",),
            "sharpe_ratio",
        )


def test_rank_reports_rejects_invalid_report():
    with pytest.raises(TypeError):
        rank_reports(
            [{"sharpe_ratio": 1.0}, "invalid"],
            "sharpe_ratio",
        )


def test_rank_reports_rejects_invalid_metric(
    reports,
):
    with pytest.raises(TypeError):
        rank_reports(
            reports,
            123,
        )


def test_rank_reports_rejects_missing_metric():
    with pytest.raises(ValueError):
        rank_reports(
            [{"strategy": "A"}],
            "sharpe_ratio",
        )


def test_rank_reports_rejects_non_numeric_metric():
    with pytest.raises(ValueError):
        rank_reports(
            [{"sharpe_ratio": "1.5"}],
            "sharpe_ratio",
        )


def test_rank_reports_rejects_boolean_descending(
    reports,
):
    with pytest.raises(TypeError):
        rank_reports(
            reports,
            "sharpe_ratio",
            descending=1,
        )


def test_assign_report_ranks(reports):
    result = assign_report_ranks(
        reports,
        "sharpe_ratio",
    )

    assert result == [
        {
            "strategy": "B",
            "sharpe_ratio": 1.8,
            "rank": 1,
        },
        {
            "strategy": "C",
            "sharpe_ratio": 1.5,
            "rank": 2,
        },
        {
            "strategy": "A",
            "sharpe_ratio": 1.2,
            "rank": 3,
        },
    ]


def test_assign_report_ranks_does_not_modify_original_reports(
    reports,
):
    assign_report_ranks(
        reports,
        "sharpe_ratio",
    )

    assert "rank" not in reports[0]
    assert "rank" not in reports[1]
    assert "rank" not in reports[2]


def test_assign_report_ranks_supports_ascending(
    reports,
):
    result = assign_report_ranks(
        reports,
        "sharpe_ratio",
        descending=False,
    )

    assert [report["strategy"] for report in result] == [
        "A",
        "C",
        "B",
    ]

    assert [report["rank"] for report in result] == [
        1,
        2,
        3,
    ]


def test_get_top_reports_returns_requested_count(
    reports,
):
    result = get_top_reports(
        reports,
        "sharpe_ratio",
        count=2,
    )

    assert [report["strategy"] for report in result] == [
        "B",
        "C",
    ]


def test_get_top_reports_returns_all_when_count_is_large(
    reports,
):
    result = get_top_reports(
        reports,
        "sharpe_ratio",
        count=10,
    )

    assert len(result) == 3


def test_get_top_reports_supports_ascending(
    reports,
):
    result = get_top_reports(
        reports,
        "sharpe_ratio",
        count=2,
        descending=False,
    )

    assert [report["strategy"] for report in result] == [
        "A",
        "C",
    ]


def test_get_top_reports_rejects_zero_count(
    reports,
):
    with pytest.raises(ValueError):
        get_top_reports(
            reports,
            "sharpe_ratio",
            count=0,
        )


def test_get_top_reports_rejects_negative_count(
    reports,
):
    with pytest.raises(ValueError):
        get_top_reports(
            reports,
            "sharpe_ratio",
            count=-1,
        )


def test_get_top_reports_rejects_invalid_count(
    reports,
):
    with pytest.raises(TypeError):
        get_top_reports(
            reports,
            "sharpe_ratio",
            count="3",
        )
