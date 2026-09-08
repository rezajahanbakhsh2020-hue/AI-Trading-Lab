import pytest

from src.evaluation.report_grouping import (
    count_reports_by_field,
    count_reports_by_status,
    group_reports_by_field,
    group_reports_by_status,
)


@pytest.fixture
def reports():
    return [
        {
            "strategy": "A",
            "status": "pass",
            "market": "gold",
        },
        {
            "strategy": "B",
            "status": "fail",
            "market": "gold",
        },
        {
            "strategy": "C",
            "status": "pass",
            "market": "forex",
        },
        {
            "strategy": "D",
            "status": "pass",
            "market": "gold",
        },
    ]


def test_group_reports_by_field(reports):
    result = group_reports_by_field(
        reports,
        "market",
    )

    assert [report["strategy"] for report in result["gold"]] == [
        "A",
        "B",
        "D",
    ]

    assert [report["strategy"] for report in result["forex"]] == [
        "C",
    ]


def test_group_reports_by_field_preserves_report_contents(
    reports,
):
    result = group_reports_by_field(
        reports,
        "status",
    )

    assert result["pass"][0] == reports[0]
    assert result["fail"][0] == reports[1]


def test_group_reports_by_field_preserves_order(
    reports,
):
    result = group_reports_by_field(
        reports,
        "market",
    )

    assert list(result.keys()) == [
        "gold",
        "forex",
    ]


def test_group_reports_by_field_empty_list():
    result = group_reports_by_field(
        [],
        "status",
    )

    assert result == {}


def test_group_reports_by_field_rejects_invalid_reports():
    with pytest.raises(TypeError):
        group_reports_by_field(
            ("invalid",),
            "status",
        )


def test_group_reports_by_field_rejects_invalid_report():
    with pytest.raises(TypeError):
        group_reports_by_field(
            [{"status": "pass"}, "invalid"],
            "status",
        )


def test_group_reports_by_field_rejects_invalid_field(
    reports,
):
    with pytest.raises(TypeError):
        group_reports_by_field(
            reports,
            123,
        )


def test_group_reports_by_field_rejects_missing_field():
    with pytest.raises(ValueError):
        group_reports_by_field(
            [{"strategy": "A"}],
            "status",
        )


def test_group_reports_by_field_rejects_unhashable_value():
    with pytest.raises(ValueError):
        group_reports_by_field(
            [{"tags": ["a", "b"]}],
            "tags",
        )


def test_group_reports_by_status(reports):
    result = group_reports_by_status(reports)

    assert len(result["pass"]) == 3
    assert len(result["fail"]) == 1

    assert [
        report["strategy"]
        for report in result["pass"]
    ] == [
        "A",
        "C",
        "D",
    ]


def test_group_reports_by_status_requires_status():
    with pytest.raises(ValueError):
        group_reports_by_status(
            [{"strategy": "A"}],
        )


def test_count_reports_by_field(reports):
    result = count_reports_by_field(
        reports,
        "market",
    )

    assert result == {
        "gold": 3,
        "forex": 1,
    }


def test_count_reports_by_field_empty_list():
    result = count_reports_by_field(
        [],
        "market",
    )

    assert result == {}


def test_count_reports_by_field_rejects_invalid_reports():
    with pytest.raises(TypeError):
        count_reports_by_field(
            ("invalid",),
            "market",
        )


def test_count_reports_by_field_rejects_invalid_field(
    reports,
):
    with pytest.raises(TypeError):
        count_reports_by_field(
            reports,
            123,
        )


def test_count_reports_by_field_rejects_missing_field():
    with pytest.raises(ValueError):
        count_reports_by_field(
            [{"strategy": "A"}],
            "status",
        )


def test_count_reports_by_status(reports):
    result = count_reports_by_status(reports)

    assert result == {
        "pass": 3,
        "fail": 1,
    }


def test_count_reports_by_status_empty_list():
    assert count_reports_by_status([]) == {}


def test_count_reports_by_status_rejects_missing_status():
    with pytest.raises(ValueError):
        count_reports_by_status(
            [{"strategy": "A"}],
        )
