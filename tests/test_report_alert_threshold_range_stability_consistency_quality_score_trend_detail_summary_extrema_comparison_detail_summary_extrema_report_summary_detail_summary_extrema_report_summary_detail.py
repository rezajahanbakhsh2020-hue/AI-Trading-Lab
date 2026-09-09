import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail,
)


def test_report_summary_detail_builds_detailed_metric_records() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 3
    assert result["metric_count"] == 3
    assert len(result["metric_details"]) == 3

    assert [item["metric"] for item in result["metric_details"]] == [
        "quality",
        "stability",
        "consistency",
    ]


def test_report_summary_detail_classifies_metric_changes() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    quality = result["metric_details"][0]
    stability = result["metric_details"][1]
    consistency = result["metric_details"][2]

    assert quality["direction"] == "up"
    assert quality["change_class"] == "positive"
    assert quality["is_improving"] is True
    assert quality["is_declining"] is False
    assert quality["is_flat"] is False

    assert stability["direction"] == "up"
    assert stability["change_class"] == "positive"

    assert consistency["direction"] == "flat"
    assert consistency["change_class"] == "neutral"
    assert consistency["is_improving"] is False
    assert consistency["is_declining"] is False
    assert consistency["is_flat"] is True


def test_report_summary_detail_preserves_extrema_information() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    for detail in result["metric_details"]:
        assert "peak_snapshot" in detail
        assert "floor_snapshot" in detail
        assert "range_rank" in detail

    assert result["metric_details"][0]["range"] == pytest.approx(2 / 3)
    assert result["metric_details"][1]["range"] == pytest.approx(2 / 3)
    assert result["metric_details"][2]["range"] == pytest.approx(1.0)


def test_report_summary_detail_builds_metric_groups() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
            [
                {"alert_count": 2},
                {"alert_count": 5},
                {"alert_count": 6},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == (
        "quality",
        "stability",
    )
    assert result["flat_metrics"] == ("consistency",)

    assert result["direction_counts"] == {
        "up": 0,
        "down": 2,
        "flat": 1,
    }


def test_report_summary_detail_handles_all_flat_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
            [
                {"alert_count": 3},
                {"alert_count": 3},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["direction_counts"] == {
        "up": 0,
        "down": 0,
        "flat": 3,
    }

    for detail in result["metric_details"]:
        assert detail["direction"] == "flat"
        assert detail["change_class"] == "neutral"
        assert detail["is_flat"] is True


def test_report_summary_detail_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary_detail(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0
    assert result["metric_count"] == 3
    assert len(result["metric_details"]) == 3

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["direction_counts"] == {
        "up": 0,
        "down": 0,
        "flat": 3,
    }

    for detail in result["metric_details"]:
        assert detail["change"] == pytest.approx(0.0)
        assert detail["direction"] == "flat"
        assert detail["change_class"] == "neutral"
        assert detail["is_flat"] is True
