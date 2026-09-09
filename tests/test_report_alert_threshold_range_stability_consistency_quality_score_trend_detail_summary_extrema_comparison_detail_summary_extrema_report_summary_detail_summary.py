import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary,
)


def test_report_summary_detail_summary_builds_metric_groups() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
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

    assert result["improving_metrics"] == (
        "quality",
        "stability",
    )
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == ("consistency",)

    assert result["positive_change_metrics"] == (
        "quality",
        "stability",
    )
    assert result["negative_change_metrics"] == ()

    assert result["improving_metric_count"] == 2
    assert result["declining_metric_count"] == 0
    assert result["flat_metric_count"] == 1
    assert result["positive_change_metric_count"] == 2
    assert result["negative_change_metric_count"] == 0


def test_report_summary_detail_summary_handles_declining_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
            [
                {"alert_count": 2},
                {"alert_count": 5},
                {"alert_count": 6},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["overall_change_direction"] == "down"

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == (
        "quality",
        "stability",
    )
    assert result["flat_metrics"] == ("consistency",)

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == (
        "quality",
        "stability",
    )

    assert result["improving_metric_count"] == 0
    assert result["declining_metric_count"] == 2
    assert result["flat_metric_count"] == 1
    assert result["positive_change_metric_count"] == 0
    assert result["negative_change_metric_count"] == 2


def test_report_summary_detail_summary_handles_all_flat_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
            [
                {"alert_count": 3},
                {"alert_count": 3},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["overall_change_direction"] == "flat"

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == ()

    assert result["improving_metric_count"] == 0
    assert result["declining_metric_count"] == 0
    assert result["flat_metric_count"] == 3
    assert result["positive_change_metric_count"] == 0
    assert result["negative_change_metric_count"] == 0


def test_report_summary_detail_summary_contains_status_counts() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert sum(result["status_counts"].values()) == 3

    for detail in result["metric_details"]:
        direction = detail["status"]["direction"]
        assert direction in result["status_counts"]


def test_report_summary_detail_summary_preserves_core_report_values() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

    assert result["quality_change"] == pytest.approx(2 / 3)
    assert result["stability_change"] == pytest.approx(2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(2 / 3)

    assert result["widest_metric"] == "consistency"
    assert result["widest_range"] == pytest.approx(1.0)


def test_report_summary_detail_summary_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0
    assert result["metric_count"] == 3

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(0.0)

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(0.0)

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == ()

    assert result["improving_metric_count"] == 0
    assert result["declining_metric_count"] == 0
    assert result["flat_metric_count"] == 3
    assert result["positive_change_metric_count"] == 0
    assert result["negative_change_metric_count"] == 0
