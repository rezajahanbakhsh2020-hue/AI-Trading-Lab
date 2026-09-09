import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report,
)


def test_report_builds_complete_quality_score_report() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report(
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
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

    assert result["quality_change"] == pytest.approx(2 / 3)
    assert result["stability_change"] == pytest.approx(2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(2 / 3)

    assert result["overall_change_direction"] == "up"

    assert result["widest_metric"] == "consistency"
    assert result["widest_range"] == pytest.approx(1.0)

    assert result["largest_change_abs_metric"] == "quality"
    assert result["largest_change_abs"] == pytest.approx(2 / 3)

    assert result["smallest_change_abs_metric"] == "consistency"
    assert result["smallest_change_abs"] == pytest.approx(0.0)

    assert result["maximum_change_metric"] == "quality"
    assert result["maximum_change"] == pytest.approx(2 / 3)

    assert result["minimum_change_metric"] == "consistency"
    assert result["minimum_change"] == pytest.approx(0.0)

    assert result["change_range"] == pytest.approx(2 / 3)

    assert result["narrowest_metric"] == "quality"
    assert result["narrowest_range"] == pytest.approx(2 / 3)

    assert result["range_range"] == pytest.approx(1 / 3)


def test_report_preserves_metric_collections() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

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


def test_report_preserves_status_and_detail_information() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report(
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
    assert len(result["metric_details"]) == 3
    assert len(result["details"]) == 3

    for detail in result["metric_details"]:
        assert detail["metric"] in {
            "quality",
            "stability",
            "consistency",
        }
        assert detail["status"]["direction"] in {
            "up",
            "down",
            "flat",
        }


def test_report_handles_declining_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report(
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

    assert result["largest_change_abs_metric"] == "quality"
    assert result["largest_change_abs"] == pytest.approx(2 / 3)

    assert result["maximum_change_metric"] == "consistency"
    assert result["maximum_change"] == pytest.approx(0.0)

    assert result["minimum_change_metric"] == "quality"
    assert result["minimum_change"] == pytest.approx(-2 / 3)


def test_report_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0
    assert result["metric_count"] == 3

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(0.0)

    assert result["largest_change_abs_metric"] == "quality"
    assert result["largest_change_abs"] == pytest.approx(0.0)

    assert result["smallest_change_abs_metric"] == "quality"
    assert result["smallest_change_abs"] == pytest.approx(0.0)

    assert result["maximum_change_metric"] == "quality"
    assert result["maximum_change"] == pytest.approx(0.0)

    assert result["minimum_change_metric"] == "quality"
    assert result["minimum_change"] == pytest.approx(0.0)

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(0.0)

    assert result["narrowest_metric"] == "quality"
    assert result["narrowest_range"] == pytest.approx(0.0)

    assert result["change_range"] == pytest.approx(0.0)
    assert result["range_range"] == pytest.approx(0.0)

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )
