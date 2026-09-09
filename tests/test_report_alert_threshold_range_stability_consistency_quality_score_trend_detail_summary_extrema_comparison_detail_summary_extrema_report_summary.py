import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary,
)


def test_report_summary_identifies_improving_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
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

    assert result["quality_change"] == pytest.approx(2 / 3)
    assert result["stability_change"] == pytest.approx(2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["improving_metrics"] == (
        "quality",
        "stability",
    )
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == ("consistency",)

    assert result["improving_metric_count"] == 2
    assert result["declining_metric_count"] == 0
    assert result["flat_metric_count"] == 1


def test_report_summary_identifies_declining_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
            [
                {"alert_count": 2},
                {"alert_count": 5},
                {"alert_count": 6},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["quality_change"] == pytest.approx(-2 / 3)
    assert result["stability_change"] == pytest.approx(-2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == (
        "quality",
        "stability",
    )
    assert result["flat_metrics"] == ("consistency",)

    assert result["improving_metric_count"] == 0
    assert result["declining_metric_count"] == 2
    assert result["flat_metric_count"] == 1


def test_report_summary_identifies_all_flat_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
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

    assert result["improving_metric_count"] == 0
    assert result["declining_metric_count"] == 0
    assert result["flat_metric_count"] == 3


def test_report_summary_preserves_extrema_and_change_information() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(2 / 3)
    assert result["overall_change_direction"] == "up"

    assert result["widest_metric"] == "consistency"
    assert result["widest_range"] == pytest.approx(1.0)

    assert set(result["metric_status"]) == {
        "quality",
        "stability",
        "consistency",
    }

    assert set(result["metric_extrema"]) == {
        "quality",
        "stability",
        "consistency",
    }


def test_report_summary_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["consistency_trend"] == "flat"

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == 0.0
    assert result["overall_change_direction"] == "flat"

    assert result["improving_metrics"] == ()
    assert result["declining_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["improving_metric_count"] == 0
    assert result["declining_metric_count"] == 0
    assert result["flat_metric_count"] == 3

    assert result["details"] == []


def test_report_summary_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary(
            [{"alert_count": 3}],
            lower_bound=2,
            upper_bound=4,
        )
    )

    expected_keys = {
        "snapshot_count",
        "lower_bound",
        "upper_bound",
        "quality_trend",
        "stability_trend",
        "consistency_trend",
        "quality_change",
        "stability_change",
        "consistency_change",
        "largest_change_metric",
        "largest_change",
        "overall_change_direction",
        "widest_metric",
        "widest_range",
        "improving_metrics",
        "declining_metrics",
        "flat_metrics",
        "improving_metric_count",
        "declining_metric_count",
        "flat_metric_count",
        "metric_status",
        "metric_extrema",
        "details",
        "metric_details",
    }

    assert set(result) == expected_keys
