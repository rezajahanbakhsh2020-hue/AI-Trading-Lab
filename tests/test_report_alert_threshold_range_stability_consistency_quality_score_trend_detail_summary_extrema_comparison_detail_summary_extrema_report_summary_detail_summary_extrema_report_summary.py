import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary,
)


def test_report_summary_preserves_core_report_values() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary(
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

    assert result["narrowest_metric"] == "quality"
    assert result["narrowest_range"] == pytest.approx(2 / 3)
    assert result["range_range"] == pytest.approx(1 / 3)


def test_report_summary_builds_direction_counts() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["direction_counts"] == {
        "up": 2,
        "down": 0,
        "flat": 1,
    }

    assert sum(result["direction_counts"].values()) == 3


def test_report_summary_preserves_metric_groups() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary(
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

    assert result["direction_counts"] == {
        "up": 0,
        "down": 2,
        "flat": 1,
    }


def test_report_summary_handles_all_flat_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary(
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

    assert result["direction_counts"] == {
        "up": 0,
        "down": 0,
        "flat": 3,
    }

    assert result["largest_change"] == pytest.approx(0.0)
    assert result["change_range"] == pytest.approx(0.0)


def test_report_summary_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema_report_summary(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0
    assert result["metric_count"] == 3

    assert result["direction_counts"] == {
        "up": 0,
        "down": 0,
        "flat": 3,
    }

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(0.0)

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(0.0)

    assert result["narrowest_metric"] == "quality"
    assert result["narrowest_range"] == pytest.approx(0.0)

    assert result["change_range"] == pytest.approx(0.0)
    assert result["range_range"] == pytest.approx(0.0)
