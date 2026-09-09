import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary,
)


def test_detail_summary_reports_upward_metrics() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
            history,
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 3

    assert result["quality_change"] == pytest.approx(2 / 3)
    assert result["stability_change"] == pytest.approx(2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(2 / 3)
    assert result["overall_change_direction"] == "up"

    assert result["up_metrics"] == ("quality", "stability")
    assert result["down_metrics"] == ()
    assert result["flat_metrics"] == ("consistency",)

    assert result["up_metric_count"] == 2
    assert result["down_metric_count"] == 0
    assert result["flat_metric_count"] == 1
    assert result["metric_count"] == 3


def test_detail_summary_reports_downward_metrics() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
            history,
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["quality_change"] == pytest.approx(-2 / 3)
    assert result["stability_change"] == pytest.approx(-2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(-2 / 3)
    assert result["overall_change_direction"] == "down"

    assert result["up_metrics"] == ()
    assert result["down_metrics"] == ("quality", "stability")
    assert result["flat_metrics"] == ("consistency",)

    assert result["up_metric_count"] == 0
    assert result["down_metric_count"] == 2
    assert result["flat_metric_count"] == 1


def test_detail_summary_reports_flat_metrics() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
            history,
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["quality_change"] == pytest.approx(0.0)
    assert result["stability_change"] == pytest.approx(0.0)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(0.0)
    assert result["overall_change_direction"] == "flat"

    assert result["up_metrics"] == ()
    assert result["down_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["up_metric_count"] == 0
    assert result["down_metric_count"] == 0
    assert result["flat_metric_count"] == 3
    assert result["metric_count"] == 3


def test_detail_summary_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == 0.0
    assert result["stability_change"] == 0.0
    assert result["consistency_change"] == 0.0

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == 0.0
    assert result["overall_change_direction"] == "flat"

    assert result["up_metrics"] == ()
    assert result["down_metrics"] == ()
    assert result["flat_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["up_metric_count"] == 0
    assert result["down_metric_count"] == 0
    assert result["flat_metric_count"] == 3
    assert result["metric_count"] == 3

    assert result["details"] == []


def test_detail_summary_preserves_extrema_information() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert "quality_range" in result
    assert "stability_range" in result
    assert "consistency_range" in result

    assert "quality_peak_snapshot" in result
    assert "quality_floor_snapshot" in result
    assert "stability_peak_snapshot" in result
    assert "stability_floor_snapshot" in result
    assert "consistency_peak_snapshot" in result
    assert "consistency_floor_snapshot" in result


def test_detail_summary_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary(
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
        "quality_range",
        "stability_range",
        "consistency_range",
        "quality_peak_snapshot",
        "quality_floor_snapshot",
        "stability_peak_snapshot",
        "stability_floor_snapshot",
        "consistency_peak_snapshot",
        "consistency_floor_snapshot",
        "largest_change_metric",
        "largest_change",
        "overall_change_direction",
        "positive_change_metrics",
        "negative_change_metrics",
        "up_metrics",
        "down_metrics",
        "flat_metrics",
        "up_metric_count",
        "down_metric_count",
        "flat_metric_count",
        "metric_count",
        "details",
        "metric_details",
    }

    assert set(result) == expected_keys
