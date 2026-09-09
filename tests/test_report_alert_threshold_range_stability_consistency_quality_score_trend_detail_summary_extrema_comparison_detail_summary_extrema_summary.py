import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary,
)


def test_extrema_summary_orders_metrics_by_range() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["widest_metric"] == "consistency"
    assert result["widest_range"] == pytest.approx(1.0)

    assert result["range_order"] == (
        "consistency",
        "quality",
        "stability",
    )

    assert result["range_rank"] == {
        "consistency": 1,
        "quality": 2,
        "stability": 3,
    }


def test_extrema_summary_exposes_range_spread() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["range_spread"]["quality"] == pytest.approx(2 / 3)
    assert result["range_spread"]["stability"] == pytest.approx(2 / 3)
    assert result["range_spread"]["consistency"] == pytest.approx(1.0)


def test_extrema_summary_exposes_peak_and_floor_snapshots() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["peak_snapshots"] == {
        "quality": 2,
        "stability": 2,
        "consistency": 0,
    }

    assert result["floor_snapshots"] == {
        "quality": 0,
        "stability": 0,
        "consistency": 1,
    }


def test_extrema_summary_preserves_change_information() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
            [
                {"alert_count": 2},
                {"alert_count": 5},
                {"alert_count": 3},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["quality_change"] == pytest.approx(-1.0)
    assert result["stability_change"] == pytest.approx(-1 / 3)
    assert result["consistency_change"] == pytest.approx(-1.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(-1.0)
    assert result["overall_change_direction"] == "down"


def test_extrema_summary_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(0.0)

    assert result["range_order"] == (
        "quality",
        "stability",
        "consistency",
    )

    assert result["range_rank"] == {
        "quality": 1,
        "stability": 2,
        "consistency": 3,
    }

    assert result["range_spread"] == {
        "quality": 0.0,
        "stability": 0.0,
        "consistency": 0.0,
    }

    assert result["details"] == []


def test_extrema_summary_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_summary(
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
        "range_order",
        "range_rank",
        "range_spread",
        "peak_snapshots",
        "floor_snapshots",
        "metric_extrema",
        "details",
        "metric_details",
    }

    assert set(result) == expected_keys
