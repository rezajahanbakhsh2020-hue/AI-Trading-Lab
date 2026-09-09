import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema,
)


def test_report_summary_detail_summary_extrema_builds_change_extrema() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["metric_count"] == 3

    assert result["changes"]["quality"] == pytest.approx(2 / 3)
    assert result["changes"]["stability"] == pytest.approx(2 / 3)
    assert result["changes"]["consistency"] == pytest.approx(0.0)

    assert result["largest_change_abs_metric"] == "quality"
    assert result["largest_change_abs"] == pytest.approx(2 / 3)

    assert result["smallest_change_abs_metric"] == "consistency"
    assert result["smallest_change_abs"] == pytest.approx(0.0)

    assert result["maximum_change_metric"] == "quality"
    assert result["maximum_change"] == pytest.approx(2 / 3)

    assert result["minimum_change_metric"] == "consistency"
    assert result["minimum_change"] == pytest.approx(0.0)

    assert result["change_range"] == pytest.approx(2 / 3)


def test_report_summary_detail_summary_extrema_builds_range_extrema() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["ranges"]["quality"] == pytest.approx(2 / 3)
    assert result["ranges"]["stability"] == pytest.approx(2 / 3)
    assert result["ranges"]["consistency"] == pytest.approx(1.0)

    assert result["widest_metric"] == "consistency"
    assert result["widest_range"] == pytest.approx(1.0)

    assert result["narrowest_metric"] == "quality"
    assert result["narrowest_range"] == pytest.approx(2 / 3)

    assert result["range_range"] == pytest.approx(1 / 3)


def test_report_summary_detail_summary_extrema_handles_declining_changes() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema(
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

    assert result["changes"]["quality"] == pytest.approx(-2 / 3)
    assert result["changes"]["stability"] == pytest.approx(-2 / 3)
    assert result["changes"]["consistency"] == pytest.approx(0.0)

    assert result["largest_change_abs_metric"] == "quality"
    assert result["largest_change_abs"] == pytest.approx(2 / 3)

    assert result["maximum_change_metric"] == "consistency"
    assert result["maximum_change"] == pytest.approx(0.0)

    assert result["minimum_change_metric"] == "quality"
    assert result["minimum_change"] == pytest.approx(-2 / 3)


def test_report_summary_detail_summary_extrema_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail_summary_extrema(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0
    assert result["metric_count"] == 3

    assert result["largest_change_abs_metric"] == "quality"
    assert result["largest_change_abs"] == pytest.approx(0.0)

    assert result["smallest_change_abs_metric"] == "quality"
    assert result["smallest_change_abs"] == pytest.approx(0.0)

    assert result["maximum_change_metric"] == "quality"
    assert result["maximum_change"] == pytest.approx(0.0)

    assert result["minimum_change_metric"] == "quality"
    assert result["minimum_change"] == pytest.approx(0.0)

    assert result["change_range"] == pytest.approx(0.0)

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(0.0)

    assert result["narrowest_metric"] == "quality"
    assert result["narrowest_range"] == pytest.approx(0.0)

    assert result["range_range"] == pytest.approx(0.0)

    assert result["changes"] == {
        "quality": pytest.approx(0.0),
        "stability": pytest.approx(0.0),
        "consistency": pytest.approx(0.0),
    }
