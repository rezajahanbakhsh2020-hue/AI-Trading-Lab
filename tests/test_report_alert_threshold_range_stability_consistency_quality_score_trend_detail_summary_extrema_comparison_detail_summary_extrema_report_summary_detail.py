import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail,
)


def test_report_summary_detail_builds_per_metric_details() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
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
    assert len(result["metric_details"]) == 3

    quality = result["metric_details"][0]
    assert quality["metric"] == "quality"
    assert quality["change"] == pytest.approx(2 / 3)
    assert quality["direction"] == "up"
    assert quality["range"] == pytest.approx(2 / 3)
    assert quality["peak_snapshot"] == 2
    assert quality["floor_snapshot"] == 0
    assert quality["range_rank"] is None
    assert quality["is_widest"] is False
    assert quality["is_largest_change"] is True
    assert quality["is_improving"] is True
    assert quality["is_declining"] is False
    assert quality["is_flat"] is False

    stability = result["metric_details"][1]
    assert stability["metric"] == "stability"
    assert stability["change"] == pytest.approx(2 / 3)
    assert stability["direction"] == "up"
    assert stability["range"] == pytest.approx(2 / 3)
    assert stability["peak_snapshot"] == 2
    assert stability["floor_snapshot"] == 0
    assert stability["range_rank"] is None
    assert stability["is_widest"] is False
    assert stability["is_largest_change"] is False
    assert stability["is_improving"] is True
    assert stability["is_declining"] is False
    assert stability["is_flat"] is False

    consistency = result["metric_details"][2]
    assert consistency["metric"] == "consistency"
    assert consistency["change"] == pytest.approx(0.0)
    assert consistency["direction"] == "flat"
    assert consistency["range"] == pytest.approx(1.0)
    assert consistency["peak_snapshot"] == 0
    assert consistency["floor_snapshot"] == 1
    assert consistency["range_rank"] is None
    assert consistency["is_widest"] is True
    assert consistency["is_largest_change"] is False
    assert consistency["is_improving"] is False
    assert consistency["is_declining"] is False
    assert consistency["is_flat"] is True


def test_report_summary_detail_handles_declining_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
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

    quality = result["metric_details"][0]
    assert quality["direction"] == "down"
    assert quality["is_declining"] is True
    assert quality["is_improving"] is False
    assert quality["is_flat"] is False

    stability = result["metric_details"][1]
    assert stability["direction"] == "down"
    assert stability["is_declining"] is True

    consistency = result["metric_details"][2]
    assert consistency["direction"] == "flat"
    assert consistency["is_declining"] is False
    assert consistency["is_flat"] is True


def test_report_summary_detail_handles_all_flat_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
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

    for detail in result["metric_details"]:
        assert detail["direction"] == "flat"
        assert detail["is_improving"] is False
        assert detail["is_declining"] is False
        assert detail["is_flat"] is True


def test_report_summary_detail_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0
    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == 0.0
    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == 0.0

    assert len(result["metric_details"]) == 3

    for detail in result["metric_details"]:
        assert detail["change"] == pytest.approx(0.0)
        assert detail["direction"] == "flat"
        assert detail["range"] == pytest.approx(0.0)
        assert detail["is_improving"] is False
        assert detail["is_declining"] is False
        assert detail["is_flat"] is True

    assert result["metric_details"][0]["is_largest_change"] is True
    assert result["metric_details"][0]["is_widest"] is True
    assert result["metric_details"][1]["is_largest_change"] is False
    assert result["metric_details"][2]["is_largest_change"] is False


def test_report_summary_detail_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
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
        "metric_details",
        "details",
    }

    assert set(result) == expected_keys


def test_metric_detail_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report_summary_detail(
            [{"alert_count": 3}],
            lower_bound=2,
            upper_bound=4,
        )
    )

    expected_detail_keys = {
        "metric",
        "change",
        "direction",
        "range",
        "peak_snapshot",
        "floor_snapshot",
        "range_rank",
        "is_widest",
        "is_largest_change",
        "is_improving",
        "is_declining",
        "is_flat",
        "status",
    }

    for detail in result["metric_details"]:
        assert set(detail) == expected_detail_keys
