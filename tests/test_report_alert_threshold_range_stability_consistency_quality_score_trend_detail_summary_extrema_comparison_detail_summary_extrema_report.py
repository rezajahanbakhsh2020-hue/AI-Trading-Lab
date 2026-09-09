import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report,
)


def test_report_builds_metric_status() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
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
    assert result["largest_change_metric"] == "quality"
    assert result["overall_change_direction"] == "up"

    assert set(result["metric_status"]) == {
        "quality",
        "stability",
        "consistency",
    }

    assert result["metric_status"]["quality"]["change"] == pytest.approx(2 / 3)
    assert result["metric_status"]["quality"]["direction"] == "up"
    assert result["metric_status"]["quality"]["range"] == pytest.approx(2 / 3)
    assert result["metric_status"]["quality"]["range_rank"] == 2

    assert result["metric_status"]["stability"]["change"] == pytest.approx(2 / 3)
    assert result["metric_status"]["stability"]["direction"] == "up"
    assert result["metric_status"]["stability"]["range"] == pytest.approx(2 / 3)
    assert result["metric_status"]["stability"]["range_rank"] == 3

    assert result["metric_status"]["consistency"]["change"] == pytest.approx(0.0)
    assert result["metric_status"]["consistency"]["direction"] == "flat"
    assert result["metric_status"]["consistency"]["range"] == pytest.approx(1.0)
    assert result["metric_status"]["consistency"]["range_rank"] == 1


def test_report_preserves_peak_and_floor_information() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
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

    assert result["metric_status"]["quality"]["peak_snapshot"] == 2
    assert result["metric_status"]["quality"]["floor_snapshot"] == 0


def test_report_handles_declining_metrics() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
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

    assert result["overall_change_direction"] == "down"

    assert result["metric_status"]["quality"]["direction"] == "down"
    assert result["metric_status"]["stability"]["direction"] == "down"
    assert result["metric_status"]["consistency"]["direction"] == "flat"


def test_report_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
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

    for metric in ("quality", "stability", "consistency"):
        assert result["metric_status"][metric]["change"] == 0.0
        assert result["metric_status"][metric]["direction"] == "flat"
        assert result["metric_status"][metric]["range"] == 0.0

    assert result["details"] == []


def test_report_contains_expected_top_level_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
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
        "metric_status",
        "details",
        "metric_details",
    }

    assert set(result) == expected_keys


def test_report_metric_status_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema_report(
            [{"alert_count": 3}],
            lower_bound=2,
            upper_bound=4,
        )
    )

    expected_keys = {
        "change",
        "direction",
        "range",
        "peak_snapshot",
        "floor_snapshot",
        "range_rank",
    }

    for metric in ("quality", "stability", "consistency"):
        assert set(result["metric_status"][metric]) == expected_keys
