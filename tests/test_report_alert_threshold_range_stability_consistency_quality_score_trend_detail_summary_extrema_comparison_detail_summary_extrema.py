import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema,
)


def test_extrema_summary_reports_widest_metric() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
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

    assert set(result["metric_extrema"]) == {
        "quality",
        "stability",
        "consistency",
    }

    assert result["metric_extrema"]["quality"]["range"] == pytest.approx(2 / 3)
    assert result["metric_extrema"]["stability"]["range"] == pytest.approx(2 / 3)
    assert result["metric_extrema"]["consistency"]["range"] == pytest.approx(1.0)

    assert result["widest_metric"] == "consistency"
    assert result["widest_range"] == pytest.approx(1.0)


def test_extrema_summary_preserves_peak_and_floor_snapshots() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
            [
                {"alert_count": 5},
                {"alert_count": 3},
                {"alert_count": 2},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["metric_extrema"]["quality"]["peak_snapshot"] == 2
    assert result["metric_extrema"]["quality"]["floor_snapshot"] == 0

    assert result["metric_extrema"]["stability"]["peak_snapshot"] == 2
    assert result["metric_extrema"]["stability"]["floor_snapshot"] == 0

    assert result["metric_extrema"]["consistency"]["peak_snapshot"] == 0
    assert result["metric_extrema"]["consistency"]["floor_snapshot"] == 1


def test_extrema_summary_handles_equal_ranges_deterministically() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
            [
                {"alert_count": 2},
                {"alert_count": 5},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["metric_extrema"]["quality"]["range"] == pytest.approx(1.0)
    assert result["metric_extrema"]["stability"]["range"] == pytest.approx(0.5)
    assert result["metric_extrema"]["consistency"]["range"] == pytest.approx(1.0)

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(1.0)


def test_extrema_summary_handles_empty_history() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
            [],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["snapshot_count"] == 0

    assert result["metric_extrema"]["quality"]["range"] == pytest.approx(0.0)
    assert result["metric_extrema"]["stability"]["range"] == pytest.approx(0.0)
    assert result["metric_extrema"]["consistency"]["range"] == pytest.approx(0.0)

    assert result["widest_metric"] == "quality"
    assert result["widest_range"] == pytest.approx(0.0)

    assert result["details"] == []


def test_extrema_summary_preserves_change_summary() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
            [
                {"alert_count": 2},
                {"alert_count": 5},
                {"alert_count": 3},
            ],
            lower_bound=2,
            upper_bound=4,
        )
    )

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(-1.0)
    assert result["overall_change_direction"] == "down"

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == (
        "quality",
        "stability",
        "consistency",
    )


def test_extrema_summary_contains_expected_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
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
        "positive_change_metrics",
        "negative_change_metrics",
        "up_metrics",
        "down_metrics",
        "flat_metrics",
        "metric_extrema",
        "widest_metric",
        "widest_range",
        "details",
        "metric_details",
    }

    assert set(result) == expected_keys


def test_extrema_summary_contains_expected_metric_extrema_keys() -> None:
    result = (
        build_quality_score_trend_detail_summary_extrema_comparison_detail_summary_extrema(
            [{"alert_count": 3}],
            lower_bound=2,
            upper_bound=4,
        )
    )

    expected_keys = {
        "range",
        "peak_snapshot",
        "floor_snapshot",
    }

    for metric in ("quality", "stability", "consistency"):
        assert set(result["metric_extrema"][metric]) == expected_keys
