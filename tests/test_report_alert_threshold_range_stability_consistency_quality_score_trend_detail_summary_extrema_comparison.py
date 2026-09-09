import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison import (
    build_quality_score_trend_detail_summary_extrema_comparison,
)


def test_extrema_comparison_identifies_largest_positive_change() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = build_quality_score_trend_detail_summary_extrema_comparison(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 3

    assert result["quality_trend"] == "up"
    assert result["stability_trend"] == "up"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == pytest.approx(2 / 3)
    assert result["stability_change"] == pytest.approx(2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(2 / 3)

    assert result["positive_change_metrics"] == (
        "quality",
        "stability",
    )
    assert result["negative_change_metrics"] == ()

    assert result["quality_range"] == pytest.approx(2 / 3)
    assert result["stability_range"] == pytest.approx(2 / 3)
    assert result["consistency_range"] == pytest.approx(1.0)

    assert result["quality_peak_snapshot"] == 2
    assert result["quality_floor_snapshot"] == 0


def test_extrema_comparison_identifies_largest_negative_change() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_trend_detail_summary_extrema_comparison(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "down"
    assert result["stability_trend"] == "down"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == pytest.approx(-2 / 3)
    assert result["stability_change"] == pytest.approx(-2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(-2 / 3)

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == (
        "quality",
        "stability",
    )

    assert result["quality_peak_snapshot"] == 0
    assert result["quality_floor_snapshot"] == 1


def test_extrema_comparison_handles_flat_history() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    result = build_quality_score_trend_detail_summary_extrema_comparison(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == pytest.approx(0.0)
    assert result["stability_change"] == pytest.approx(0.0)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(0.0)

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == ()

    assert result["quality_range"] == pytest.approx(0.0)
    assert result["stability_range"] == pytest.approx(0.0)
    assert result["consistency_range"] == pytest.approx(0.0)


def test_extrema_comparison_handles_empty_history() -> None:
    result = build_quality_score_trend_detail_summary_extrema_comparison(
        [],
        lower_bound=2,
        upper_bound=4,
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

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == ()

    assert result["quality_range"] == 0.0
    assert result["stability_range"] == 0.0
    assert result["consistency_range"] == 0.0

    assert result["details"] == []


def test_extrema_comparison_contains_expected_keys() -> None:
    result = build_quality_score_trend_detail_summary_extrema_comparison(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
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
        "positive_change_metrics",
        "negative_change_metrics",
        "details",
    }

    assert set(result) == expected_keys
