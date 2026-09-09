import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary_extrema_comparison_detail import (
    build_quality_score_trend_detail_summary_extrema_comparison_detail,
)


def test_comparison_detail_reports_metric_changes() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = build_quality_score_trend_detail_summary_extrema_comparison_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 3
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

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

    details = result["metric_details"]

    assert len(details) == 3

    quality = details[0]
    assert quality["metric"] == "quality"
    assert quality["change"] == pytest.approx(2 / 3)
    assert quality["direction"] == "up"
    assert quality["is_largest_change"] is True
    assert quality["is_positive"] is True
    assert quality["is_negative"] is False

    stability = details[1]
    assert stability["metric"] == "stability"
    assert stability["change"] == pytest.approx(2 / 3)
    assert stability["direction"] == "up"
    assert stability["is_largest_change"] is False
    assert stability["is_positive"] is True
    assert stability["is_negative"] is False

    consistency = details[2]
    assert consistency["metric"] == "consistency"
    assert consistency["change"] == pytest.approx(0.0)
    assert consistency["direction"] == "flat"
    assert consistency["is_largest_change"] is False
    assert consistency["is_positive"] is False
    assert consistency["is_negative"] is False


def test_comparison_detail_reports_negative_changes() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_trend_detail_summary_extrema_comparison_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

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

    details = result["metric_details"]

    assert details[0]["metric"] == "quality"
    assert details[0]["direction"] == "down"
    assert details[0]["is_largest_change"] is True
    assert details[0]["is_positive"] is False
    assert details[0]["is_negative"] is True

    assert details[1]["metric"] == "stability"
    assert details[1]["direction"] == "down"
    assert details[1]["is_largest_change"] is False
    assert details[1]["is_positive"] is False
    assert details[1]["is_negative"] is True

    assert details[2]["metric"] == "consistency"
    assert details[2]["direction"] == "flat"
    assert details[2]["is_largest_change"] is False
    assert details[2]["is_positive"] is False
    assert details[2]["is_negative"] is False


def test_comparison_detail_handles_flat_changes() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    result = build_quality_score_trend_detail_summary_extrema_comparison_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_change"] == pytest.approx(0.0)
    assert result["stability_change"] == pytest.approx(0.0)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["largest_change_metric"] == "quality"
    assert result["largest_change"] == pytest.approx(0.0)

    assert result["positive_change_metrics"] == ()
    assert result["negative_change_metrics"] == ()

    for detail in result["metric_details"]:
        assert detail["direction"] == "flat"
        assert detail["is_positive"] is False
        assert detail["is_negative"] is False

    assert result["metric_details"][0]["is_largest_change"] is True
    assert result["metric_details"][1]["is_largest_change"] is False
    assert result["metric_details"][2]["is_largest_change"] is False


def test_comparison_detail_handles_empty_history() -> None:
    result = build_quality_score_trend_detail_summary_extrema_comparison_detail(
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
    assert result["metric_details"] == [
        {
            "metric": "quality",
            "change": 0.0,
            "direction": "flat",
            "is_largest_change": True,
            "is_positive": False,
            "is_negative": False,
        },
        {
            "metric": "stability",
            "change": 0.0,
            "direction": "flat",
            "is_largest_change": False,
            "is_positive": False,
            "is_negative": False,
        },
        {
            "metric": "consistency",
            "change": 0.0,
            "direction": "flat",
            "is_largest_change": False,
            "is_positive": False,
            "is_negative": False,
        },
    ]

    assert result["details"] == []


def test_comparison_detail_contains_expected_top_level_keys() -> None:
    result = build_quality_score_trend_detail_summary_extrema_comparison_detail(
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
        "metric_details",
        "details",
    }

    assert set(result) == expected_keys


def test_comparison_detail_metric_details_have_expected_keys() -> None:
    result = build_quality_score_trend_detail_summary_extrema_comparison_detail(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
    )

    expected_detail_keys = {
        "metric",
        "change",
        "direction",
        "is_largest_change",
        "is_positive",
        "is_negative",
    }

    assert len(result["metric_details"]) == 3

    for detail in result["metric_details"]:
        assert set(detail) == expected_detail_keys
