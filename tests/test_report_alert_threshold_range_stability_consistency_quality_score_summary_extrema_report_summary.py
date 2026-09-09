import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_extrema_report_summary import (
    build_quality_score_summary_extrema_report_summary,
)


def test_extrema_report_summary_builds_compact_summary() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    result = build_quality_score_summary_extrema_report_summary(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["snapshot_count"] == 4
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 5

    assert result["stable_count"] == 3
    assert result["unstable_count"] == 1
    assert result["stability_status"] == "stable"

    assert result["minimum_alert_count"] == 2
    assert result["maximum_alert_count"] == 7
    assert result["minimum_snapshot"] == 2
    assert result["maximum_snapshot"] == 1
    assert result["alert_count_range"] == 5

    assert result["minimum_position"] == "inside_range"
    assert result["maximum_position"] == "above_range"
    assert result["extrema_spread"] == "crosses_range"

    assert result["positions"] == {
        "minimum": "inside_range",
        "maximum": "above_range",
    }

    assert result["position_counts"] == {
        "below_range": 0,
        "inside_range": 1,
        "above_range": 1,
        "none": 0,
    }

    assert result["quality_score"] == pytest.approx(3 / 4)
    assert result["quality_percentage"] == pytest.approx(75.0)


def test_extrema_report_summary_detects_unstable_status() -> None:
    history = [
        {"alert_count": 1},
        {"alert_count": 1},
        {"alert_count": 6},
        {"alert_count": 7},
    ]

    result = build_quality_score_summary_extrema_report_summary(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["stable_count"] == 0
    assert result["unstable_count"] == 4
    assert result["stability_status"] == "unstable"

    assert result["minimum_position"] == "below_range"
    assert result["maximum_position"] == "above_range"

    assert result["position_counts"] == {
        "below_range": 1,
        "inside_range": 0,
        "above_range": 1,
        "none": 0,
    }


def test_extrema_report_summary_detects_balanced_status() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 6},
    ]

    result = build_quality_score_summary_extrema_report_summary(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["stable_count"] == 1
    assert result["unstable_count"] == 1
    assert result["stability_status"] == "balanced"


def test_extrema_report_summary_handles_all_inside_range() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 5},
    ]

    result = build_quality_score_summary_extrema_report_summary(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["minimum_position"] == "inside_range"
    assert result["maximum_position"] == "inside_range"
    assert result["extrema_spread"] == "inside_range"

    assert result["position_counts"] == {
        "below_range": 0,
        "inside_range": 2,
        "above_range": 0,
        "none": 0,
    }

    assert result["quality_score"] == pytest.approx(1.0)
    assert result["quality_percentage"] == pytest.approx(100.0)


def test_extrema_report_summary_handles_empty_history() -> None:
    result = build_quality_score_summary_extrema_report_summary(
        [],
        lower_bound=2,
        upper_bound=5,
    )

    assert result["snapshot_count"] == 0
    assert result["stable_count"] == 0
    assert result["unstable_count"] == 0
    assert result["stability_status"] == "balanced"

    assert result["minimum_alert_count"] is None
    assert result["maximum_alert_count"] is None
    assert result["minimum_snapshot"] is None
    assert result["maximum_snapshot"] is None
    assert result["alert_count_range"] == 0

    assert result["minimum_position"] == "none"
    assert result["maximum_position"] == "none"
    assert result["extrema_spread"] == "flat"

    assert result["positions"] == {
        "minimum": "none",
        "maximum": "none",
    }

    assert result["position_counts"] == {
        "below_range": 0,
        "inside_range": 0,
        "above_range": 0,
        "none": 2,
    }

    assert result["quality_score"] == pytest.approx(0.0)
    assert result["quality_percentage"] == pytest.approx(0.0)
    assert result["details"] == []
