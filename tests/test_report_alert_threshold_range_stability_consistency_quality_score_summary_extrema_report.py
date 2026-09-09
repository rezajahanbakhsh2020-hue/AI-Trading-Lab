import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_extrema_report import (
    build_quality_score_summary_extrema_report,
)


def test_extrema_report_builds_complete_range_report() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    result = build_quality_score_summary_extrema_report(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["snapshot_count"] == 4
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 5

    assert result["stable_count"] == 3
    assert result["unstable_count"] == 1

    assert result["minimum_alert_count"] == 2
    assert result["maximum_alert_count"] == 7
    assert result["minimum_snapshot"] == 2
    assert result["maximum_snapshot"] == 1
    assert result["alert_count_range"] == 5

    assert result["minimum_position"] == "inside_range"
    assert result["maximum_position"] == "above_range"
    assert result["extrema_spread"] == "crosses_range"

    assert result["stability_ratio"] == pytest.approx(3 / 4)
    assert result["consistency_ratio"] == pytest.approx(3 / 4)
    assert result["quality_score"] == pytest.approx(3 / 4)
    assert result["quality_percentage"] == pytest.approx(75.0)


def test_extrema_report_detects_extrema_inside_range() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
        {"alert_count": 5},
    ]

    result = build_quality_score_summary_extrema_report(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["minimum_alert_count"] == 2
    assert result["maximum_alert_count"] == 5
    assert result["minimum_position"] == "inside_range"
    assert result["maximum_position"] == "inside_range"
    assert result["extrema_spread"] == "inside_range"


def test_extrema_report_detects_below_and_above_range() -> None:
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 8},
    ]

    result = build_quality_score_summary_extrema_report(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["minimum_position"] == "below_range"
    assert result["maximum_position"] == "above_range"
    assert result["extrema_spread"] == "crosses_range"


def test_extrema_report_handles_flat_history() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 3},
        {"alert_count": 3},
    ]

    result = build_quality_score_summary_extrema_report(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["minimum_alert_count"] == 3
    assert result["maximum_alert_count"] == 3
    assert result["alert_count_range"] == 0
    assert result["minimum_position"] == "inside_range"
    assert result["maximum_position"] == "inside_range"
    assert result["extrema_spread"] == "flat"


def test_extrema_report_handles_empty_history() -> None:
    result = build_quality_score_summary_extrema_report(
        [],
        lower_bound=2,
        upper_bound=5,
    )

    assert result["snapshot_count"] == 0
    assert result["minimum_alert_count"] is None
    assert result["maximum_alert_count"] is None
    assert result["minimum_snapshot"] is None
    assert result["maximum_snapshot"] is None
    assert result["alert_count_range"] == 0

    assert result["minimum_position"] == "none"
    assert result["maximum_position"] == "none"
    assert result["extrema_spread"] == "flat"

    assert result["quality_score"] == pytest.approx(0.0)
    assert result["quality_percentage"] == pytest.approx(0.0)
    assert result["details"] == []
