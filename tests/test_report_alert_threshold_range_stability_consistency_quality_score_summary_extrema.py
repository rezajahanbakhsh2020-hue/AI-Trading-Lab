import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_extrema import (
    build_quality_score_summary_extrema,
)


def test_extrema_summary_builds_minimum_and_maximum_values() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    result = build_quality_score_summary_extrema(
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

    assert result["stability_ratio"] == pytest.approx(3 / 4)
    assert result["consistency_ratio"] == pytest.approx(3 / 4)
    assert result["quality_score"] == pytest.approx(3 / 4)
    assert result["quality_percentage"] == pytest.approx(75.0)


def test_extrema_summary_preserves_details() -> None:
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 6},
    ]

    result = build_quality_score_summary_extrema(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["details"] == [
        {
            "snapshot_index": 0,
            "alert_count": 1,
            "is_stable": False,
        },
        {
            "snapshot_index": 1,
            "alert_count": 4,
            "is_stable": True,
        },
        {
            "snapshot_index": 2,
            "alert_count": 6,
            "is_stable": False,
        },
    ]

    assert result["stable_count"] == 1
    assert result["unstable_count"] == 2


def test_extrema_summary_uses_first_snapshot_for_tied_minimum_and_maximum() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 5},
    ]

    result = build_quality_score_summary_extrema(
        history,
        lower_bound=2,
        upper_bound=5,
    )

    assert result["minimum_alert_count"] == 2
    assert result["maximum_alert_count"] == 5
    assert result["minimum_snapshot"] == 0
    assert result["maximum_snapshot"] == 1
    assert result["alert_count_range"] == 3


def test_extrema_summary_handles_empty_history() -> None:
    result = build_quality_score_summary_extrema(
        [],
        lower_bound=2,
        upper_bound=5,
    )

    assert result["snapshot_count"] == 0
    assert result["stable_count"] == 0
    assert result["unstable_count"] == 0

    assert result["minimum_alert_count"] is None
    assert result["maximum_alert_count"] is None
    assert result["minimum_snapshot"] is None
    assert result["maximum_snapshot"] is None
    assert result["alert_count_range"] == 0

    assert result["quality_score"] == pytest.approx(0.0)
    assert result["quality_percentage"] == pytest.approx(0.0)
    assert result["details"] == []
