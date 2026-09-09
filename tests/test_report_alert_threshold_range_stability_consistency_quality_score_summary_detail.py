import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_summary_detail import (
    build_quality_score_summary_detail,
)


def test_build_summary_detail_contains_complete_summary() -> None:
    history = [
        {"alert_count": 1},
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 6},
    ]

    result = build_quality_score_summary_detail(
        history,
        lower_bound=1,
        upper_bound=3,
    )

    assert result["snapshot_count"] == 4
    assert result["lower_bound"] == 1
    assert result["upper_bound"] == 3

    assert result["stability_ratio"] == pytest.approx(0.75)
    assert result["consistency_ratio"] == pytest.approx(0.75)
    assert result["quality_score"] == pytest.approx(0.75)
    assert result["quality_percentage"] == pytest.approx(75.0)

    assert result["stable_count"] == 3
    assert result["unstable_count"] == 1


def test_details_are_created_for_each_snapshot() -> None:
    history = [
        {"alert_count": 1},
        {"alert_count": 4},
        {"alert_count": 3},
    ]

    result = build_quality_score_summary_detail(
        history,
        lower_bound=1,
        upper_bound=3,
    )

    assert result["details"] == [
        {
            "snapshot_index": 0,
            "alert_count": 1,
            "is_stable": True,
        },
        {
            "snapshot_index": 1,
            "alert_count": 4,
            "is_stable": False,
        },
        {
            "snapshot_index": 2,
            "alert_count": 3,
            "is_stable": True,
        },
    ]


def test_empty_history_returns_zero_summary() -> None:
    result = build_quality_score_summary_detail(
        [],
        lower_bound=1,
        upper_bound=3,
    )

    assert result["snapshot_count"] == 0
    assert result["stability_ratio"] == 0.0
    assert result["consistency_ratio"] == 0.0
    assert result["quality_score"] == 0.0
    assert result["quality_percentage"] == 0.0
    assert result["stable_count"] == 0
    assert result["unstable_count"] == 0
    assert result["details"] == []


def test_all_snapshots_inside_range() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    result = build_quality_score_summary_detail(
        history,
        lower_bound=2,
        upper_bound=3,
    )

    assert result["stable_count"] == 3
    assert result["unstable_count"] == 0
    assert result["stability_ratio"] == pytest.approx(1.0)
    assert result["consistency_ratio"] == pytest.approx(1.0)
    assert result["quality_score"] == pytest.approx(1.0)
    assert result["quality_percentage"] == pytest.approx(100.0)

    assert all(
        detail["is_stable"]
        for detail in result["details"]
    )


def test_all_snapshots_outside_range() -> None:
    history = [
        {"alert_count": 1},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_summary_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["stable_count"] == 0
    assert result["unstable_count"] == 3
    assert result["stability_ratio"] == pytest.approx(0.0)
    assert result["consistency_ratio"] == pytest.approx(0.0)
    assert result["quality_score"] == pytest.approx(0.0)
    assert result["quality_percentage"] == pytest.approx(0.0)


def test_boundary_values_are_stable() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 4},
    ]

    result = build_quality_score_summary_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["stable_count"] == 2
    assert result["unstable_count"] == 0

    assert result["details"][0]["is_stable"] is True
    assert result["details"][1]["is_stable"] is True


def test_expected_keys_are_returned() -> None:
    result = build_quality_score_summary_detail(
        [{"alert_count": 2}],
        lower_bound=1,
        upper_bound=3,
    )

    expected_keys = {
        "snapshot_count",
        "lower_bound",
        "upper_bound",
        "stability_ratio",
        "consistency_ratio",
        "quality_score",
        "quality_percentage",
        "stable_count",
        "unstable_count",
        "details",
    }

    assert set(result) == expected_keys


def test_invalid_history_item_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        build_quality_score_summary_detail(
            [{"alert_count": 2}, 3],  # type: ignore[list-item]
            lower_bound=1,
            upper_bound=3,
        )


def test_invalid_alert_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer alert_count"):
        build_quality_score_summary_detail(
            [{"alert_count": True}],
            lower_bound=1,
            upper_bound=3,
        )


def test_negative_alert_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        build_quality_score_summary_detail(
            [{"alert_count": -1}],
            lower_bound=1,
            upper_bound=3,
        )


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        build_quality_score_summary_detail(
            [{"alert_count": 2}],
            lower_bound=5,
            upper_bound=3,
        )


def test_negative_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        build_quality_score_summary_detail(
            [{"alert_count": 2}],
            lower_bound=-1,
            upper_bound=3,
        )


def test_boolean_bound_is_rejected() -> None:
    with pytest.raises(TypeError, match="lower_bound"):
        build_quality_score_summary_detail(
            [{"alert_count": 2}],
            lower_bound=True,  # type: ignore[arg-type]
            upper_bound=3,
        )
