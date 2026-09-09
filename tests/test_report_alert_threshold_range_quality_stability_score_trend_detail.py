import pytest

from src.evaluation.report_alert_threshold_range_quality_stability_score_trend_detail import (
    build_quality_stability_score_trend_detail,
)


def test_detail_contains_per_snapshot_information() -> None:
    history = [
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_trend_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 3
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

    assert result["quality_series"] == pytest.approx(
        [1.0, 0.5, 2 / 3]
    )
    assert result["stability_series"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    assert result["score_series"] == pytest.approx(
        [1.0, 0.25, 1 / 3]
    )

    assert result["details"] == pytest.approx(
        [
            {
                "snapshot_index": 0,
                "quality": 1.0,
                "stability": 1.0,
                "score": 1.0,
            },
            {
                "snapshot_index": 1,
                "quality": 0.5,
                "stability": 0.0,
                "score": 0.25,
            },
            {
                "snapshot_index": 2,
                "quality": 2 / 3,
                "stability": 0.0,
                "score": 1 / 3,
            },
        ]
    )

    assert result["quality_trend"] == "down"
    assert result["stability_trend"] == "down"
    assert result["score_trend"] == "down"

    assert result["quality_change"] == pytest.approx(-1 / 3)
    assert result["stability_change"] == pytest.approx(-1.0)
    assert result["score_change"] == pytest.approx(-2 / 3)


def test_empty_history_returns_empty_details() -> None:
    result = build_quality_stability_score_trend_detail(
        [],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 0
    assert result["quality_series"] == []
    assert result["stability_series"] == []
    assert result["score_series"] == []
    assert result["details"] == []

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["score_trend"] == "flat"

    assert result["quality_change"] == 0.0
    assert result["stability_change"] == 0.0
    assert result["score_change"] == 0.0


def test_single_snapshot_contains_one_detail_record() -> None:
    result = build_quality_stability_score_trend_detail(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 1
    assert result["details"] == [
        {
            "snapshot_index": 0,
            "quality": 1.0,
            "stability": 1.0,
            "score": 1.0,
        }
    ]

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["score_trend"] == "flat"


def test_upward_quality_trend_is_preserved() -> None:
    history = [
        {"alert_count": 7},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_trend_detail(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "up"
    assert result["stability_trend"] == "down"
    assert result["score_trend"] == "up"


def test_bounds_are_preserved() -> None:
    result = build_quality_stability_score_trend_detail(
        [{"alert_count": 5}],
        lower_bound=3,
        upper_bound=6,
    )

    assert result["lower_bound"] == 3
    assert result["upper_bound"] == 6


def test_invalid_history_item_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        build_quality_stability_score_trend_detail(
            [{"alert_count": 3}, 4],  # type: ignore[list-item]
            lower_bound=2,
            upper_bound=4,
        )


def test_invalid_alert_count_is_rejected() -> None:
    with pytest.raises(TypeError, match="alert_count"):
        build_quality_stability_score_trend_detail(
            [{"alert_count": True}],
            lower_bound=2,
            upper_bound=4,
        )


def test_negative_alert_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_quality_stability_score_trend_detail(
            [{"alert_count": -1}],
            lower_bound=2,
            upper_bound=4,
        )


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="lower_bound"):
        build_quality_stability_score_trend_detail(
            [{"alert_count": 3}],
            lower_bound=5,
            upper_bound=2,
        )


def test_negative_bounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_quality_stability_score_trend_detail(
            [{"alert_count": 3}],
            lower_bound=-1,
            upper_bound=4,
        )


def test_boolean_bound_is_rejected() -> None:
    with pytest.raises(TypeError, match="lower_bound"):
        build_quality_stability_score_trend_detail(
            [{"alert_count": 3}],
            lower_bound=True,  # type: ignore[arg-type]
            upper_bound=4,
        )


def test_expected_keys_are_returned() -> None:
    result = build_quality_stability_score_trend_detail(
        [{"alert_count": 3}, {"alert_count": 4}],
        lower_bound=2,
        upper_bound=4,
    )

    expected_keys = {
        "snapshot_count",
        "lower_bound",
        "upper_bound",
        "quality_series",
        "stability_series",
        "score_series",
        "details",
        "quality_trend",
        "stability_trend",
        "score_trend",
        "quality_change",
        "stability_change",
        "score_change",
    }

    assert set(result) == expected_keys
