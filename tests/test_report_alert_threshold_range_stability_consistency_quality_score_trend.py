import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend import (
    build_quality_score_trend,
)


def test_quality_score_trend_contains_expected_series() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    result = build_quality_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 3
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

    assert result["quality_series"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )

    assert result["stability_series"] == pytest.approx(
        [1.0, 0.5, 2 / 3]
    )

    assert result["consistency_series"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )


def test_quality_trend_is_up_when_quality_improves() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    result = build_quality_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "up"
    assert result["quality_change"] == pytest.approx(1.0)


def test_quality_trend_is_down_when_quality_declines() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_trend"] == "down"
    assert result["quality_change"] == pytest.approx(-2 / 3)


def test_stability_trend_is_up_when_stability_improves() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = build_quality_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["stability_trend"] == "up"
    assert result["stability_change"] == pytest.approx(2 / 3)


def test_stability_trend_is_down_when_stability_declines() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["stability_trend"] == "down"
    assert result["stability_change"] == pytest.approx(-2 / 3)


def test_consistency_detects_state_changes() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    result = build_quality_score_trend(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["consistency_series"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )

    assert result["consistency_trend"] == "down"
    assert result["consistency_change"] == pytest.approx(-1.0)


def test_single_snapshot_returns_flat_trends() -> None:
    result = build_quality_score_trend(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["quality_series"] == pytest.approx([1.0])
    assert result["stability_series"] == pytest.approx([1.0])
    assert result["consistency_series"] == pytest.approx([1.0])

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == 0.0
    assert result["stability_change"] == 0.0
    assert result["consistency_change"] == 0.0


def test_empty_history_returns_empty_series() -> None:
    result = build_quality_score_trend(
        [],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 0
    assert result["quality_series"] == []
    assert result["stability_series"] == []
    assert result["consistency_series"] == []

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == 0.0
    assert result["stability_change"] == 0.0
    assert result["consistency_change"] == 0.0


def test_expected_keys_are_returned() -> None:
    result = build_quality_score_trend(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
    )

    expected_keys = {
        "snapshot_count",
        "lower_bound",
        "upper_bound",
        "quality_series",
        "stability_series",
        "consistency_series",
        "quality_trend",
        "stability_trend",
        "consistency_trend",
        "quality_change",
        "stability_change",
        "consistency_change",
    }

    assert set(result) == expected_keys
