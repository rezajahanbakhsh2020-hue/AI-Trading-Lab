import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail_summary import (
    build_quality_score_trend_detail_summary,
)


def test_trend_detail_summary_returns_best_and_worst_snapshots() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = build_quality_score_trend_detail_summary(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 3
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

    assert result["quality_trend"] == "up"
    assert result["stability_trend"] == "up"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == pytest.approx(2 / 3)
    assert result["stability_change"] == pytest.approx(2 / 3)
    assert result["consistency_change"] == pytest.approx(0.0)

    assert result["best_quality_snapshot"] == 2
    assert result["best_quality_score"] == pytest.approx(2 / 3)
    assert result["worst_quality_snapshot"] == 0
    assert result["worst_quality_score"] == pytest.approx(0.0)

    assert result["best_stability_snapshot"] == 2
    assert result["best_stability_score"] == pytest.approx(2 / 3)
    assert result["worst_stability_snapshot"] == 0
    assert result["worst_stability_score"] == pytest.approx(0.0)

    assert result["best_consistency_snapshot"] == 0
    assert result["best_consistency_score"] == pytest.approx(1.0)
    assert result["worst_consistency_snapshot"] == 1
    assert result["worst_consistency_score"] == pytest.approx(0.0)

    assert len(result["details"]) == 3


def test_trend_detail_summary_handles_declining_history() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_trend_detail_summary(
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

    assert result["best_quality_snapshot"] == 0
    assert result["best_quality_score"] == pytest.approx(1.0)

    assert result["worst_quality_snapshot"] == 1
    assert result["worst_quality_score"] == pytest.approx(0.0)

    assert result["best_stability_snapshot"] == 0
    assert result["best_stability_score"] == pytest.approx(1.0)

    assert result["worst_stability_snapshot"] == 2
    assert result["worst_stability_score"] == pytest.approx(1 / 3)

    assert result["best_consistency_snapshot"] == 0
    assert result["best_consistency_score"] == pytest.approx(1.0)

    assert result["worst_consistency_snapshot"] == 1
    assert result["worst_consistency_score"] == pytest.approx(0.0)


def test_trend_detail_summary_handles_single_snapshot() -> None:
    result = build_quality_score_trend_detail_summary(
        [{"alert_count": 3}],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 1

    assert result["quality_trend"] == "flat"
    assert result["stability_trend"] == "flat"
    assert result["consistency_trend"] == "flat"

    assert result["quality_change"] == 0.0
    assert result["stability_change"] == 0.0
    assert result["consistency_change"] == 0.0

    assert result["best_quality_snapshot"] == 0
    assert result["best_quality_score"] == pytest.approx(1.0)
    assert result["worst_quality_snapshot"] == 0
    assert result["worst_quality_score"] == pytest.approx(1.0)

    assert result["best_stability_snapshot"] == 0
    assert result["best_stability_score"] == pytest.approx(1.0)
    assert result["worst_stability_snapshot"] == 0
    assert result["worst_stability_score"] == pytest.approx(1.0)

    assert result["best_consistency_snapshot"] == 0
    assert result["best_consistency_score"] == pytest.approx(1.0)
    assert result["worst_consistency_snapshot"] == 0
    assert result["worst_consistency_score"] == pytest.approx(1.0)


def test_trend_detail_summary_handles_empty_history() -> None:
    result = build_quality_score_trend_detail_summary(
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

    assert result["best_quality_snapshot"] is None
    assert result["best_quality_score"] is None
    assert result["worst_quality_snapshot"] is None
    assert result["worst_quality_score"] is None

    assert result["best_stability_snapshot"] is None
    assert result["best_stability_score"] is None
    assert result["worst_stability_snapshot"] is None
    assert result["worst_stability_score"] is None

    assert result["best_consistency_snapshot"] is None
    assert result["best_consistency_score"] is None
    assert result["worst_consistency_snapshot"] is None
    assert result["worst_consistency_score"] is None

    assert result["details"] == []


def test_trend_detail_summary_contains_expected_keys() -> None:
    result = build_quality_score_trend_detail_summary(
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
        "best_quality_snapshot",
        "best_quality_score",
        "worst_quality_snapshot",
        "worst_quality_score",
        "best_stability_snapshot",
        "best_stability_score",
        "worst_stability_snapshot",
        "worst_stability_score",
        "best_consistency_snapshot",
        "best_consistency_score",
        "worst_consistency_snapshot",
        "worst_consistency_score",
        "details",
    }

    assert set(result) == expected_keys
