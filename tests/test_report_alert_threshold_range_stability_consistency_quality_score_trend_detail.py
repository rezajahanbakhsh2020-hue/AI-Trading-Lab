import pytest

from src.evaluation.report_alert_threshold_range_stability_consistency_quality_score_trend_detail import (
    build_quality_score_trend_detail,
)


def test_trend_detail_returns_per_snapshot_scores_and_changes() -> None:
    history = [
        {"alert_count": 5},
        {"alert_count": 3},
        {"alert_count": 2},
    ]

    result = build_quality_score_trend_detail(
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

    details = result["details"]

    assert len(details) == 3

    assert details[0]["snapshot_index"] == 0
    assert details[0]["alert_count"] == 5
    assert details[0]["quality_score"] == pytest.approx(0.0)
    assert details[0]["stability_score"] == pytest.approx(0.0)
    assert details[0]["consistency_score"] == pytest.approx(1.0)
    assert details[0]["quality_change"] == pytest.approx(0.0)
    assert details[0]["stability_change"] == pytest.approx(0.0)
    assert details[0]["consistency_change"] == pytest.approx(0.0)

    assert details[1]["snapshot_index"] == 1
    assert details[1]["alert_count"] == 3
    assert details[1]["quality_score"] == pytest.approx(0.5)
    assert details[1]["stability_score"] == pytest.approx(0.5)
    assert details[1]["consistency_score"] == pytest.approx(0.0)
    assert details[1]["quality_change"] == pytest.approx(0.5)
    assert details[1]["stability_change"] == pytest.approx(0.5)
    assert details[1]["consistency_change"] == pytest.approx(-1.0)

    assert details[2]["snapshot_index"] == 2
    assert details[2]["alert_count"] == 2
    assert details[2]["quality_score"] == pytest.approx(2 / 3)
    assert details[2]["stability_score"] == pytest.approx(2 / 3)
    assert details[2]["consistency_score"] == pytest.approx(1.0)
    assert details[2]["quality_change"] == pytest.approx(1 / 6)
    assert details[2]["stability_change"] == pytest.approx(1 / 6)
    assert details[2]["consistency_change"] == pytest.approx(1.0)


def test_trend_detail_reports_declining_quality() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 6},
    ]

    result = build_quality_score_trend_detail(
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

    assert len(result["details"]) == 3

    assert result["details"][0]["quality_score"] == pytest.approx(1.0)
    assert result["details"][1]["quality_score"] == pytest.approx(0.0)
    assert result["details"][2]["quality_score"] == pytest.approx(1 / 3)

    assert result["details"][1]["quality_change"] == pytest.approx(-1.0)
    assert result["details"][2]["quality_change"] == pytest.approx(1 / 3)


def test_trend_detail_handles_single_snapshot() -> None:
    result = build_quality_score_trend_detail(
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

    assert len(result["details"]) == 1

    detail = result["details"][0]

    assert detail["snapshot_index"] == 0
    assert detail["alert_count"] == 3
    assert detail["quality_score"] == pytest.approx(1.0)
    assert detail["stability_score"] == pytest.approx(1.0)
    assert detail["consistency_score"] == pytest.approx(1.0)
    assert detail["quality_change"] == pytest.approx(0.0)
    assert detail["stability_change"] == pytest.approx(0.0)
    assert detail["consistency_change"] == pytest.approx(0.0)


def test_trend_detail_handles_empty_history() -> None:
    result = build_quality_score_trend_detail(
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
    assert result["details"] == []


def test_trend_detail_contains_expected_keys() -> None:
    result = build_quality_score_trend_detail(
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
        "details",
    }

    assert set(result) == expected_keys

    expected_detail_keys = {
        "snapshot_index",
        "alert_count",
        "quality_score",
        "stability_score",
        "consistency_score",
        "quality_change",
        "stability_change",
        "consistency_change",
    }

    assert set(result["details"][0]) == expected_detail_keys
