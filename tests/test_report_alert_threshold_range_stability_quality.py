import pytest

from src.evaluation.report_alert_threshold_range_stability_quality import (
    build_stability_quality_summary,
    calculate_stability_quality_percentage,
    calculate_stability_quality_score,
)


def test_stability_quality_score_combines_stability_and_quality() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    score = calculate_stability_quality_score(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert score == pytest.approx(2 / 3)


def test_stability_quality_score_is_one_when_all_snapshots_are_in_range() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 4},
    ]

    assert calculate_stability_quality_score(
        history,
        lower_bound=2,
        upper_bound=4,
    ) == pytest.approx(1.0)


def test_stability_quality_score_is_zero_when_no_snapshot_is_in_range() -> None:
    history = [
        {"alert_count": 0},
        {"alert_count": 5},
        {"alert_count": 7},
    ]

    assert calculate_stability_quality_score(
        history,
        lower_bound=2,
        upper_bound=4,
    ) == pytest.approx(0.0)


def test_stability_quality_percentage() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    assert calculate_stability_quality_percentage(
        history,
        lower_bound=2,
        upper_bound=4,
    ) == pytest.approx(200 / 3)


def test_build_stability_quality_summary() -> None:
    history = [
        {"alert_count": 2},
        {"alert_count": 5},
        {"alert_count": 3},
    ]

    result = build_stability_quality_summary(
        history,
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 3
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4

    assert result["stability_score"] == pytest.approx(2 / 3)
    assert result["quality_score"] == pytest.approx(2 / 3)
    assert result["stability_quality_score"] == pytest.approx(2 / 3)

    assert result["stability_percentage"] == pytest.approx(200 / 3)
    assert result["quality_percentage"] == pytest.approx(200 / 3)
    assert result["stability_quality_percentage"] == pytest.approx(200 / 3)

    assert result["stability_quality_gap"] == pytest.approx(0.0)


def test_build_stability_quality_summary_handles_empty_history() -> None:
    result = build_stability_quality_summary(
        [],
        lower_bound=2,
        upper_bound=4,
    )

    assert result["snapshot_count"] == 0
    assert result["stability_score"] == pytest.approx(0.0)
    assert result["quality_score"] == pytest.approx(0.0)
    assert result["stability_quality_score"] == pytest.approx(0.0)
    assert result["stability_percentage"] == pytest.approx(0.0)
    assert result["quality_percentage"] == pytest.approx(0.0)
    assert result["stability_quality_percentage"] == pytest.approx(0.0)
    assert result["stability_quality_gap"] == pytest.approx(0.0)
