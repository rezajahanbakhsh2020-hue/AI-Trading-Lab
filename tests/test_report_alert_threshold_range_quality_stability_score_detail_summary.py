import pytest

from src.evaluation.report_alert_threshold_range_quality_stability_score_detail_summary import (
    build_quality_stability_score_detail_summary,
    calculate_quality_ratio,
    calculate_quality_stability_score,
    calculate_stability_ratio,
    classify_score,
)


def test_quality_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    assert calculate_quality_ratio(history, 2, 4) == 0.75


def test_stability_ratio():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    assert calculate_stability_ratio(history, 2, 4) == pytest.approx(1 / 3)


def test_quality_stability_score():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    expected = (0.75 + (1 / 3)) / 2

    assert calculate_quality_stability_score(
        history,
        2,
        4,
    ) == pytest.approx(expected)


def test_classify_stable():
    assert classify_score(0.80) == "stable"
    assert classify_score(1.0) == "stable"


def test_classify_acceptable():
    assert classify_score(0.60) == "acceptable"
    assert classify_score(0.79) == "acceptable"


def test_classify_unstable():
    assert classify_score(0.59) == "unstable"
    assert classify_score(0.0) == "unstable"


def test_custom_thresholds():
    assert classify_score(
        0.70,
        stable_threshold=0.75,
        acceptable_threshold=0.65,
    ) == "acceptable"


def test_build_summary():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_detail_summary(
        history,
        2,
        4,
    )

    expected_score = (0.75 + (1 / 3)) / 2

    assert result["snapshot_count"] == 4
    assert result["lower_bound"] == 2
    assert result["upper_bound"] == 4
    assert result["quality_ratio"] == pytest.approx(0.75)
    assert result["stability_ratio"] == pytest.approx(1 / 3)
    assert result["quality_stability_score"] == pytest.approx(
        expected_score
    )
    assert result["quality_stability_percentage"] == pytest.approx(
        expected_score * 100
    )
    assert result["stable_threshold"] == 0.80
    assert result["acceptable_threshold"] == 0.60
    assert result["classification"] == "unstable"


def test_empty_history():
    result = build_quality_stability_score_detail_summary(
        [],
        2,
        4,
    )

    assert result["snapshot_count"] == 0
    assert result["quality_ratio"] == 0.0
    assert result["stability_ratio"] == 0.0
    assert result["quality_stability_score"] == 0.0
    assert result["classification"] == "unstable"


def test_single_observation():
    history = [{"alert_count": 3}]

    result = build_quality_stability_score_detail_summary(
        history,
        2,
        4,
    )

    assert result["quality_ratio"] == 1.0
    assert result["stability_ratio"] == 1.0
    assert result["quality_stability_score"] == 1.0
    assert result["classification"] == "stable"


def test_invalid_history_type():
    with pytest.raises(TypeError):
        calculate_quality_ratio((), 2, 4)


def test_invalid_history_item():
    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}, 3], 2, 4)


def test_boolean_alert_count_rejected():
    with pytest.raises(TypeError):
        calculate_quality_ratio(
            [{"alert_count": True}],
            2,
            4,
        )


def test_negative_alert_count_rejected():
    with pytest.raises(ValueError):
        calculate_quality_ratio(
            [{"alert_count": -1}],
            2,
            4,
        )


def test_invalid_bounds():
    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}], "2", 4)

    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}], 2, "4")

    with pytest.raises(ValueError):
        calculate_quality_ratio([{"alert_count": 2}], -1, 4)

    with pytest.raises(ValueError):
        calculate_quality_ratio([{"alert_count": 2}], 5, 2)


def test_invalid_score():
    with pytest.raises(TypeError):
        classify_score("0.5")

    with pytest.raises(ValueError):
        classify_score(-0.1)

    with pytest.raises(ValueError):
        classify_score(1.1)


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        classify_score(0.5, stable_threshold=0.50, acceptable_threshold=0.60)

    with pytest.raises(ValueError):
        classify_score(0.5, stable_threshold=1.1)

    with pytest.raises(ValueError):
        classify_score(0.5, acceptable_threshold=-0.1)


def test_boolean_threshold_rejected():
    with pytest.raises(TypeError):
        classify_score(True)
