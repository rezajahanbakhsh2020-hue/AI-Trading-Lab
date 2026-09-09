import pytest

from src.evaluation.report_alert_threshold_range_quality_stability_score_summary_detail import (
    build_quality_stability_score_summary_detail,
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


def test_classification():
    assert classify_score(0.80) == "stable"
    assert classify_score(0.60) == "acceptable"
    assert classify_score(0.59) == "unstable"


def test_summary_detail():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
        {"alert_count": 7},
        {"alert_count": 4},
    ]

    result = build_quality_stability_score_summary_detail(
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
    assert result["quality_gap_to_stable"] == pytest.approx(0.05)
    assert result["stability_gap_to_stable"] == pytest.approx(0.80 - 1 / 3)
    assert result["score_gap_to_stable"] == pytest.approx(
        0.80 - expected_score
    )


def test_stable_result_has_zero_score_gap():
    history = [
        {"alert_count": 2},
        {"alert_count": 3},
    ]

    result = build_quality_stability_score_summary_detail(
        history,
        2,
        4,
    )

    assert result["quality_ratio"] == 1.0
    assert result["stability_ratio"] == 1.0
    assert result["quality_stability_score"] == 1.0
    assert result["classification"] == "stable"
    assert result["quality_gap_to_stable"] == 0.0
    assert result["stability_gap_to_stable"] == 0.0
    assert result["score_gap_to_stable"] == 0.0


def test_empty_history():
    result = build_quality_stability_score_summary_detail(
        [],
        2,
        4,
    )

    assert result["snapshot_count"] == 0
    assert result["quality_ratio"] == 0.0
    assert result["stability_ratio"] == 0.0
    assert result["quality_stability_score"] == 0.0
    assert result["classification"] == "unstable"
    assert result["quality_gap_to_stable"] == 0.80
    assert result["stability_gap_to_stable"] == 0.80
    assert result["score_gap_to_stable"] == 0.80


def test_single_observation():
    result = build_quality_stability_score_summary_detail(
        [{"alert_count": 3}],
        2,
        4,
    )

    assert result["quality_ratio"] == 1.0
    assert result["stability_ratio"] == 1.0
    assert result["quality_stability_score"] == 1.0
    assert result["classification"] == "stable"


def test_custom_thresholds():
    result = build_quality_stability_score_summary_detail(
        [
            {"alert_count": 2},
            {"alert_count": 3},
            {"alert_count": 7},
            {"alert_count": 4},
        ],
        2,
        4,
        stable_threshold=0.75,
        acceptable_threshold=0.50,
    )

    assert result["classification"] == "acceptable"


def test_invalid_history():
    with pytest.raises(TypeError):
        calculate_quality_ratio((), 2, 4)

    with pytest.raises(TypeError):
        calculate_quality_ratio([{"alert_count": 2}, 3], 2, 4)


def test_invalid_alert_count():
    with pytest.raises(TypeError):
        calculate_quality_ratio(
            [{"alert_count": True}],
            2,
            4,
        )

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


def test_invalid_classification_thresholds():
    with pytest.raises(TypeError):
        classify_score("0.5")

    with pytest.raises(ValueError):
        classify_score(-0.1)

    with pytest.raises(ValueError):
        classify_score(1.1)

    with pytest.raises(ValueError):
        classify_score(
            0.5,
            stable_threshold=0.50,
            acceptable_threshold=0.60,
        )


def test_boolean_threshold_is_rejected():
    with pytest.raises(TypeError):
        classify_score(True)
