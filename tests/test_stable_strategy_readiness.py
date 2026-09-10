import pytest

from src.evaluation.stable_strategy_readiness import (
    calculate_readiness_score,
    evaluate_strategy_readiness,
    is_strategy_ready,
)


def test_calculate_readiness_score():
    score = calculate_readiness_score(
        strategy_stability_score=0.90,
        portfolio_stability_score=0.80,
    )

    assert score == pytest.approx(0.87)


def test_custom_weights_are_supported():
    score = calculate_readiness_score(
        strategy_stability_score=0.90,
        portfolio_stability_score=0.60,
        strategy_weight=0.50,
        portfolio_weight=0.50,
    )

    assert score == pytest.approx(0.75)


def test_ready_when_all_gates_pass():
    result = evaluate_strategy_readiness(
        strategy="momentum",
        strategy_stability_score=0.90,
        portfolio_stability_score=0.85,
    )

    assert result["ready"] is True
    assert result["status"] == "READY"
    assert result["failed_gates"] == []
    assert result["strategy_gate"] is True
    assert result["portfolio_gate"] is True
    assert result["readiness_gate"] is True


def test_blocked_when_strategy_stability_fails():
    result = evaluate_strategy_readiness(
        strategy="momentum",
        strategy_stability_score=0.60,
        portfolio_stability_score=0.90,
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert "strategy_stability" in result["failed_gates"]


def test_blocked_when_portfolio_stability_fails():
    result = evaluate_strategy_readiness(
        strategy="momentum",
        strategy_stability_score=0.90,
        portfolio_stability_score=0.60,
    )

    assert result["ready"] is False
    assert result["status"] == "BLOCKED"
    assert "portfolio_stability" in result["failed_gates"]


def test_blocked_when_combined_score_fails():
    result = evaluate_strategy_readiness(
        strategy="momentum",
        strategy_stability_score=0.71,
        portfolio_stability_score=0.70,
        minimum_strategy_stability=0.70,
        minimum_portfolio_stability=0.70,
        minimum_readiness_score=0.90,
    )

    assert result["strategy_gate"] is True
    assert result["portfolio_gate"] is True
    assert result["readiness_gate"] is False
    assert result["ready"] is False
    assert "combined_readiness" in result["failed_gates"]


def test_is_strategy_ready_returns_boolean():
    assert is_strategy_ready(
        strategy_stability_score=0.90,
        portfolio_stability_score=0.90,
    ) is True

    assert is_strategy_ready(
        strategy_stability_score=0.50,
        portfolio_stability_score=0.90,
    ) is False


def test_invalid_strategy_name_is_rejected():
    with pytest.raises(ValueError):
        evaluate_strategy_readiness(
            strategy="",
            strategy_stability_score=0.90,
            portfolio_stability_score=0.90,
        )


def test_scores_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        calculate_readiness_score(
            strategy_stability_score=1.1,
            portfolio_stability_score=0.90,
        )


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        calculate_readiness_score(
            strategy_stability_score=0.90,
            portfolio_stability_score=0.90,
            strategy_weight=0.80,
            portfolio_weight=0.30,
        )


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        calculate_readiness_score(
            strategy_stability_score=0.90,
            portfolio_stability_score=0.90,
            strategy_weight=-0.10,
            portfolio_weight=1.10,
        )


def test_thresholds_are_validated():
    with pytest.raises(ValueError):
        evaluate_strategy_readiness(
            strategy="momentum",
            strategy_stability_score=0.90,
            portfolio_stability_score=0.90,
            minimum_readiness_score=1.1,
        )


def test_readiness_result_contains_a_complete_decision():
    result = evaluate_strategy_readiness(
        strategy="trend",
        strategy_stability_score=0.82,
        portfolio_stability_score=0.76,
    )

    expected_keys = {
        "strategy",
        "strategy_stability_score",
        "portfolio_stability_score",
        "readiness_score",
        "minimum_strategy_stability",
        "minimum_portfolio_stability",
        "minimum_readiness_score",
        "strategy_gate",
        "portfolio_gate",
        "readiness_gate",
        "ready",
        "failed_gates",
        "status",
    }

    assert expected_keys.issubset(result.keys())
