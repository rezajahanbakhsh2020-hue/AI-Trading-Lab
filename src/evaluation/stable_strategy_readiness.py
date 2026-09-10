from __future__ import annotations

from typing import Any


def _validate_score(name: str, value: float) -> float:
    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def calculate_readiness_score(
    strategy_stability_score: float,
    portfolio_stability_score: float,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> float:
    """
    Calculate a deployment-readiness score from strategy and
    portfolio stability.

    Both input scores must be normalized to [0, 1].

    The weights are configurable and must be non-negative and
    sum to 1.
    """
    strategy_stability_score = _validate_score(
        "strategy_stability_score",
        strategy_stability_score,
    )
    portfolio_stability_score = _validate_score(
        "portfolio_stability_score",
        portfolio_stability_score,
    )

    strategy_weight = float(strategy_weight)
    portfolio_weight = float(portfolio_weight)

    if strategy_weight < 0.0:
        raise ValueError("strategy_weight must be non-negative.")

    if portfolio_weight < 0.0:
        raise ValueError("portfolio_weight must be non-negative.")

    if abs((strategy_weight + portfolio_weight) - 1.0) > 1e-12:
        raise ValueError(
            "strategy_weight and portfolio_weight must sum to 1."
        )

    score = (
        strategy_stability_score * strategy_weight
        + portfolio_stability_score * portfolio_weight
    )

    return float(max(0.0, min(1.0, score)))


def evaluate_strategy_readiness(
    strategy: str,
    strategy_stability_score: float,
    portfolio_stability_score: float,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> dict[str, Any]:
    """
    Evaluate whether a stable strategy is ready for the next
    practical execution stage.

    Readiness requires all three gates to pass:

    1. strategy stability gate
    2. portfolio stability gate
    3. combined readiness score gate
    """
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("strategy must be a non-empty string.")

    minimum_strategy_stability = _validate_score(
        "minimum_strategy_stability",
        minimum_strategy_stability,
    )
    minimum_portfolio_stability = _validate_score(
        "minimum_portfolio_stability",
        minimum_portfolio_stability,
    )
    minimum_readiness_score = _validate_score(
        "minimum_readiness_score",
        minimum_readiness_score,
    )

    strategy_score = _validate_score(
        "strategy_stability_score",
        strategy_stability_score,
    )
    portfolio_score = _validate_score(
        "portfolio_stability_score",
        portfolio_stability_score,
    )

    readiness_score = calculate_readiness_score(
        strategy_stability_score=strategy_score,
        portfolio_stability_score=portfolio_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    strategy_gate = strategy_score >= minimum_strategy_stability
    portfolio_gate = portfolio_score >= minimum_portfolio_stability
    readiness_gate = readiness_score >= minimum_readiness_score

    ready = bool(
        strategy_gate
        and portfolio_gate
        and readiness_gate
    )

    failed_gates: list[str] = []

    if not strategy_gate:
        failed_gates.append("strategy_stability")

    if not portfolio_gate:
        failed_gates.append("portfolio_stability")

    if not readiness_gate:
        failed_gates.append("combined_readiness")

    return {
        "strategy": strategy,
        "strategy_stability_score": strategy_score,
        "portfolio_stability_score": portfolio_score,
        "readiness_score": readiness_score,
        "minimum_strategy_stability": (
            minimum_strategy_stability
        ),
        "minimum_portfolio_stability": (
            minimum_portfolio_stability
        ),
        "minimum_readiness_score": minimum_readiness_score,
        "strategy_gate": strategy_gate,
        "portfolio_gate": portfolio_gate,
        "readiness_gate": readiness_gate,
        "ready": ready,
        "failed_gates": failed_gates,
        "status": "READY" if ready else "BLOCKED",
    }


def is_strategy_ready(
    strategy_stability_score: float,
    portfolio_stability_score: float,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> bool:
    """
    Return only the final readiness decision.
    """
    result = evaluate_strategy_readiness(
        strategy="strategy",
        strategy_stability_score=strategy_stability_score,
        portfolio_stability_score=portfolio_stability_score,
        minimum_strategy_stability=minimum_strategy_stability,
        minimum_portfolio_stability=minimum_portfolio_stability,
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    return bool(result["ready"])
