from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.production_selection import (
    select_production_strategy,
)
from src.evaluation.stable_strategy_readiness import (
    evaluate_strategy_readiness,
)


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def build_production_readiness(
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> dict[str, Any]:
    """
    Combine the currently selected production strategy with
    portfolio stability and produce a final readiness decision.

    This function does not modify production selection. It adds
    a final readiness layer on top of the existing selection result.
    """
    selection = select_production_strategy(
        results_dir=results_dir,
    )

    strategy = selection.get("strategy")
    strategy_stability_score = selection.get("stability_score")

    if strategy is None:
        return {
            "strategy": None,
            "strategy_stability_score": strategy_stability_score,
            "portfolio_stability_score": float(
                portfolio_stability_score
            ),
            "readiness_score": 0.0,
            "strategy_gate": False,
            "portfolio_gate": False,
            "readiness_gate": False,
            "ready": False,
            "status": "BLOCKED",
            "failed_gates": ["strategy_selection"],
            "selection": selection,
        }

    if strategy_stability_score is None:
        return {
            "strategy": strategy,
            "strategy_stability_score": None,
            "portfolio_stability_score": float(
                portfolio_stability_score
            ),
            "readiness_score": 0.0,
            "strategy_gate": False,
            "portfolio_gate": False,
            "readiness_gate": False,
            "ready": False,
            "status": "BLOCKED",
            "failed_gates": ["strategy_stability"],
            "selection": selection,
        }

    readiness = evaluate_strategy_readiness(
        strategy=strategy,
        strategy_stability_score=float(
            strategy_stability_score
        ),
        portfolio_stability_score=float(
            portfolio_stability_score
        ),
        minimum_strategy_stability=(
            minimum_strategy_stability
        ),
        minimum_portfolio_stability=(
            minimum_portfolio_stability
        ),
        minimum_readiness_score=(
            minimum_readiness_score
        ),
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    return {
        **readiness,
        "selection": selection,
    }


def is_production_ready(
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> bool:
    """
    Return only the final production readiness decision.
    """
    result = build_production_readiness(
        portfolio_stability_score=portfolio_stability_score,
        results_dir=results_dir,
        minimum_strategy_stability=minimum_strategy_stability,
        minimum_portfolio_stability=minimum_portfolio_stability,
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    return bool(result["ready"])


def production_readiness_message(
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
) -> str:
    """
    Build a human-readable production readiness message.
    """
    result = build_production_readiness(
        portfolio_stability_score=portfolio_stability_score,
        results_dir=results_dir,
        minimum_strategy_stability=minimum_strategy_stability,
        minimum_portfolio_stability=minimum_portfolio_stability,
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    strategy = result.get("strategy")

    if result["ready"]:
        return (
            "PRODUCTION READY: "
            f"{strategy} passed strategy stability, "
            "portfolio stability, and combined readiness gates."
        )

    failed_gates = result.get("failed_gates", [])

    if not failed_gates:
        return "PRODUCTION BLOCKED."

    return (
        "PRODUCTION BLOCKED: "
        f"{strategy or 'no strategy selected'}; "
        f"failed gates: {', '.join(failed_gates)}."
    )
