from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.production_readiness import (
    build_production_readiness,
)
from src.evaluation.production_runtime_gate import (
    build_production_runtime_gate,
)


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def build_production_runtime_decision(
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
    Build the single decision artifact used by the production runtime.

    Flow:
        production readiness
            -> runtime gate
            -> final runtime decision

    This function does not execute orders or start a live broker.
    """

    readiness = build_production_readiness(
        portfolio_stability_score,
        results_dir=results_dir,
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

    runtime_gate = build_production_runtime_gate(
        readiness,
        minimum_readiness_score=(
            minimum_readiness_score
        ),
    )

    return {
        "runtime_allowed": bool(
            runtime_gate["runtime_ready"]
        ),
        "status": runtime_gate["status"],
        "strategy": runtime_gate.get("strategy"),
        "readiness": readiness,
        "runtime_gate": runtime_gate,
    }


def is_production_runtime_allowed(
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
    Return only the final runtime permission.
    """

    decision = build_production_runtime_decision(
        portfolio_stability_score,
        results_dir=results_dir,
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

    return bool(decision["runtime_allowed"])


def production_runtime_decision_message(
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
    Return a concise human-readable runtime decision.
    """

    decision = build_production_runtime_decision(
        portfolio_stability_score,
        results_dir=results_dir,
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

    strategy = (
        decision.get("strategy")
        or "no strategy selected"
    )

    if decision["runtime_allowed"]:
        return (
            "PRODUCTION RUNTIME ALLOWED: "
            f"{strategy}"
        )

    failed_gates = decision[
        "runtime_gate"
    ].get("failed_gates", [])

    return (
        "PRODUCTION RUNTIME BLOCKED: "
        f"{strategy}; "
        f"failed gates: "
        f"{', '.join(failed_gates)}."
    )
