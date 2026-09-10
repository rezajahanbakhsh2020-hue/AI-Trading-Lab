from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.production_readiness import (
    DEFAULT_RESULTS_DIR,
    build_production_readiness,
)
from src.evaluation.production_runtime_gate import (
    build_production_runtime_gate,
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
    Build the complete decision chain:

    strategy selection
        -> production readiness
        -> runtime gate

    The returned structure is the single machine-readable
    decision for entering the live production runtime.
    """
    readiness = build_production_readiness(
        portfolio_stability_score=portfolio_stability_score,
        results_dir=results_dir,
        minimum_strategy_stability=minimum_strategy_stability,
        minimum_portfolio_stability=minimum_portfolio_stability,
        minimum_readiness_score=minimum_readiness_score,
        strategy_weight=strategy_weight,
        portfolio_weight=portfolio_weight,
    )

    runtime_gate = build_production_runtime_gate(
        readiness,
        minimum_readiness_score=minimum_readiness_score,
    )

    return {
        "runtime_ready": runtime_gate["runtime_ready"],
        "status": runtime_gate["status"],
        "strategy": runtime_gate["strategy"],
        "strategy_stability_score": runtime_gate[
            "strategy_stability_score"
        ],
        "portfolio_stability_score": runtime_gate[
            "portfolio_stability
