from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.production_runtime import (
    build_production_runtime_decision,
)
from src.evaluation.live_runtime_session import (
    LiveRuntimeSession,
    build_live_runtime_session,
)

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


@dataclass(frozen=True)
class LiveRuntimeController:
    production_decision: dict[str, Any]
    session: LiveRuntimeSession | None
    controller_ready: bool


def build_live_runtime_controller(
    data: pd.DataFrame,
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
    min_stability_score: float = 0.50,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> LiveRuntimeController:
    """
    Connect production runtime permission to the live runtime session.

    Flow:

        production readiness
            ->
        production runtime gate
            ->
        live runtime session
            ->
        controller

    This layer never connects to a broker and never executes
    an order.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data must not be empty."
        )

    production_decision = (
        build_production_runtime_decision(
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
    )

    if not production_decision.get(
        "runtime_allowed",
        False,
    ):
        return LiveRuntimeController(
            production_decision=production_decision,
            session=None,
            controller_ready=False,
        )

    strategy = production_decision.get(
        "strategy"
    )

    readiness = production_decision.get(
        "readiness",
        {},
    )

    stability_score = readiness.get(
        "strategy_stability_score"
    )

    if strategy is None:
        return LiveRuntimeController(
            production_decision=production_decision,
            session=None,
            controller_ready=False,
        )

    if stability_score is None:
        return LiveRuntimeController(
            production_decision=production_decision,
            session=None,
            controller_ready=False,
        )

    session = build_live_runtime_session(
        data,
        stable_strategy=str(strategy),
        stability_score=float(
            stability_score
        ),
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    return LiveRuntimeController(
        production_decision=production_decision,
        session=session,
        controller_ready=bool(
            session.session_ready
        ),
    )


def is_live_runtime_controller_ready(
    data: pd.DataFrame,
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
    min_stability_score: float = 0.50,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> bool:
    """
    Return only the final live-runtime controller permission.
    """

    controller = build_live_runtime_controller(
        data,
        portfolio_stability_score,
        results_dir=results_dir,
        symbol=symbol,
        interval=interval,
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
        min_stability_score=min_stability_score,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    return bool(controller.controller_ready)


def live_runtime_controller_message(
    data: pd.DataFrame,
    portfolio_stability_score: float,
    *,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    minimum_strategy_stability: float = 0.70,
    minimum_portfolio_stability: float = 0.70,
    minimum_readiness_score: float = 0.70,
    strategy_weight: float = 0.70,
    portfolio_weight: float = 0.30,
    min_stability_score: float = 0.50,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> str:
    """
    Return a concise human-readable controller status.
    """

    controller = build_live_runtime_controller(
        data,
        portfolio_stability_score,
        results_dir=results_dir,
        symbol=symbol,
        interval=interval,
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
        min_stability_score=min_stability_score,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    if controller.controller_ready:
        strategy = (
            controller.production_decision.get(
                "strategy"
            )
            or "unknown"
        )

        return (
            "LIVE RUNTIME CONTROLLER READY: "
            f"{strategy}"
        )

    failed_gates = (
        controller.production_decision
        .get("runtime_gate", {})
        .get("failed_gates", [])
    )

    if failed_gates:
        return (
            "LIVE RUNTIME CONTROLLER BLOCKED: "
            f"failed gates: "
            f"{', '.join(failed_gates)}."
        )

    if controller.session is not None:
        failed_checks = (
            controller.session.preflight.get(
                "failed_checks",
                [],
            )
        )

        return (
            "LIVE RUNTIME CONTROLLER BLOCKED: "
            f"failed checks: "
            f"{', '.join(failed_checks)}."
        )

    return (
        "LIVE RUNTIME CONTROLLER BLOCKED."
    )
