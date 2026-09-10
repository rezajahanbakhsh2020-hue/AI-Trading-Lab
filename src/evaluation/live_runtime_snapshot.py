from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.evaluation.live_runtime_controller import (
    LiveRuntimeController,
    build_live_runtime_controller,
)


@dataclass(frozen=True)
class LiveRuntimeSnapshot:
    timestamp: str
    symbol: str
    interval: str
    strategy: str | None
    stability_score: float | None
    controller_ready: bool
    session_ready: bool
    decision: str | None
    trend: str | None
    entry: float | None
    stop_loss: float | None
    take_profit_1: float | None
    take_profit_2: float | None
    take_profit_3: float | None
    failed_gates: tuple[str, ...]
    failed_checks: tuple[str, ...]


def build_live_runtime_snapshot(
    data: pd.DataFrame,
    portfolio_stability_score: float,
    *,
    results_dir: str = "results/walk_forward",
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
) -> LiveRuntimeSnapshot:
    """
    Build a serializable snapshot of the live runtime state.

    This layer records the current runtime state only.
    It never connects to a broker and never executes an order.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data must not be empty."
        )

    controller: LiveRuntimeController = (
        build_live_runtime_controller(
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
    )

    production_decision = controller.production_decision

    strategy = production_decision.get("strategy")

    readiness = production_decision.get(
        "readiness",
        {},
    )

    stability_score = readiness.get(
        "strategy_stability_score"
    )

    failed_gates = tuple(
        production_decision.get(
            "runtime_gate",
            {},
        ).get(
            "failed_gates",
            [],
        )
    )

    failed_checks: tuple[str, ...] = ()

    session_ready = False
    decision = None
    trend = None
    entry = None
    stop_loss = None
    take_profit_1 = None
    take_profit_2 = None
    take_profit_3 = None

    if controller.session is not None:
        session = controller.session

        session_ready = bool(
            session.session_ready
        )

        failed_checks = tuple(
            session.preflight.get(
                "failed_checks",
                [],
            )
        )

        runtime = session.runtime

        runtime_decision = runtime.decision
        runtime_display = runtime.display

        decision = runtime_decision.get(
            "decision"
        )

        trend = runtime_decision.get(
            "trend"
        )

        entry = runtime_display.get(
            "entry"
        )

        stop_loss = runtime_display.get(
            "stop_loss"
        )

        take_profit_1 = runtime_display.get(
            "take_profit_1"
        )

        take_profit_2 = runtime_display.get(
            "take_profit_2"
        )

        take_profit_3 = runtime_display.get(
            "take_profit_3"
        )

    return LiveRuntimeSnapshot(
        timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
        symbol=symbol,
        interval=interval,
        strategy=(
            str(strategy)
            if strategy is not None
            else None
        ),
        stability_score=(
            float(stability_score)
            if stability_score is not None
            else None
        ),
        controller_ready=bool(
            controller.controller_ready
        ),
        session_ready=session_ready,
        decision=decision,
        trend=trend,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        take_profit_3=take_profit_3,
        failed_gates=failed_gates,
        failed_checks=failed_checks,
    )


def live_runtime_snapshot_dict(
    snapshot: LiveRuntimeSnapshot,
) -> dict[str, Any]:
    """
    Convert a runtime snapshot into a plain dictionary.
    """

    if not isinstance(
        snapshot,
        LiveRuntimeSnapshot,
    ):
        raise TypeError(
            "snapshot must be a LiveRuntimeSnapshot."
        )

    return asdict(snapshot)


def live_runtime_snapshot_message(
    snapshot: LiveRuntimeSnapshot,
) -> str:
    """
    Return a concise human-readable snapshot status.
    """

    if not isinstance(
        snapshot,
        LiveRuntimeSnapshot,
    ):
        raise TypeError(
            "snapshot must be a LiveRuntimeSnapshot."
        )

    if snapshot.controller_ready:
        return (
            "LIVE RUNTIME SNAPSHOT READY: "
            f"{snapshot.symbol} "
            f"{snapshot.interval}; "
            f"strategy={snapshot.strategy}; "
            f"decision={snapshot.decision}"
        )

    failures = list(
        snapshot.failed_gates
    ) + list(
        snapshot.failed_checks
    )

    if failures:
        return (
            "LIVE RUNTIME SNAPSHOT BLOCKED: "
            f"{snapshot.symbol} "
            f"{snapshot.interval}; "
            f"failures={', '.join(failures)}"
        )

    return (
        "LIVE RUNTIME SNAPSHOT BLOCKED: "
        f"{snapshot.symbol} "
        f"{snapshot.interval}"
    )
