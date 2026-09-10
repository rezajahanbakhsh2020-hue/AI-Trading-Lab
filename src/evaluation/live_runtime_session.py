from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.live_runtime import (
    LiveRuntimeResult,
    build_live_runtime,
)
from src.evaluation.live_runtime_preflight import (
    build_live_runtime_preflight,
)


@dataclass(frozen=True)
class LiveRuntimeSession:
    preflight: dict[str, Any]
    runtime: LiveRuntimeResult
    session_ready: bool


def build_live_runtime_session(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> LiveRuntimeSession:
    """
    Build a complete live-runtime session.

    Flow:

        live production decision
            -> runtime
            -> preflight
            -> session

    This layer validates the runtime before exposing it as a
    session. It never connects to a broker and never executes
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

    runtime = build_live_runtime(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
    )

    runtime_decision = {
        "runtime_allowed": (
            runtime.decision.get("decision")
            not in {"BLOCKED", "NO_TRADE"}
        ),
        "status": (
            "READY"
            if runtime.decision.get("decision")
            not in {"BLOCKED"}
            else "BLOCKED"
        ),
        "strategy": stable_strategy,
        "readiness": {
            "status": "READY"
            if runtime.decision.get("decision")
            not in {"BLOCKED"}
            else "BLOCKED",
        },
        "runtime_gate": {
            "runtime_ready": (
                runtime.decision.get("decision")
                not in {"BLOCKED"}
            ),
            "status": (
                "READY"
                if runtime.decision.get("decision")
                not in {"BLOCKED"}
                else "BLOCKED"
            ),
        },
    }

    preflight = build_live_runtime_preflight(
        runtime_decision,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    return LiveRuntimeSession(
        preflight=preflight,
        runtime=runtime,
        session_ready=bool(
            preflight["preflight_ready"]
        ),
    )


def is_live_runtime_session_ready(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> bool:
    """
    Return only whether the complete live-runtime session
    is ready.
    """

    session = build_live_runtime_session(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    return bool(session.session_ready)


def live_runtime_session_message(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
    market_data_ready: bool = True,
    quote_ready: bool = True,
    timestamp_ready: bool = True,
) -> str:
    """
    Return a concise session status message.
    """

    session = build_live_runtime_session(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
        market_data_ready=market_data_ready,
        quote_ready=quote_ready,
        timestamp_ready=timestamp_ready,
    )

    if session.session_ready:
        return (
            "LIVE RUNTIME SESSION READY: "
            f"{stable_strategy}"
        )

    failed = session.preflight.get(
        "failed_checks",
        [],
    )

    return (
        "LIVE RUNTIME SESSION BLOCKED: "
        f"{stable_strategy}; "
        f"failed checks: {', '.join(failed)}."
    )
