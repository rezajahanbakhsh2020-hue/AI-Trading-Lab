from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.evaluation.live_runtime_bridge import (
    run_live_runtime_bridge,
    validate_live_runtime_bridge,
)


REQUIRED_LIVE_RESULT_FIELDS = (
    "signal",
    "signal_label",
    "strategy",
    "trend",
    "entry_price",
    "stop_loss",
    "take_profit",
    "risk_reward_ratio",
    "timestamp",
)


def build_live_runtime_proof(
    live_result: Mapping[str, Any],
    *,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    market_state: str = "open",
    quote_age_seconds: int | float = 0,
    quote_stale: bool = False,
    candle_count: int | None = None,
    history: list[Mapping[str, Any]] | None = None,
    store_path: str | None = None,
) -> dict[str, Any]:
    """
    Build a practical proof package from the real live
    signal/trend/risk application output.

    The existing live result is passed through unchanged.
    No signal, trend, risk, or stability value is recalculated.
    """
    if not isinstance(live_result, Mapping):
        raise TypeError("live_result must be a mapping")

    missing = [
        field
        for field in REQUIRED_LIVE_RESULT_FIELDS
        if field not in live_result
    ]

    if missing:
        raise ValueError(
            "live_result is missing required fields: "
            + ", ".join(missing)
        )

    runtime_result = dict(live_result)

    runtime_result["symbol"] = symbol
    runtime_result["interval"] = interval
    runtime_result["market_state"] = market_state
    runtime_result["quote_age_seconds"] = quote_age_seconds
    runtime_result["quote_stale"] = quote_stale

    if candle_count is not None:
        runtime_result["candle_count"] = candle_count

    bridge = run_live_runtime_bridge(
        runtime_result,
        history=history,
        store_path=store_path,
    )

    validate_live_runtime_bridge(bridge)

    proof = bridge["bridge"]["end_to_end"]["proof"]

    return {
        "symbol": symbol,
        "interval": interval,
        "signal": runtime_result["signal"],
        "signal_label": runtime_result["signal_label"],
        "trend": runtime_result["trend"],
        "strategy": runtime_result["strategy"],
        "entry_price": runtime_result["entry_price"],
        "stop_loss": runtime_result["stop_loss"],
        "take_profit": runtime_result["take_profit"],
        "risk_reward_ratio": runtime_result["risk_reward_ratio"],
        "stability_score": runtime_result.get(
            "stability_score"
        ),
        "market_state": market_state,
        "quote_age_seconds": quote_age_seconds,
        "quote_stale": quote_stale,
        "candle_count": candle_count,
        "timestamp": runtime_result["timestamp"],
        "proof": proof,
        "bridge": bridge,
        "passed": bridge["passed"],
    }


def validate_live_runtime_proof(
    result: Mapping[str, Any],
) -> bool:
    """
    Validate the structure of a live runtime proof package.
    """
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    required_fields = (
        "symbol",
        "interval",
        "signal",
        "signal_label",
        "trend",
        "strategy",
        "entry_price",
        "stop_loss",
        "take_profit",
        "risk_reward_ratio",
        "stability_score",
        "market_state",
        "quote_age_seconds",
        "quote_stale",
        "candle_count",
        "timestamp",
        "proof",
        "bridge",
        "passed",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise ValueError(
            "live runtime proof is missing required fields: "
            + ", ".join(missing)
        )

    if not isinstance(result["proof"], Mapping):
        raise ValueError("proof must be a mapping")

    if not isinstance(result["bridge"], Mapping):
        raise ValueError("bridge must be a mapping")

    if not isinstance(result["passed"], bool):
        raise ValueError("passed must be boolean")

    if not isinstance(result["quote_stale"], bool):
        raise ValueError("quote_stale must be boolean")

    return True
