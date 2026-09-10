from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd

from src.evaluation.live_production_decision import (
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    build_live_production_decision,
)


DEFAULT_TP1_MULTIPLIER = 1.0
DEFAULT_TP2_MULTIPLIER = 2.0
DEFAULT_TP3_MULTIPLIER = 3.0


def _validate_multiplier(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number.")

    value = float(value)

    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")

    return value


def _validate_decision(decision: dict[str, Any]) -> None:
    if not isinstance(decision, dict):
        raise ValueError("decision must be a dictionary.")

    required = (
        "decision",
        "entry_price",
        "stop_loss",
        "take_profit",
        "strategy_supported",
        "stability_score",
    )

    missing = [key for key in required if key not in decision]

    if missing:
        raise ValueError(
            f"decision is missing required fields: {', '.join(missing)}"
        )


def build_live_trade_display(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    min_stability_score: float = 0.50,
    momentum_window: int = 10,
    fast_window: int = 20,
    slow_window: int = 50,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
    tp1_multiplier: float = DEFAULT_TP1_MULTIPLIER,
    tp2_multiplier: float = DEFAULT_TP2_MULTIPLIER,
    tp3_multiplier: float = DEFAULT_TP3_MULTIPLIER,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """
    Build the final human-readable live trade levels.

    The function does not create a new trading signal. It consumes the
    existing production decision and converts its risk/reward structure
    into Entry, SL, TP1, TP2 and TP3 levels.

    For BUY decisions:
        TP1 = entry + risk_distance * tp1_multiplier
        TP2 = entry + risk_distance * tp2_multiplier
        TP3 = entry + risk_distance * tp3_multiplier

    For NO TRADE:
        entry, SL and all TP levels are None.

    No SELL logic is invented here.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    tp1_multiplier = _validate_multiplier(
        "tp1_multiplier",
        tp1_multiplier,
    )
    tp2_multiplier = _validate_multiplier(
        "tp2_multiplier",
        tp2_multiplier,
    )
    tp3_multiplier = _validate_multiplier(
        "tp3_multiplier",
        tp3_multiplier,
    )

    if not (
        tp1_multiplier < tp2_multiplier < tp3_multiplier
    ):
        raise ValueError(
            "TP multipliers must satisfy TP1 < TP2 < TP3."
        )

    decision = build_live_production_decision(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        min_stability_score=min_stability_score,
        momentum_window=momentum_window,
        fast_window=fast_window,
        slow_window=slow_window,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        symbol=symbol,
        interval=interval,
    )

    _validate_decision(decision)

    entry_price = decision["entry_price"]
    stop_loss = decision["stop_loss"]

    tp1 = None
    tp2 = None
    tp3 = None

    if decision["decision"] == "BUY":
        if entry_price is None or stop_loss is None:
            raise ValueError(
                "BUY decision must contain entry_price and stop_loss."
            )

        entry_price = float(entry_price)
        stop_loss = float(stop_loss)

        risk_distance = entry_price - stop_loss

        if risk_distance <= 0:
            raise ValueError(
                "BUY risk distance must be greater than zero."
            )

        tp1 = entry_price + (
            risk_distance * tp1_multiplier
        )
        tp2 = entry_price + (
            risk_distance * tp2_multiplier
        )
        tp3 = entry_price + (
            risk_distance * tp3_multiplier
        )

    return {
        "symbol": decision["symbol"],
        "interval": decision["interval"],
        "decision": decision["decision"],
        "reason": decision["reason"],
        "stable_strategy": decision["stable_strategy"],
        "stability_score": decision["stability_score"],
        "strategy_supported": decision["strategy_supported"],
        "signal": decision["signal"],
        "signal_label": decision["signal_label"],
        "trend": decision["trend"],
        "momentum": decision["momentum"],
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "take_profit": decision["take_profit"],
        "risk_distance": (
            None
            if entry_price is None or stop_loss is None
            else float(entry_price) - float(stop_loss)
        ),
        "risk_reward_tp1": (
            None
            if tp1 is None
            else (
                (tp1 - float(entry_price))
                / (float(entry_price) - float(stop_loss))
            )
        ),
        "risk_reward_tp2": (
            None
            if tp2 is None
            else (
                (tp2 - float(entry_price))
                / (float(entry_price) - float(stop_loss))
            )
        ),
        "risk_reward_tp3": (
            None
            if tp3 is None
            else (
                (tp3 - float(entry_price))
                / (float(entry_price) - float(stop_loss))
            )
        ),
        "stop_loss_pct": decision["stop_loss_pct"],
        "take_profit_pct": decision["take_profit_pct"],
        "tp1_multiplier": tp1_multiplier,
        "tp2_multiplier": tp2_multiplier,
        "tp3_multiplier": tp3_multiplier,
        "momentum_window": decision["momentum_window"],
        "fast_window": decision["fast_window"],
        "slow_window": decision["slow_window"],
        "timestamp": decision["timestamp"],
    }
