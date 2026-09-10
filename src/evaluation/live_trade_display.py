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
    Build final live trade levels from the existing production decision.

    BUY:
        Entry + SL + TP1 + TP2 + TP3 are returned.

    NO TRADE:
        All trade price levels are returned as None.

    This layer does not create SELL logic.
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

    if not tp1_multiplier < tp2_multiplier < tp3_multiplier:
        raise ValueError("TP multipliers must satisfy TP1 < TP2 < TP3.")

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

    is_buy = decision["decision"] == "BUY"

    if not is_buy:
        entry_price = None
        stop_loss = None
        take_profit = None
        tp1 = None
        tp2 = None
        tp3 = None
        risk_distance = None
        risk_reward_tp1 = None
        risk_reward_tp2 = None
        risk_reward_tp3 = None
    else:
        if (
            decision["entry_price"] is None
            or decision["stop_loss"] is None
        ):
            raise ValueError(
                "BUY decision must contain entry_price and stop_loss."
            )

        entry_price = float(decision["entry_price"])
        stop_loss = float(decision["stop_loss"])

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

        take_profit = float(decision["take_profit"])

        risk_reward_tp1 = (
            (tp1 - entry_price) / risk_distance
        )
        risk_reward_tp2 = (
            (tp2 - entry_price) / risk_distance
        )
        risk_reward_tp3 = (
            (tp3 - entry_price) / risk_distance
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
        "take_profit": take_profit,
        "risk_distance": risk_distance,
        "risk_reward_tp1": risk_reward_tp1,
        "risk_reward_tp2": risk_reward_tp2,
        "risk_reward_tp3": risk_reward_tp3,
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
