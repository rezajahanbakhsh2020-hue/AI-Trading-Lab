"""Build live stop-loss and take-profit levels from the existing signal output."""

from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd

from live_signal import DEFAULT_MOMENTUM_WINDOW, build_live_signal_snapshot


DEFAULT_STOP_LOSS_PCT = 0.01
DEFAULT_TAKE_PROFIT_PCT = 0.02
DEFAULT_TP1_MULTIPLIER = 1.0
DEFAULT_TP2_MULTIPLIER = 2.0
DEFAULT_TP3_MULTIPLIER = 3.0


def _validate_percentage(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive number.")

    value = float(value)

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def build_live_risk_levels(
    data: pd.DataFrame,
    *,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    tp1_multiplier: float = DEFAULT_TP1_MULTIPLIER,
    tp2_multiplier: float = DEFAULT_TP2_MULTIPLIER,
    tp3_multiplier: float = DEFAULT_TP3_MULTIPLIER,
) -> dict[str, Any]:
    """
    Build risk levels from the existing live momentum signal.

    The current project strategy exposes BUY (1) and NO TRADE (0).
    Therefore SL/TP/TP1/TP2/TP3 levels are only produced for an active BUY signal.
    When signal is 0 (NO TRADE), stop_loss, take_profit, and risk_reward_ratio return None,
    while entry_price reflects the close price.

    Parameters are proportional decimal distances:
        stop_loss_pct=0.01  -> 1% below entry
        take_profit_pct=0.02 -> 2% above entry
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if "close" not in data.columns:
        raise ValueError("Column 'close' not found in DataFrame.")

    stop_loss_pct = _validate_percentage(
        "stop_loss_pct",
        stop_loss_pct,
    )
    take_profit_pct = _validate_percentage(
        "take_profit_pct",
        take_profit_pct,
    )

    signal_snapshot = build_live_signal_snapshot(
        data,
        window=momentum_window,
    )

    entry_price = float(signal_snapshot["close"])
    signal = int(signal_snapshot["signal"])

    if signal == 1:
        stop_loss = entry_price * (1.0 - stop_loss_pct)
        take_profit = entry_price * (1.0 + take_profit_pct)

        risk_distance = entry_price - stop_loss
        reward_distance = take_profit - entry_price

        risk_reward_ratio = reward_distance / risk_distance
        tp1 = entry_price + (risk_distance * tp1_multiplier)
        tp2 = entry_price + (risk_distance * tp2_multiplier)
        tp3 = entry_price + (risk_distance * tp3_multiplier)
    else:
        stop_loss = None
        take_profit = None
        tp1 = None
        tp2 = None
        tp3 = None
        risk_reward_ratio = None

    return {
        "signal": signal_snapshot["signal"],
        "signal_label": signal_snapshot["signal_label"],
        "strategy": signal_snapshot["strategy"],
        "momentum": signal_snapshot["momentum"],
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "risk_reward_ratio": risk_reward_ratio,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "window": signal_snapshot["window"],
        "timestamp": signal_snapshot["timestamp"],
    }
