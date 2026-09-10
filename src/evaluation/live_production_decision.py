from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd

from live_risk_levels import (
    DEFAULT_MOMENTUM_WINDOW,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    build_live_risk_levels,
)
from live_trend import (
    DEFAULT_FAST_WINDOW,
    DEFAULT_SLOW_WINDOW,
    build_live_trend_snapshot,
)

DEFAULT_MIN_STABILITY_SCORE = 0.50
DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_INTERVAL = "5m"


def _validate_score(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number.")

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def _validate_strategy(stable_strategy: str) -> str:
    if not isinstance(stable_strategy, str):
        raise ValueError("stable_strategy must be a string.")

    strategy = stable_strategy.strip()

    if not strategy:
        raise ValueError("stable_strategy must not be empty.")

    return strategy


def build_live_production_decision(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    min_stability_score: float = DEFAULT_MIN_STABILITY_SCORE,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """
    Convert the stable production strategy and current live analysis
    into one human-readable trading decision.

    The decision is intentionally conservative:

    1. A stable strategy must exist.
    2. Its stability score must pass the configured threshold.
    3. The current live implementation must support that strategy.
    4. The live trend must confirm the direction.
    5. The live strategy signal must be active.

    For the current project, the production-selected strategy is
    expected to be "momentum", and the existing live momentum module
    exposes BUY / NO TRADE semantics.

    No SELL decision is invented until the underlying strategy supports
    an explicit SELL signal.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must not be empty.")

    if not isinstance(interval, str) or not interval.strip():
        raise ValueError("interval must not be empty.")

    stable_strategy = _validate_strategy(stable_strategy)
    stability_score = _validate_score(
        "stability_score",
        stability_score,
    )
    min_stability_score = _validate_score(
        "min_stability_score",
        min_stability_score,
    )

    trend_snapshot = build_live_trend_snapshot(
        data,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    risk_snapshot = build_live_risk_levels(
        data,
        momentum_window=momentum_window,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    decision = "NO TRADE"
    reason = "conditions_not_confirmed"

    if stability_score < min_stability_score:
        reason = "stability_score_below_threshold"
    elif stable_strategy != "momentum":
        reason = "stable_strategy_not_supported_by_live_signal"
    elif trend_snapshot["trend"] != "UP":
        reason = "trend_not_confirmed"
    elif int(risk_snapshot["signal"]) != 1:
        reason = "live_signal_not_active"
    else:
        decision = "BUY"
        reason = "stable_strategy_live_signal_and_trend_confirmed"

    return {
        "symbol": symbol,
        "interval": interval,
        "decision": decision,
        "reason": reason,
        "stable_strategy": stable_strategy,
        "stability_score": stability_score,
        "min_stability_score": min_stability_score,
        "strategy_supported": stable_strategy == "momentum",
        "signal": int(risk_snapshot["signal"]),
        "signal_label": str(risk_snapshot["signal_label"]),
        "trend": str(trend_snapshot["trend"]),
        "momentum": risk_snapshot["momentum"],
        "entry_price": risk_snapshot["entry_price"],
        "stop_loss": risk_snapshot["stop_loss"],
        "take_profit": risk_snapshot["take_profit"],
        "risk_reward_ratio": risk_snapshot["risk_reward_ratio"],
        "stop_loss_pct": risk_snapshot["stop_loss_pct"],
        "take_profit_pct": risk_snapshot["take_profit_pct"],
        "momentum_window": risk_snapshot["window"],
        "fast_window": fast_window,
        "slow_window": slow_window,
        "timestamp": risk_snapshot["timestamp"],
    }
