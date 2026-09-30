"""Pure presentation consumer building human-readable live trade displays from CanonicalLiveDecision artifacts."""

from __future__ import annotations

from numbers import Real
from typing import Any, Optional

import pandas as pd

from src.evaluation.live_decision_lifecycle import CanonicalLiveDecision
from src.evaluation.live_production_decision import (
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    Direction,
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
    canonical_decision: Optional[CanonicalLiveDecision] = None,
    stable_strategy: str = "momentum",
    stability_score: float = 1.0,
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
    Build final live trade display presentation purely from a CanonicalLiveDecision artifact.

    The display layer does NOT independently construct production decisions, resolve candidates, or authorize execution.
    If canonical_decision is absent, returns a safe BLOCKED presentation payload.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    tp1_multiplier = _validate_multiplier("tp1_multiplier", tp1_multiplier)
    tp2_multiplier = _validate_multiplier("tp2_multiplier", tp2_multiplier)
    tp3_multiplier = _validate_multiplier("tp3_multiplier", tp3_multiplier)

    if not tp1_multiplier < tp2_multiplier < tp3_multiplier:
        raise ValueError("TP multipliers must satisfy TP1 < TP2 < TP3.")

    # Pure presentation consumer requirement: Display layer MUST require CanonicalLiveDecision.
    # NEVER invoke build_live_runtime(), build_live_production_decision(), authorize_production_runtime(), or resolve_promoted_candidate().
    if canonical_decision is None or not isinstance(canonical_decision, CanonicalLiveDecision):
        close_price = float(data["close"].iloc[-1]) if ("close" in data.columns and not data.empty) else None
        ts = (
            pd.to_datetime(data["timestamp"].iloc[-1], utc=True).isoformat()
            if ("timestamp" in data.columns and not data.empty)
            else None
        )
        return {
            "symbol": str(symbol).upper() if symbol else DEFAULT_SYMBOL,
            "interval": str(interval) if interval else DEFAULT_INTERVAL,
            "decision": "BLOCKED",
            "reason": "missing_canonical_live_decision",
            "stable_strategy": str(stable_strategy),
            "stability_score": float(stability_score) if stability_score is not None else 0.0,
            "strategy_supported": str(stable_strategy).lower() == "momentum",
            "signal": 0,
            "signal_label": "BLOCKED",
            "trend": "NEUTRAL",
            "momentum": close_price,
            "entry_price": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
            "take_profit": None,
            "risk_distance": None,
            "risk_reward_ratio": None,
            "risk_reward_tp1": None,
            "risk_reward_tp2": None,
            "risk_reward_tp3": None,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "tp1_multiplier": tp1_multiplier,
            "tp2_multiplier": tp2_multiplier,
            "tp3_multiplier": tp3_multiplier,
            "momentum_window": momentum_window,
            "fast_window": fast_window,
            "slow_window": slow_window,
            "timestamp": ts,
        }

    dec_obj = canonical_decision.decision
    risk_obj = canonical_decision.risk_levels
    receipt = canonical_decision.authorization_receipt

    from live_trend import build_live_trend_snapshot
    trend_snap = build_live_trend_snapshot(data, fast_window=fast_window, slow_window=slow_window)

    close_price = float(data["close"].iloc[-1]) if ("close" in data.columns and not data.empty) else None

    decision = {
        "symbol": receipt.symbol,
        "interval": receipt.timeframe,
        "decision": dec_obj.direction.value,
        "reason": dec_obj.reason,
        "stable_strategy": receipt.strategy_name,
        "stability_score": dec_obj.confidence if dec_obj.confidence is not None else stability_score,
        "min_stability_score": min_stability_score,
        "strategy_supported": receipt.strategy_name.lower() == "momentum",
        "signal": 1 if dec_obj.direction == Direction.BUY else 0,
        "signal_label": dec_obj.direction.value,
        "trend": str(trend_snap.get("trend", "NEUTRAL")),
        "momentum": close_price,
        "entry_price": risk_obj.entry_price,
        "stop_loss": risk_obj.stop_loss,
        "take_profit": risk_obj.tp2 if risk_obj.tp2 is not None else risk_obj.tp1,
        "risk_reward_ratio": risk_obj.risk_reward_ratio,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
        "momentum_window": momentum_window,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "timestamp": dec_obj.market_timestamp,
        "decision_id": dec_obj.decision_id,
        "canonical_live_decision_fingerprint": canonical_decision.canonical_live_decision_fingerprint,
        "authorization_fingerprint": receipt.authorization_fingerprint,
    }

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
        risk_reward_ratio = None
        risk_reward_tp1 = None
        risk_reward_tp2 = None
        risk_reward_tp3 = None
    else:
        if decision["entry_price"] is None or decision["stop_loss"] is None:
            raise ValueError("BUY decision must contain entry_price and stop_loss.")

        entry_price = float(decision["entry_price"])
        stop_loss = float(decision["stop_loss"])
        risk_distance = entry_price - stop_loss

        if risk_distance <= 0:
            raise ValueError("BUY risk distance must be greater than zero.")

        tp1 = entry_price + (risk_distance * tp1_multiplier)
        tp2 = entry_price + (risk_distance * tp2_multiplier)
        tp3 = entry_price + (risk_distance * tp3_multiplier)

        take_profit = float(decision["take_profit"]) if decision.get("take_profit") is not None else tp2
        reward_distance = take_profit - entry_price
        risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else None

        risk_reward_tp1 = (tp1 - entry_price) / risk_distance
        risk_reward_tp2 = (tp2 - entry_price) / risk_distance
        risk_reward_tp3 = (tp3 - entry_price) / risk_distance

    res = {
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
        "risk_reward_ratio": risk_reward_ratio,
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

    for opt_key in ("decision_id", "canonical_live_decision_fingerprint", "authorization_fingerprint"):
        if opt_key in decision:
            res[opt_key] = decision[opt_key]

    return res
