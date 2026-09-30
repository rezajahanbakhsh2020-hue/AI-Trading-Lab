from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd

from src.evaluation.live_production_decision import (
    AuthorizedLiveDecision,
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


def build_live_trade_display(
    authorized_decision: AuthorizedLiveDecision,
    *,
    stability_score: float | None = None,
    min_stability_score: float = 0.50,
    tp1_multiplier: float = DEFAULT_TP1_MULTIPLIER,
    tp2_multiplier: float = DEFAULT_TP2_MULTIPLIER,
    tp3_multiplier: float = DEFAULT_TP3_MULTIPLIER,
) -> dict[str, Any]:
    """Build final live trade display solely as a pure consumer of AuthorizedLiveDecision.

    Does not resolve candidates, fetch market data, or independently evaluate decisions.
    Fails closed if authorized_decision is missing or invalid.
    """
    if not isinstance(authorized_decision, AuthorizedLiveDecision):
        raise TypeError(
            f"authorized_decision must be an AuthorizedLiveDecision instance, got {type(authorized_decision).__name__}"
        )

    tp1_multiplier = _validate_multiplier("tp1_multiplier", tp1_multiplier)
    tp2_multiplier = _validate_multiplier("tp2_multiplier", tp2_multiplier)
    tp3_multiplier = _validate_multiplier("tp3_multiplier", tp3_multiplier)

    if not tp1_multiplier < tp2_multiplier < tp3_multiplier:
        raise ValueError("TP multipliers must satisfy TP1 < TP2 < TP3.")

    dec = authorized_decision.decision
    risk = authorized_decision.risk_levels

    is_buy = dec.direction == Direction.BUY

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
        if risk.entry_price is None or risk.stop_loss is None:
            raise ValueError("BUY decision must contain entry_price and stop_loss.")

        entry_price = float(risk.entry_price)
        stop_loss = float(risk.stop_loss)

        risk_distance = entry_price - stop_loss
        if risk_distance <= 0:
            raise ValueError("BUY risk distance must be greater than zero.")

        tp1 = entry_price + (risk_distance * tp1_multiplier)
        tp2 = entry_price + (risk_distance * tp2_multiplier)
        tp3 = entry_price + (risk_distance * tp3_multiplier)

        take_profit = tp2 if tp2 is not None else tp1
        reward_distance = take_profit - entry_price if take_profit is not None else 0.0
        risk_reward_ratio = risk.risk_reward_ratio or (reward_distance / risk_distance if risk_distance > 0 else None)

        risk_reward_tp1 = ((tp1 - entry_price) / risk_distance) if tp1 is not None else None
        risk_reward_tp2 = ((tp2 - entry_price) / risk_distance) if tp2 is not None else None
        risk_reward_tp3 = ((tp3 - entry_price) / risk_distance) if tp3 is not None else None

    eff_stab_score = stability_score if stability_score is not None else dec.confidence

    return {
        "symbol": authorized_decision.symbol,
        "interval": authorized_decision.timeframe,
        "decision": dec.direction.value,
        "reason": dec.reason,
        "stable_strategy": authorized_decision.strategy_name,
        "stability_score": eff_stab_score,
        "strategy_supported": authorized_decision.strategy_name.lower() == "momentum",
        "signal": 1 if dec.direction == Direction.BUY else 0,
        "signal_label": dec.direction.value,
        "trend": "UP" if dec.direction == Direction.BUY else "NEUTRAL",
        "momentum": entry_price if is_buy else None,
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
        "stop_loss_pct": dec.parameters.get("stop_loss_pct"),
        "take_profit_pct": dec.parameters.get("take_profit_pct"),
        "tp1_multiplier": tp1_multiplier,
        "tp2_multiplier": tp2_multiplier,
        "tp3_multiplier": tp3_multiplier,
        "momentum_window": dec.parameters.get("momentum_window", dec.parameters.get("window")),
        "fast_window": dec.parameters.get("fast_window", 20),
        "slow_window": dec.parameters.get("slow_window", 50),
        "timestamp": dec.market_timestamp,
        "candidate_id": authorized_decision.candidate_id,
        "authorization_fingerprint": authorized_decision.authorization_fingerprint,
        "promoted_artifact_fingerprint": authorized_decision.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": authorized_decision.governance_decision_fingerprint,
    }
