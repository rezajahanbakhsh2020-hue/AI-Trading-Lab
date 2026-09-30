from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.live_production_decision import (
    AuthorizedLiveDecision,
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionRiskLevels,
    authorize_production_runtime,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_trade_display import (
    build_live_trade_display,
)
from src.evaluation.research_store import (
    DEFAULT_RESEARCH_DIR,
    resolve_promoted_candidate,
)


@dataclass(frozen=True)
class LiveRuntimeResult:
    decision: dict[str, Any]
    display: dict[str, Any]
    authorized_decision: AuthorizedLiveDecision | None = None


def build_live_runtime(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
    candidate_id: str | None = None,
    research_dir: Any = DEFAULT_RESEARCH_DIR,
) -> LiveRuntimeResult:
    """Build the complete live decision and display payload via canonical authorization chain.

    Executes: resolve promoted candidate -> authorize runtime -> evaluate production decision ->
    calculate risk levels -> AuthorizedLiveDecision -> pure display.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    candidate = resolve_promoted_candidate(
        candidate_id=candidate_id,
        strategy_id=stable_strategy,
        symbol=symbol,
        timeframe=interval,
        base_dir=research_dir,
    )

    if candidate is None:
        raise ValueError(
            f"No authoritative promoted candidate resolved for strategy '{stable_strategy}' "
            f"(candidate_id={candidate_id!r}, symbol={symbol!r}, timeframe={interval!r})."
        )

    if "timestamp" in data.columns and not data.empty:
        last_ts = pd.to_datetime(data["timestamp"].iloc[-1], utc=True)
        now_ts = pd.Timestamp.now(tz="UTC")
        if (now_ts - last_ts).total_seconds() > 300.0:
            ref_now = last_ts
        else:
            ref_now = now_ts
    else:
        ref_now = None

    authorization = authorize_production_runtime(
        candidate,
        symbol=symbol,
        timeframe=interval,
        now=ref_now,
    )

    receipt = ProductionAuthorizationReceipt.from_authorization(authorization)

    decision_obj = evaluate_production_decision(
        candidate=candidate,
        data=data,
        reference_now=ref_now,
        max_age_seconds=float("inf"),
        authorization=receipt,
    )

    # Apply operational stability gate
    if stability_score < min_stability_score or stable_strategy.lower() != "momentum":
        final_direction = Direction.NO_TRADE
        reason = "stability_score_below_threshold" if stability_score < min_stability_score else "unsupported_live_strategy"
        entry_p = None
        inval = None
    else:
        final_direction = decision_obj.direction
        reason = decision_obj.reason
        entry_p = decision_obj.entry_price
        inval = decision_obj.invalidation_condition

    final_decision_obj = ProductionDecision(
        candidate_id=candidate.candidate_id,
        evidence_id=candidate.evidence.evidence_id,
        experiment_fingerprint=candidate.evidence.experiment_fingerprint,
        symbol=candidate.symbol,
        timeframe=candidate.timeframe,
        decision_timestamp=decision_obj.decision_timestamp,
        market_timestamp=decision_obj.market_timestamp,
        direction=final_direction,
        reason=reason,
        entry_price=entry_p,
        invalidation_condition=inval,
        confidence=stability_score,
        parameters=candidate.parameters,
    )

    risk_obj = calculate_production_risk_levels(
        decision=final_decision_obj,
        candidate=candidate,
    )

    auth_decision = AuthorizedLiveDecision.create(
        authorization=receipt,
        candidate=candidate,
        decision=final_decision_obj,
        risk_levels=risk_obj,
    )

    display = build_live_trade_display(
        authorized_decision=auth_decision,
        stability_score=stability_score,
        min_stability_score=min_stability_score,
    )

    legacy_decision_dict = {
        "symbol": auth_decision.symbol,
        "interval": auth_decision.timeframe,
        "decision": auth_decision.decision.direction.value,
        "reason": auth_decision.decision.reason,
        "stable_strategy": stable_strategy,
        "stability_score": stability_score,
        "min_stability_score": min_stability_score,
        "strategy_supported": stable_strategy.lower() == "momentum",
        "signal": 1 if auth_decision.decision.direction == Direction.BUY else 0,
        "signal_label": auth_decision.decision.direction.value,
        "trend": "UP" if auth_decision.decision.direction == Direction.BUY else "NEUTRAL",
        "momentum": float(data["close"].iloc[-1]) if "close" in data.columns else None,
        "entry_price": auth_decision.risk_levels.entry_price,
        "stop_loss": auth_decision.risk_levels.stop_loss,
        "take_profit": auth_decision.risk_levels.tp2 if auth_decision.risk_levels.tp2 is not None else auth_decision.risk_levels.tp1,
        "risk_reward_ratio": auth_decision.risk_levels.risk_reward_ratio,
        "timestamp": auth_decision.decision.market_timestamp,
        "candidate_id": candidate.candidate_id,
        "evidence_id": candidate.evidence.evidence_id,
        "experiment_fingerprint": candidate.evidence.experiment_fingerprint,
        "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
        "authorization_policy_version": receipt.authorization_policy_version,
        "authorized_at_utc": receipt.authorized_at_utc,
        "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
        "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
    }

    return LiveRuntimeResult(
        decision=legacy_decision_dict,
        display=display,
        authorized_decision=auth_decision,
    )
