"""Canonical Live Runtime Orchestrator enforcing authoritative candidate resolution, single runtime authorization, lifecycle state transitions, persistence boundary, and publication boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
    transition_live_decision,
)
from src.evaluation.live_decision_store import (
    DEFAULT_STORE_PATH,
    persist_canonical_live_decision,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionSignal,
    authorize_production_runtime,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_publication_store import (
    publish_canonical_live_decision,
)
from src.evaluation.live_runtime_context import (
    AuthorizedProductionRuntimeContext,
    RuntimeContextValidationError,
    create_authorized_runtime_context,
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
    canonical_decision: CanonicalLiveDecision | None = None


def build_live_runtime(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
    candidate_id: str | None = None,
    research_dir: Any | None = None,
    store_path: Path | str | None = None,
    publisher: Any | None = None,
    publish: bool = False,
    skip_if_no_trade: bool = False,
    persist: bool = True,
    authorized_context: AuthorizedProductionRuntimeContext | None = None,
) -> LiveRuntimeResult:
    """Build and execute the complete canonical live decision lifecycle.

    Enforces the single canonical runtime flow:
    resolve candidate -> authorize exactly once -> evaluate decision -> calculate risk ->
    create CanonicalLiveDecision (AUTHORIZED) -> transition EVALUATED -> RISK_VALIDATED ->
    PRESENTABLE -> persist_canonical_live_decision (PERSISTED) ->
    if publish: publish_canonical_live_decision (PUBLISHED) -> presentation display.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if "timestamp" in data.columns and not data.empty:
        last_ts = pd.to_datetime(data["timestamp"].iloc[-1], utc=True)
        now_ts = pd.Timestamp.now(tz="UTC")
        if (now_ts - last_ts).total_seconds() > 300.0:
            ref_now = last_ts
        else:
            ref_now = now_ts
    else:
        ref_now = None

    if authorized_context is not None:
        if not isinstance(authorized_context, AuthorizedProductionRuntimeContext):
            raise RuntimeContextValidationError(
                f"authorized_context must be an AuthorizedProductionRuntimeContext, got {type(authorized_context).__name__}"
            )

        # Validate scope matching against context
        req_sym = str(symbol).strip().upper()
        req_tf = str(interval).strip()

        if authorized_context.symbol.upper() != req_sym:
            raise RuntimeContextValidationError(
                f"authorized_context symbol '{authorized_context.symbol}' does not match requested symbol '{req_sym}'."
            )
        if authorized_context.timeframe != req_tf:
            raise RuntimeContextValidationError(
                f"authorized_context timeframe '{authorized_context.timeframe}' does not match requested timeframe '{req_tf}'."
            )
        if (
            candidate_id is not None
            and str(candidate_id).strip()
            and authorized_context.candidate_id != str(candidate_id).strip()
        ):
            raise RuntimeContextValidationError(
                f"authorized_context candidate_id '{authorized_context.candidate_id}' does not match requested candidate_id '{candidate_id}'."
            )

        resolved_candidate = authorized_context.candidate
        receipt = authorized_context.authorization_receipt
    else:
        # Legacy/Direct invocation: resolve and authorize exactly once
        r_dir = research_dir if research_dir is not None else DEFAULT_RESEARCH_DIR
        resolved_candidate = resolve_promoted_candidate(
            candidate_id=candidate_id,
            strategy_id=stable_strategy,
            symbol=symbol,
            timeframe=interval,
            base_dir=r_dir,
        )

        if resolved_candidate is None:
            raise ValueError(
                f"No authoritative promoted candidate resolved for strategy '{stable_strategy}' "
                f"(candidate_id={candidate_id!r}, symbol={symbol!r}, timeframe={interval!r}). "
                f"Operational production runtime fails closed."
            )

        authorization = authorize_production_runtime(
            resolved_candidate,
            symbol=symbol,
            timeframe=interval,
            now=ref_now,
        )
        receipt = ProductionAuthorizationReceipt.from_authorization(authorization)
        authorized_context = create_authorized_runtime_context(
            candidate=resolved_candidate,
            authorization=authorization,
            authorization_receipt=receipt,
        )

    # 2. Evaluate strategy decision
    decision_obj = evaluate_production_decision(
        candidate=resolved_candidate,
        data=data,
        reference_now=ref_now,
        max_age_seconds=float("inf"),
    )

    from live_signal import generate_live_signal
    from live_trend import build_live_trend_snapshot

    eff_window = resolved_candidate.parameters.get(
        "momentum_window",
        resolved_candidate.parameters.get("window", 10),
    )
    sig_df = generate_live_signal(data, window=int(eff_window))
    sig_val = int(sig_df["signal"].iloc[-1])

    trend_snap = build_live_trend_snapshot(data)
    trend_val = trend_snap["trend"]

    # Operational gating criteria
    if stability_score < min_stability_score:
        reason = "stability_score_below_threshold"
        final_direction = Direction.NO_TRADE
    elif stable_strategy != "momentum":
        reason = "stable_strategy_not_supported_by_live_signal"
        final_direction = Direction.NO_TRADE
    elif trend_val != "UP":
        reason = "trend_not_confirmed"
        final_direction = Direction.NO_TRADE
    elif sig_val != 1:
        reason = "live_signal_not_active"
        final_direction = Direction.NO_TRADE
    else:
        reason = "stable_strategy_live_signal_and_trend_confirmed"
        final_direction = Direction.BUY

    now_ts = decision_obj.decision_timestamp
    market_ts = decision_obj.market_timestamp
    close_price = float(data["close"].iloc[-1]) if "close" in data.columns else None

    # Calculate quote age in seconds
    quote_age = None
    quote_stale = False
    if "timestamp" in data.columns and not data.empty:
        last_dt = pd.to_datetime(data["timestamp"].iloc[-1], utc=True)
        now_dt = pd.Timestamp.now(tz="UTC") if ref_now is None else pd.to_datetime(ref_now, utc=True)
        quote_age = (now_dt - last_dt).total_seconds()
        if quote_age > 300.0:
            quote_stale = True

    final_decision_obj = ProductionDecision(
        candidate_id=resolved_candidate.candidate_id,
        evidence_id=resolved_candidate.evidence.evidence_id,
        experiment_fingerprint=resolved_candidate.evidence.experiment_fingerprint,
        symbol=resolved_candidate.symbol,
        timeframe=resolved_candidate.timeframe,
        decision_timestamp=now_ts,
        market_timestamp=market_ts,
        direction=final_direction,
        reason=reason,
        entry_price=close_price if final_direction == Direction.BUY else None,
        invalidation_condition="Close below stop_loss or trend turns DOWN" if final_direction == Direction.BUY else None,
        confidence=stability_score,
        parameters=resolved_candidate.parameters,
    )

    # 3. Derive signal & risk
    signal_obj = ProductionSignal.from_decision(final_decision_obj)
    risk_obj = calculate_production_risk_levels(
        decision=final_decision_obj,
        candidate=resolved_candidate,
    )

    # 4. Create Canonical Live Decision Artifact (Initial State: AUTHORIZED)
    ts_now = ref_now.isoformat() if ref_now is not None else datetime.now(timezone.utc).isoformat()
    canonical_dec = create_canonical_live_decision(
        authorization_receipt=receipt,
        decision=final_decision_obj,
        signal=signal_obj,
        risk_levels=risk_obj,
        actor="live_runtime",
        timestamp_utc=ts_now,
        reason="runtime_authorization_and_evaluation",
    )

    # 5. Lifecycle transitions to PRESENTABLE
    canonical_dec = transition_live_decision(
        canonical_dec,
        LiveDecisionLifecycleState.EVALUATED,
        actor="live_runtime",
        timestamp_utc=ts_now,
        reason="strategy_evaluation_complete",
    )
    canonical_dec = transition_live_decision(
        canonical_dec,
        LiveDecisionLifecycleState.RISK_VALIDATED,
        actor="live_runtime",
        timestamp_utc=ts_now,
        reason="risk_geometry_validated",
    )
    canonical_dec = transition_live_decision(
        canonical_dec,
        LiveDecisionLifecycleState.PRESENTABLE,
        actor="live_runtime",
        timestamp_utc=ts_now,
        reason="canonical_artifact_presentable",
    )

    # 6. Enforce persistence boundary if persist is True
    st_path = Path(store_path) if store_path is not None else DEFAULT_STORE_PATH
    if persist:
        final_cld = persist_canonical_live_decision(
            canonical_dec,
            path=st_path,
            actor="live_runtime",
            timestamp_utc=ts_now,
        )
    else:
        final_cld = canonical_dec

    # 7. Enforce publication boundary if publishing requested
    pub_result = None
    if publish and publisher is not None:
        pub_store_path = st_path.parent / "publication_history.json"
        published_dec, _, pub_res = publish_canonical_live_decision(
            final_cld,
            publisher=publisher,
            candidate=resolved_candidate,
            path=pub_store_path,
            skip_if_no_trade=skip_if_no_trade,
            actor="live_runtime",
            timestamp_utc=ts_now,
        )
        final_cld = published_dec
        pub_result = pub_res

    # 8. Presentation display consumer
    display = build_live_trade_display(
        data,
        canonical_decision=final_cld,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        min_stability_score=min_stability_score,
        symbol=symbol,
        interval=interval,
    )
    display["quote_stale"] = quote_stale
    display["quote_age_seconds"] = quote_age

    decision_dict = {
        "symbol": symbol,
        "interval": interval,
        "decision": final_decision_obj.direction.value,
        "reason": final_decision_obj.reason,
        "stable_strategy": stable_strategy,
        "stability_score": stability_score,
        "min_stability_score": min_stability_score,
        "strategy_supported": stable_strategy == "momentum",
        "signal": sig_val if stable_strategy == "momentum" else 0,
        "signal_label": "BUY" if (sig_val == 1 and stable_strategy == "momentum") else "NO TRADE",
        "trend": str(trend_val),
        "momentum": close_price,
        "entry_price": risk_obj.entry_price,
        "stop_loss": risk_obj.stop_loss,
        "take_profit": risk_obj.tp2 if risk_obj.tp2 is not None else risk_obj.tp1,
        "risk_reward_ratio": risk_obj.risk_reward_ratio,
        "stop_loss_pct": resolved_candidate.parameters.get("stop_loss_pct"),
        "take_profit_pct": resolved_candidate.parameters.get("take_profit_pct"),
        "momentum_window": eff_window,
        "fast_window": 20,
        "slow_window": 50,
        "timestamp": final_decision_obj.market_timestamp,
        "quote_stale": quote_stale,
        "quote_age_seconds": quote_age,
        "candidate_id": resolved_candidate.candidate_id,
        "evidence_id": resolved_candidate.evidence.evidence_id,
        "experiment_fingerprint": resolved_candidate.evidence.experiment_fingerprint,
        "decision_id": final_decision_obj.decision_id,
        "canonical_live_decision_fingerprint": final_cld.canonical_live_decision_fingerprint,
        "current_lifecycle_state": final_cld.current_state.value,
        "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
        "authorization_policy_version": receipt.authorization_policy_version,
        "authorized_at_utc": receipt.authorized_at_utc,
        "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
        "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
        "context_fingerprint": authorized_context.context_fingerprint,
    }
    if pub_result:
        decision_dict["publish_result"] = pub_result

    if display["decision"] != decision_dict["decision"]:
        raise ValueError("Live decision and live display decisions do not match.")

    return LiveRuntimeResult(
        decision=decision_dict,
        display=display,
        canonical_decision=final_cld,
    )
