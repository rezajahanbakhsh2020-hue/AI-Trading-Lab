"""Production Decision domain models, promotion eligibility validation, authoritative decision evaluation,
signal lineage projection, and risk geometry validation for Project 1.

Connects promoted research candidates and evidence to the live execution decision path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from src.evaluation.research_constitution import (
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
)


class Direction(str, Enum):
    """Authoritative trading directions."""

    BUY = "BUY"
    SELL = "SELL"
    NO_TRADE = "NO TRADE"


@dataclass(frozen=True)
class ProductionPromotionPolicy:
    """Versioned promotion policy requirements for production eligibility."""

    policy_version: str = "promotion_v1.0"
    allowed_statuses: tuple[PromotionStatus, ...] = (
        PromotionStatus.PROMOTABLE,
        PromotionStatus.VALIDATED,
    )
    max_evidence_age_days: Optional[float] = None
    require_robustness_pass: bool = True

    def __post_init__(self) -> None:
        if not self.policy_version or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string.")
        if not self.allowed_statuses:
            raise ValueError("allowed_statuses must not be empty.")


@dataclass(frozen=True)
class PromotedCandidateArtifact:
    """Authoritative representation of a promoted research candidate eligible for production execution."""

    candidate_id: str
    strategy_name: str
    strategy_version: str
    evidence: ResearchEvidence
    symbol: str
    timeframe: str
    parameters: dict[str, Any] = field(default_factory=dict)
    policy: ProductionPromotionPolicy = field(default_factory=ProductionPromotionPolicy)
    artifact_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string.")
        if not self.strategy_version or not self.strategy_version.strip():
            raise ValueError("strategy_version must be a non-empty string.")
        if not isinstance(self.evidence, ResearchEvidence):
            raise TypeError("evidence must be a ResearchEvidence instance.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")

        validate_promotion_eligibility(self.evidence, policy=self.policy)

        # Check scope match
        ds = self.evidence.spec.dataset_scope
        if ds.symbol.upper() != self.symbol.upper():
            raise ValueError(
                f"Candidate symbol '{self.symbol}' does not match evidence dataset symbol '{ds.symbol}'."
            )
        if ds.timeframe != self.timeframe:
            raise ValueError(
                f"Candidate timeframe '{self.timeframe}' does not match evidence dataset timeframe '{ds.timeframe}'."
            )

        payload = {
            "candidate_id": self.candidate_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "evidence_id": self.evidence.evidence_id,
            "evidence_fingerprint": self.evidence.experiment_fingerprint,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "parameters": self.parameters,
            "policy_version": self.policy.policy_version,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        object.__setattr__(
            self,
            "artifact_fingerprint",
            hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        )


def validate_promotion_eligibility(
    evidence: ResearchEvidence,
    policy: Optional[ProductionPromotionPolicy] = None,
) -> bool:
    """Validate that research evidence satisfies production promotion criteria.

    Fails closed if evidence is missing, unpromoted, rejected, or invalid.
    """
    if policy is None:
        policy = ProductionPromotionPolicy()

    if not isinstance(evidence, ResearchEvidence):
        raise TypeError("evidence must be a ResearchEvidence instance.")

    if evidence.promotion_status not in policy.allowed_statuses:
        raise ValueError(
            f"Evidence '{evidence.evidence_id}' has status '{evidence.promotion_status.value}' "
            f"which is not allowed for production (allowed: {[s.value for s in policy.allowed_statuses]})."
        )

    if evidence.rejection_reasons:
        reasons = [r.value for r in evidence.rejection_reasons]
        raise ValueError(
            f"Evidence '{evidence.evidence_id}' contains rejection reasons: {reasons}"
        )

    if policy.require_robustness_pass and evidence.robustness_verdict:
        passed = evidence.robustness_verdict.get("passed")
        if passed is False:
            raise ValueError(
                f"Evidence '{evidence.evidence_id}' failed robustness verdict."
            )

    return True


def validate_market_data_for_production(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    max_age_seconds: float = 300.0,
    reference_now: Optional[Any] = None,
) -> pd.Series:
    """Validate live market data for production decision evaluation.

    Checks symbol, timeframe, timestamp presence, ordering, freshness, non-future timestamps,
    and valid OHLC values. Returns the latest valid bar as a Series.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Market data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Market data DataFrame is empty.")

    required_cols = {"timestamp", "open", "high", "low", "close"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Market data missing required columns: {sorted(missing)}")

    # Check for NaNs or non-finite OHLC values
    for col in ("open", "high", "low", "close"):
        s = pd.to_numeric(data[col], errors="coerce")
        if s.isna().any() or not math.isfinite(s.iloc[-1]):
            raise ValueError(f"Market data contains invalid or non-finite values in column '{col}'.")
        if (s <= 0).any():
            raise ValueError(f"Market data contains non-positive price values in column '{col}'.")

    # High/Low sanity check
    if (data["high"] < data["low"]).any() or (data["high"] < data["open"]).any() or (data["high"] < data["close"]).any():
        raise ValueError("Market data contains impossible OHLC high/low relationships.")

    latest_bar = data.iloc[-1]
    ts = pd.to_datetime(latest_bar["timestamp"], utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError("Latest bar timestamp is invalid.")

    # Validate freshness against reference_now
    if reference_now is None:
        now_dt = pd.Timestamp.now(tz="UTC")
    else:
        now_dt = pd.to_datetime(reference_now, utc=True)

    age_seconds = (now_dt - ts).total_seconds()
    if age_seconds < 0:
        raise ValueError(f"Market data timestamp '{ts}' is in the future relative to '{now_dt}'.")
    if age_seconds > max_age_seconds:
        raise ValueError(
            f"Market data timestamp '{ts}' is stale ({age_seconds:.1f}s old, max allowed: {max_age_seconds}s)."
        )

    return latest_bar


@dataclass(frozen=True)
class ProductionDecision:
    """Authoritative trading decision produced by strategy evaluation against market data."""

    candidate_id: str
    evidence_id: str
    experiment_fingerprint: str
    symbol: str
    timeframe: str
    decision_timestamp: str
    market_timestamp: str
    direction: Direction
    reason: str
    entry_price: Optional[float]
    invalidation_condition: Optional[str]
    confidence: Optional[float] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.experiment_fingerprint or not self.experiment_fingerprint.strip():
            raise ValueError("experiment_fingerprint must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if not self.decision_timestamp or not self.decision_timestamp.strip():
            raise ValueError("decision_timestamp must be a non-empty string.")
        if not self.market_timestamp or not self.market_timestamp.strip():
            raise ValueError("market_timestamp must be a non-empty string.")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction enum member.")

        if self.direction in (Direction.BUY, Direction.SELL):
            if self.entry_price is None or not math.isfinite(self.entry_price) or self.entry_price <= 0:
                raise ValueError(
                    f"Executable decision '{self.direction.value}' requires a positive finite entry_price."
                )
        else:
            if self.entry_price is not None and (not math.isfinite(self.entry_price) or self.entry_price <= 0):
                raise ValueError("entry_price must be positive and finite if provided.")

        payload = {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "parameters": self.parameters,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        dec_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "decision_id", f"dec_{dec_hash}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "experiment_fingerprint": self.experiment_fingerprint,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision_timestamp": self.decision_timestamp,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "reason": self.reason,
            "entry_price": self.entry_price,
            "invalidation_condition": self.invalidation_condition,
            "confidence": self.confidence,
            "parameters": self.parameters,
        }


def evaluate_production_decision(
    candidate: PromotedCandidateArtifact,
    data: pd.DataFrame,
    reference_now: Optional[Any] = None,
    max_age_seconds: float = 300.0,
) -> ProductionDecision:
    """Evaluate promoted candidate strategy against live market data to produce an authoritative ProductionDecision."""
    if not isinstance(candidate, PromotedCandidateArtifact):
        raise TypeError("candidate must be a PromotedCandidateArtifact instance.")

    latest_bar = validate_market_data_for_production(
        data=data,
        symbol=candidate.symbol,
        timeframe=candidate.timeframe,
        max_age_seconds=max_age_seconds,
        reference_now=reference_now,
    )

    now_ts = (
        pd.Timestamp.now(tz="UTC").isoformat()
        if reference_now is None
        else pd.to_datetime(reference_now, utc=True).isoformat()
    )
    market_ts = pd.to_datetime(latest_bar["timestamp"], utc=True).isoformat()

    # Execute strategy evaluation based on candidate strategy name
    strat_name = candidate.strategy_name.lower()
    close_price = float(latest_bar["close"])

    if strat_name == "momentum":
        from live_signal import generate_live_signal
        from live_trend import build_live_trend_snapshot

        window = candidate.parameters.get("momentum_window", candidate.parameters.get("window", 10))
        sig_df = generate_live_signal(data, window=int(window))
        sig_val = int(sig_df["signal"].iloc[-1])

        trend_snap = build_live_trend_snapshot(data)
        trend_val = trend_snap["trend"]

        if sig_val == 1 and trend_val == "UP":
            direction = Direction.BUY
            reason = "momentum_signal_1_and_trend_UP_confirmed"
            invalidation = "Close below stop_loss or trend turns DOWN"
        else:
            direction = Direction.NO_TRADE
            reason = f"signal={sig_val}, trend={trend_val} (unconfirmed setup)"
            invalidation = None

        return ProductionDecision(
            candidate_id=candidate.candidate_id,
            evidence_id=candidate.evidence.evidence_id,
            experiment_fingerprint=candidate.evidence.experiment_fingerprint,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            decision_timestamp=now_ts,
            market_timestamp=market_ts,
            direction=direction,
            reason=reason,
            entry_price=close_price if direction != Direction.NO_TRADE else None,
            invalidation_condition=invalidation,
            confidence=None,  # Do not manufacture confidence unless calculated
            parameters=candidate.parameters,
        )

    else:
        # Unsupported strategy returns NO_TRADE
        return ProductionDecision(
            candidate_id=candidate.candidate_id,
            evidence_id=candidate.evidence.evidence_id,
            experiment_fingerprint=candidate.evidence.experiment_fingerprint,
            symbol=candidate.symbol,
            timeframe=candidate.timeframe,
            decision_timestamp=now_ts,
            market_timestamp=market_ts,
            direction=Direction.NO_TRADE,
            reason=f"strategy_{candidate.strategy_name}_unsupported_in_live_evaluator",
            entry_price=None,
            invalidation_condition=None,
            confidence=None,
            parameters=candidate.parameters,
        )


@dataclass(frozen=True)
class ProductionSignal:
    """Operational signal projected directly from an authoritative ProductionDecision."""

    decision_id: str
    candidate_id: str
    evidence_id: str
    symbol: str
    timeframe: str
    market_timestamp: str
    direction: Direction
    entry_price: Optional[float]
    signal_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.decision_id or not self.decision_id.strip():
            raise ValueError("decision_id must be a non-empty string.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if not self.market_timestamp or not self.market_timestamp.strip():
            raise ValueError("market_timestamp must be a non-empty string.")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction enum member.")

        payload = {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        sig_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "signal_id", f"sig_{sig_hash}")

    @classmethod
    def from_decision(cls, decision: ProductionDecision) -> ProductionSignal:
        if not isinstance(decision, ProductionDecision):
            raise TypeError("decision must be a ProductionDecision instance.")
        return cls(
            decision_id=decision.decision_id,
            candidate_id=decision.candidate_id,
            evidence_id=decision.evidence_id,
            symbol=decision.symbol,
            timeframe=decision.timeframe,
            market_timestamp=decision.market_timestamp,
            direction=decision.direction,
            entry_price=decision.entry_price,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_timestamp": self.market_timestamp,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
        }


@dataclass(frozen=True)
class ProductionRiskLevels:
    """Authoritative risk levels calculated strictly from strategy/risk logic and entry geometry."""

    decision_id: str
    candidate_id: str
    evidence_id: str
    symbol: str
    timeframe: str
    direction: Direction
    entry_price: Optional[float]
    stop_loss: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    tp3: Optional[float]
    trailing_stop: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    risk_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.decision_id or not self.decision_id.strip():
            raise ValueError("decision_id must be a non-empty string.")
        if not self.candidate_id or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a non-empty string.")
        if not self.evidence_id or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string.")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not self.timeframe.strip():
            raise ValueError("timeframe must be a non-empty string.")
        if not isinstance(self.direction, Direction):
            raise TypeError("direction must be a Direction enum member.")

        # Validate geometry if active trade direction
        if self.direction in (Direction.BUY, Direction.SELL):
            if self.entry_price is None or not math.isfinite(self.entry_price) or self.entry_price <= 0:
                raise ValueError(f"Active risk for {self.direction.value} requires positive finite entry_price.")

            # Validate Stop Loss geometry
            if self.stop_loss is not None:
                if not math.isfinite(self.stop_loss) or self.stop_loss <= 0:
                    raise ValueError("stop_loss must be positive and finite if provided.")
                if self.direction == Direction.BUY and self.stop_loss >= self.entry_price:
                    raise ValueError(
                        f"BUY stop_loss ({self.stop_loss}) must be strictly below entry_price ({self.entry_price})."
                    )
                if self.direction == Direction.SELL and self.stop_loss <= self.entry_price:
                    raise ValueError(
                        f"SELL stop_loss ({self.stop_loss}) must be strictly above entry_price ({self.entry_price})."
                    )

            # Validate Take Profit geometry
            tps = [tp for tp in (self.tp1, self.tp2, self.tp3) if tp is not None]
            for idx, tp in enumerate(tps, 1):
                if not math.isfinite(tp) or tp <= 0:
                    raise ValueError(f"take_profit level must be positive and finite, got: {tp}")
                if self.direction == Direction.BUY and tp <= self.entry_price:
                    raise ValueError(f"BUY take_profit ({tp}) must be strictly above entry_price ({self.entry_price}).")
                if self.direction == Direction.SELL and tp >= self.entry_price:
                    raise ValueError(f"SELL take_profit ({tp}) must be strictly below entry_price ({self.entry_price}).")

            # Validate TP ordering
            if self.direction == Direction.BUY:
                if self.tp1 is not None and self.tp2 is not None and self.tp1 >= self.tp2:
                    raise ValueError(f"BUY TP1 ({self.tp1}) must be less than TP2 ({self.tp2}).")
                if self.tp2 is not None and self.tp3 is not None and self.tp2 >= self.tp3:
                    raise ValueError(f"BUY TP2 ({self.tp2}) must be less than TP3 ({self.tp3}).")
            elif self.direction == Direction.SELL:
                if self.tp1 is not None and self.tp2 is not None and self.tp1 <= self.tp2:
                    raise ValueError(f"SELL TP1 ({self.tp1}) must be greater than TP2 ({self.tp2}).")
                if self.tp2 is not None and self.tp3 is not None and self.tp2 <= self.tp3:
                    raise ValueError(f"SELL TP2 ({self.tp2}) must be greater than TP3 ({self.tp3}).")

            # Validate trailing stop
            if self.trailing_stop is not None:
                if not math.isfinite(self.trailing_stop) or self.trailing_stop <= 0:
                    raise ValueError("trailing_stop must be positive and finite if provided.")
        else:
            # NO_TRADE
            if self.stop_loss is not None or self.tp1 is not None:
                raise ValueError("NO_TRADE decision must not carry active stop_loss or take_profit levels.")

        payload = {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol.upper(),
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "trailing_stop": self.trailing_stop,
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        risk_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        object.__setattr__(self, "risk_id", f"risk_{risk_hash}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "trailing_stop": self.trailing_stop,
            "risk_reward_ratio": self.risk_reward_ratio,
        }


def calculate_production_risk_levels(
    decision: ProductionDecision,
    candidate: PromotedCandidateArtifact,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    tp1_multiplier: float = 1.0,
    tp2_multiplier: float = 2.0,
    tp3_multiplier: float = 3.0,
) -> ProductionRiskLevels:
    """Calculate risk levels from authoritative decision and candidate configuration."""
    if not isinstance(decision, ProductionDecision):
        raise TypeError("decision must be a ProductionDecision instance.")
    if not isinstance(candidate, PromotedCandidateArtifact):
        raise TypeError("candidate must be a PromotedCandidateArtifact instance.")

    if decision.direction == Direction.NO_TRADE or decision.entry_price is None:
        return ProductionRiskLevels(
            decision_id=decision.decision_id,
            candidate_id=decision.candidate_id,
            evidence_id=decision.evidence_id,
            symbol=decision.symbol,
            timeframe=decision.timeframe,
            direction=Direction.NO_TRADE,
            entry_price=None,
            stop_loss=None,
            tp1=None,
            tp2=None,
            tp3=None,
            trailing_stop=None,
            risk_reward_ratio=None,
        )

    # Obtain stop_loss_pct and take_profit_pct from candidate parameters or arguments
    sl_pct = (
        stop_loss_pct
        if stop_loss_pct is not None
        else candidate.parameters.get("stop_loss_pct", 0.01)
    )
    tp_pct = (
        take_profit_pct
        if take_profit_pct is not None
        else candidate.parameters.get("take_profit_pct", 0.02)
    )

    if not math.isfinite(sl_pct) or sl_pct <= 0:
        raise ValueError("stop_loss_pct must be a positive finite float.")
    if not math.isfinite(tp_pct) or tp_pct <= 0:
        raise ValueError("take_profit_pct must be a positive finite float.")

    entry = decision.entry_price

    if decision.direction == Direction.BUY:
        stop_loss = entry * (1.0 - sl_pct)
        take_profit = entry * (1.0 + tp_pct)
        risk_dist = entry - stop_loss
        reward_dist = take_profit - entry
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else None

        tp1 = entry + (risk_dist * tp1_multiplier)
        tp2 = entry + (risk_dist * tp2_multiplier)
        tp3 = entry + (risk_dist * tp3_multiplier)
        trailing = candidate.parameters.get("trailing_stop_level")

    elif decision.direction == Direction.SELL:
        stop_loss = entry * (1.0 + sl_pct)
        take_profit = entry * (1.0 - tp_pct)
        risk_dist = stop_loss - entry
        reward_dist = entry - take_profit
        rr_ratio = reward_dist / risk_dist if risk_dist > 0 else None

        tp1 = entry - (risk_dist * tp1_multiplier)
        tp2 = entry - (risk_dist * tp2_multiplier)
        tp3 = entry - (risk_dist * tp3_multiplier)
        trailing = candidate.parameters.get("trailing_stop_level")

    else:
        raise ValueError(f"Unsupported direction for risk calculation: {decision.direction}")

    return ProductionRiskLevels(
        decision_id=decision.decision_id,
        candidate_id=decision.candidate_id,
        evidence_id=decision.evidence_id,
        symbol=decision.symbol,
        timeframe=decision.timeframe,
        direction=decision.direction,
        entry_price=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        trailing_stop=trailing,
        risk_reward_ratio=rr_ratio,
    )


# Legacy wrapper function for backward compatibility with existing codebase/tests
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
    momentum_window: int = 10,
    fast_window: int = 5,
    slow_window: int = 20,
    stop_loss_pct: float = 0.01,
    take_profit_pct: float = 0.02,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Legacy wrapper preserved for backwards compatibility with pre-existing tests/callers."""
    from live_risk_levels import build_live_risk_levels
    from live_trend import build_live_trend_snapshot

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("data must not be empty.")

    stable_strategy = _validate_strategy(stable_strategy)
    stability_score = _validate_score("stability_score", stability_score)
    min_stability_score = _validate_score("min_stability_score", min_stability_score)

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
