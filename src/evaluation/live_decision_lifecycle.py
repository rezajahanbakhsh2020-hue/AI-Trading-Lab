"""Canonical Live Decision Lifecycle domain models, transitions, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionRiskLevels,
    ProductionRuntimeAuthorizationError,
    ProductionSignal,
)


class LiveDecisionLifecycleError(ValueError):
    """Raised when live decision lifecycle transition or validation fails."""


class LiveDecisionLifecycleState(str, Enum):
    """Authoritative lifecycle states for a canonical live decision."""

    AUTHORIZED = "AUTHORIZED"
    EVALUATED = "EVALUATED"
    RISK_VALIDATED = "RISK_VALIDATED"
    PRESENTABLE = "PRESENTABLE"
    PERSISTED = "PERSISTED"
    PUBLISHED = "PUBLISHED"

    # Terminal/non-success operational states
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


# Allowed state transitions mapping
PERMITTED_TRANSITIONS: dict[
    LiveDecisionLifecycleState, set[LiveDecisionLifecycleState]
] = {
    LiveDecisionLifecycleState.AUTHORIZED: {
        LiveDecisionLifecycleState.EVALUATED,
        LiveDecisionLifecycleState.BLOCKED,
        LiveDecisionLifecycleState.REJECTED,
        LiveDecisionLifecycleState.FAILED,
    },
    LiveDecisionLifecycleState.EVALUATED: {
        LiveDecisionLifecycleState.RISK_VALIDATED,
        LiveDecisionLifecycleState.BLOCKED,
        LiveDecisionLifecycleState.REJECTED,
        LiveDecisionLifecycleState.FAILED,
    },
    LiveDecisionLifecycleState.RISK_VALIDATED: {
        LiveDecisionLifecycleState.PRESENTABLE,
        LiveDecisionLifecycleState.BLOCKED,
        LiveDecisionLifecycleState.REJECTED,
        LiveDecisionLifecycleState.FAILED,
    },
    LiveDecisionLifecycleState.PRESENTABLE: {
        LiveDecisionLifecycleState.PERSISTED,
        LiveDecisionLifecycleState.BLOCKED,
        LiveDecisionLifecycleState.REJECTED,
        LiveDecisionLifecycleState.FAILED,
    },
    LiveDecisionLifecycleState.PERSISTED: {
        LiveDecisionLifecycleState.PUBLISHED,
        LiveDecisionLifecycleState.BLOCKED,
        LiveDecisionLifecycleState.REJECTED,
        LiveDecisionLifecycleState.FAILED,
    },
    LiveDecisionLifecycleState.PUBLISHED: set(),
    LiveDecisionLifecycleState.BLOCKED: set(),
    LiveDecisionLifecycleState.REJECTED: set(),
    LiveDecisionLifecycleState.FAILED: set(),
}


@dataclass(frozen=True)
class LifecycleTransitionRecord:
    """Immutable audit record of a single live decision state transition."""

    from_state: LiveDecisionLifecycleState
    to_state: LiveDecisionLifecycleState
    timestamp_utc: str
    actor: str
    artifact_fingerprint: str
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.from_state, LiveDecisionLifecycleState):
            raise LiveDecisionLifecycleError(
                f"from_state must be a LiveDecisionLifecycleState, got {type(self.from_state).__name__}"
            )
        if not isinstance(self.to_state, LiveDecisionLifecycleState):
            raise LiveDecisionLifecycleError(
                f"to_state must be a LiveDecisionLifecycleState, got {type(self.to_state).__name__}"
            )
        if not self.timestamp_utc or not str(self.timestamp_utc).strip():
            raise LiveDecisionLifecycleError("timestamp_utc must be a non-empty string.")
        if not self.actor or not str(self.actor).strip():
            raise LiveDecisionLifecycleError("actor must be a non-empty string.")
        if not self.artifact_fingerprint or not str(self.artifact_fingerprint).strip():
            raise LiveDecisionLifecycleError("artifact_fingerprint must be a non-empty string.")

        object.__setattr__(self, "timestamp_utc", str(self.timestamp_utc).strip())
        object.__setattr__(self, "actor", str(self.actor).strip())
        object.__setattr__(self, "artifact_fingerprint", str(self.artifact_fingerprint).strip())
        if self.reason is not None:
            r_str = str(self.reason).strip()
            object.__setattr__(self, "reason", r_str if r_str else None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp_utc": self.timestamp_utc,
            "actor": self.actor,
            "artifact_fingerprint": self.artifact_fingerprint,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LifecycleTransitionRecord:
        if not isinstance(data, Mapping):
            raise LiveDecisionLifecycleError("transition record data must be a mapping.")
        return cls(
            from_state=LiveDecisionLifecycleState(data["from_state"]),
            to_state=LiveDecisionLifecycleState(data["to_state"]),
            timestamp_utc=str(data["timestamp_utc"]),
            actor=str(data["actor"]),
            artifact_fingerprint=str(data["artifact_fingerprint"]),
            reason=data.get("reason"),
        )


@dataclass(frozen=True)
class CanonicalLiveDecision:
    """Immutable, canonical live decision artifact binding authorization, decision, signal, risk, state, and lineage."""

    live_decision_id: str
    authorization_receipt: ProductionAuthorizationReceipt
    decision: ProductionDecision
    signal: ProductionSignal
    risk_levels: ProductionRiskLevels
    current_state: LiveDecisionLifecycleState
    transition_history: tuple[LifecycleTransitionRecord, ...]

    canonical_live_decision_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.live_decision_id or not str(self.live_decision_id).strip():
            raise LiveDecisionLifecycleError("live_decision_id must be a non-empty string.")
        if not isinstance(self.authorization_receipt, ProductionAuthorizationReceipt):
            raise LiveDecisionLifecycleError(
                f"authorization_receipt must be a ProductionAuthorizationReceipt, got {type(self.authorization_receipt).__name__}"
            )
        if not isinstance(self.decision, ProductionDecision):
            raise LiveDecisionLifecycleError(
                f"decision must be a ProductionDecision, got {type(self.decision).__name__}"
            )
        if not isinstance(self.signal, ProductionSignal):
            raise LiveDecisionLifecycleError(
                f"signal must be a ProductionSignal, got {type(self.signal).__name__}"
            )
        if not isinstance(self.risk_levels, ProductionRiskLevels):
            raise LiveDecisionLifecycleError(
                f"risk_levels must be a ProductionRiskLevels, got {type(self.risk_levels).__name__}"
            )
        if not isinstance(self.current_state, LiveDecisionLifecycleState):
            raise LiveDecisionLifecycleError(
                f"current_state must be a LiveDecisionLifecycleState, got {type(self.current_state).__name__}"
            )

        if not isinstance(self.transition_history, tuple):
            if isinstance(self.transition_history, (list, Sequence)):
                object.__setattr__(self, "transition_history", tuple(self.transition_history))
            else:
                raise LiveDecisionLifecycleError("transition_history must be a tuple of LifecycleTransitionRecord.")

        for tr in self.transition_history:
            if not isinstance(tr, LifecycleTransitionRecord):
                raise LiveDecisionLifecycleError(
                    f"transition_history item must be LifecycleTransitionRecord, got {type(tr).__name__}"
                )

        # Cross-model consistency checks
        if self.signal.decision_id != self.decision.decision_id:
            raise LiveDecisionLifecycleError(
                f"Signal decision_id '{self.signal.decision_id}' does not match decision ID '{self.decision.decision_id}'."
            )
        if self.risk_levels.decision_id != self.decision.decision_id:
            raise LiveDecisionLifecycleError(
                f"Risk decision_id '{self.risk_levels.decision_id}' does not match decision ID '{self.decision.decision_id}'."
            )

        # Scope consistency checks
        receipt = self.authorization_receipt
        dec = self.decision
        if receipt.candidate_id != dec.candidate_id:
            raise LiveDecisionLifecycleError(
                f"Authorization candidate_id '{receipt.candidate_id}' does not match decision candidate_id '{dec.candidate_id}'."
            )
        if receipt.symbol.upper() != dec.symbol.upper():
            raise LiveDecisionLifecycleError(
                f"Authorization symbol '{receipt.symbol}' does not match decision symbol '{dec.symbol}'."
            )
        if receipt.timeframe != dec.timeframe:
            raise LiveDecisionLifecycleError(
                f"Authorization timeframe '{receipt.timeframe}' does not match decision timeframe '{dec.timeframe}'."
            )

        object.__setattr__(self, "live_decision_id", str(self.live_decision_id).strip())

        # Calculate canonical live decision fingerprint
        payload = {
            "live_decision_id": self.live_decision_id,
            "authorization_receipt": self.authorization_receipt.as_dict(),
            "decision": self.decision.as_dict(),
            "signal": self.signal.as_dict(),
            "risk_levels": self.risk_levels.as_dict(),
            "current_state": self.current_state.value,
            "transition_history": [tr.as_dict() for tr in self.transition_history],
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        fp = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        object.__setattr__(self, "canonical_live_decision_fingerprint", fp)

    # Mandatory Lineage Properties
    @property
    def candidate_id(self) -> str:
        return self.authorization_receipt.candidate_id

    @property
    def strategy_name(self) -> str:
        return self.authorization_receipt.strategy_name

    @property
    def strategy_version(self) -> str:
        return self.authorization_receipt.strategy_version

    @property
    def symbol(self) -> str:
        return self.authorization_receipt.symbol

    @property
    def timeframe(self) -> str:
        return self.authorization_receipt.timeframe

    @property
    def promoted_artifact_fingerprint(self) -> str:
        return self.authorization_receipt.promoted_artifact_fingerprint

    @property
    def governance_decision_fingerprint(self) -> str:
        return self.authorization_receipt.governance_decision_fingerprint

    @property
    def campaign_selection_decision_fingerprint(self) -> Optional[str]:
        return self.authorization_receipt.campaign_selection_decision_fingerprint

    @property
    def authorization_policy_version(self) -> str:
        return self.authorization_receipt.authorization_policy_version

    @property
    def authorization_fingerprint(self) -> str:
        return self.authorization_receipt.authorization_fingerprint

    @property
    def authorized_at_utc(self) -> str:
        return self.authorization_receipt.authorized_at_utc

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id

    @property
    def event_time_utc(self) -> str:
        return self.decision.market_timestamp

    def as_dict(self) -> dict[str, Any]:
        return {
            "live_decision_id": self.live_decision_id,
            "canonical_live_decision_fingerprint": self.canonical_live_decision_fingerprint,
            "current_state": self.current_state.value,
            "authorization_receipt": self.authorization_receipt.as_dict(),
            "decision": self.decision.as_dict(),
            "signal": self.signal.as_dict(),
            "risk_levels": self.risk_levels.as_dict(),
            "transition_history": [tr.as_dict() for tr in self.transition_history],
            "lineage": {
                "candidate_id": self.candidate_id,
                "strategy_name": self.strategy_name,
                "strategy_version": self.strategy_version,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "promoted_artifact_fingerprint": self.promoted_artifact_fingerprint,
                "governance_decision_fingerprint": self.governance_decision_fingerprint,
                "campaign_selection_decision_fingerprint": self.campaign_selection_decision_fingerprint,
                "authorization_policy_version": self.authorization_policy_version,
                "authorization_fingerprint": self.authorization_fingerprint,
                "authorized_at_utc": self.authorized_at_utc,
                "decision_id": self.decision_id,
                "event_time_utc": self.event_time_utc,
            },
        }


def create_canonical_live_decision(
    authorization_receipt: ProductionAuthorizationReceipt,
    decision: ProductionDecision,
    signal: ProductionSignal,
    risk_levels: ProductionRiskLevels,
    *,
    actor: str = "live_runtime",
    timestamp_utc: Optional[str] = None,
    reason: Optional[str] = "initial_authorization",
) -> CanonicalLiveDecision:
    """Factory function creating an initial CanonicalLiveDecision artifact in AUTHORIZED state."""
    if not isinstance(authorization_receipt, ProductionAuthorizationReceipt):
        raise LiveDecisionLifecycleError("authorization_receipt must be a ProductionAuthorizationReceipt.")

    if timestamp_utc is None:
        ts = datetime.now(timezone.utc).isoformat()
    else:
        ts = str(timestamp_utc).strip()

    live_dec_id = decision.decision_id

    prov_payload = {
        "live_decision_id": live_dec_id,
        "authorization_fingerprint": authorization_receipt.authorization_fingerprint,
        "decision_id": decision.decision_id,
        "state": LiveDecisionLifecycleState.AUTHORIZED.value,
    }
    prov_fp = hashlib.sha256(json.dumps(prov_payload, sort_keys=True).encode("utf-8")).hexdigest()

    init_tr = LifecycleTransitionRecord(
        from_state=LiveDecisionLifecycleState.AUTHORIZED,
        to_state=LiveDecisionLifecycleState.AUTHORIZED,
        timestamp_utc=ts,
        actor=actor,
        artifact_fingerprint=prov_fp,
        reason=reason,
    )

    return CanonicalLiveDecision(
        live_decision_id=live_dec_id,
        authorization_receipt=authorization_receipt,
        decision=decision,
        signal=signal,
        risk_levels=risk_levels,
        current_state=LiveDecisionLifecycleState.AUTHORIZED,
        transition_history=(init_tr,),
    )


def transition_live_decision(
    canonical_decision: CanonicalLiveDecision,
    target_state: LiveDecisionLifecycleState,
    *,
    actor: str,
    timestamp_utc: Optional[str] = None,
    reason: Optional[str] = None,
) -> CanonicalLiveDecision:
    """Transition a canonical live decision to a target state, producing a new updated CanonicalLiveDecision."""
    if not isinstance(canonical_decision, CanonicalLiveDecision):
        raise LiveDecisionLifecycleError("canonical_decision must be a CanonicalLiveDecision instance.")

    if not isinstance(target_state, LiveDecisionLifecycleState):
        raise LiveDecisionLifecycleError(
            f"target_state must be a LiveDecisionLifecycleState, got {type(target_state).__name__}"
        )

    current_state = canonical_decision.current_state

    # Check allowed transitions
    allowed = PERMITTED_TRANSITIONS.get(current_state, set())
    if target_state not in allowed:
        raise LiveDecisionLifecycleError(
            f"Illegal lifecycle state transition from {current_state.value} to {target_state.value}."
        )

    # Validate current decision prior to transition
    validate_live_decision_lifecycle(canonical_decision)

    if timestamp_utc is None:
        ts = datetime.now(timezone.utc).isoformat()
    else:
        ts = str(timestamp_utc).strip()

    # Record fingerprint of canonical decision before transition
    tr = LifecycleTransitionRecord(
        from_state=current_state,
        to_state=target_state,
        timestamp_utc=ts,
        actor=actor,
        artifact_fingerprint=canonical_decision.canonical_live_decision_fingerprint,
        reason=reason,
    )

    new_history = canonical_decision.transition_history + (tr,)

    updated_decision = CanonicalLiveDecision(
        live_decision_id=canonical_decision.live_decision_id,
        authorization_receipt=canonical_decision.authorization_receipt,
        decision=canonical_decision.decision,
        signal=canonical_decision.signal,
        risk_levels=canonical_decision.risk_levels,
        current_state=target_state,
        transition_history=new_history,
    )

    # Validate resulting decision artifact
    validate_live_decision_lifecycle(updated_decision)

    return updated_decision


def validate_live_decision_lifecycle(
    canonical_decision: CanonicalLiveDecision,
) -> bool:
    """Validate structural integrity, authorization, state transitions, history, and risk geometry of a CanonicalLiveDecision."""
    if not isinstance(canonical_decision, CanonicalLiveDecision):
        raise LiveDecisionLifecycleError("canonical_decision must be a CanonicalLiveDecision instance.")

    # 1. Identity Check
    if canonical_decision.live_decision_id != canonical_decision.decision.decision_id:
        raise LiveDecisionLifecycleError(
            f"live_decision_id '{canonical_decision.live_decision_id}' does not match decision_id '{canonical_decision.decision.decision_id}'."
        )

    # 2. Authorization & Provenance Check
    receipt = canonical_decision.authorization_receipt
    if not isinstance(receipt, ProductionAuthorizationReceipt):
        raise LiveDecisionLifecycleError("Missing or invalid authorization_receipt.")

    if receipt.candidate_id != canonical_decision.decision.candidate_id:
        raise LiveDecisionLifecycleError(
            f"Authorization candidate_id '{receipt.candidate_id}' mismatch with decision candidate_id '{canonical_decision.decision.candidate_id}'."
        )
    if receipt.symbol != canonical_decision.decision.symbol:
        raise LiveDecisionLifecycleError(
            f"Authorization symbol '{receipt.symbol}' mismatch with decision symbol '{canonical_decision.decision.symbol}'."
        )
    if receipt.timeframe != canonical_decision.decision.timeframe:
        raise LiveDecisionLifecycleError(
            f"Authorization timeframe '{receipt.timeframe}' mismatch with decision timeframe '{canonical_decision.decision.timeframe}'."
        )

    # Check signal & risk level IDs
    if canonical_decision.signal.decision_id != canonical_decision.decision.decision_id:
        raise LiveDecisionLifecycleError("Signal decision_id mismatch.")
    if canonical_decision.risk_levels.decision_id != canonical_decision.decision.decision_id:
        raise LiveDecisionLifecycleError("Risk decision_id mismatch.")

    # 3. State & History Legal Progression Check
    history = canonical_decision.transition_history
    if not history:
        raise LiveDecisionLifecycleError("Transition history must not be empty.")

    # Check initial transition
    first_tr = history[0]
    if first_tr.from_state != LiveDecisionLifecycleState.AUTHORIZED or first_tr.to_state != LiveDecisionLifecycleState.AUTHORIZED:
        raise LiveDecisionLifecycleError("First transition must start at AUTHORIZED -> AUTHORIZED.")

    prev_state = LiveDecisionLifecycleState.AUTHORIZED
    for idx, tr in enumerate(history[1:], 1):
        if tr.from_state != prev_state:
            raise LiveDecisionLifecycleError(
                f"Transition history gap at step {idx}: expected from_state '{prev_state.value}', got '{tr.from_state.value}'."
            )
        allowed = PERMITTED_TRANSITIONS.get(prev_state, set())
        if tr.to_state not in allowed:
            raise LiveDecisionLifecycleError(
                f"Illegal transition recorded in history at step {idx}: from '{prev_state.value}' to '{tr.to_state.value}'."
            )
        prev_state = tr.to_state

    if canonical_decision.current_state != prev_state:
        raise LiveDecisionLifecycleError(
            f"current_state '{canonical_decision.current_state.value}' does not match final transition history state '{prev_state.value}'."
        )

    # 4. Risk Level Consistency in Post-Evaluation States
    evaluated_states = (
        LiveDecisionLifecycleState.RISK_VALIDATED,
        LiveDecisionLifecycleState.PRESENTABLE,
        LiveDecisionLifecycleState.PERSISTED,
        LiveDecisionLifecycleState.PUBLISHED,
    )
    if canonical_decision.current_state in evaluated_states:
        dec = canonical_decision.decision
        risk = canonical_decision.risk_levels
        if dec.direction in (Direction.BUY, Direction.SELL):
            if dec.entry_price is None or dec.entry_price <= 0:
                raise LiveDecisionLifecycleError(f"{dec.direction.value} decision in {canonical_decision.current_state.value} missing valid entry_price.")
            if risk.stop_loss is None or risk.stop_loss <= 0:
                raise LiveDecisionLifecycleError(f"{dec.direction.value} decision in {canonical_decision.current_state.value} missing valid stop_loss.")
            if risk.tp1 is None or risk.tp1 <= 0:
                raise LiveDecisionLifecycleError(f"{dec.direction.value} decision in {canonical_decision.current_state.value} missing valid tp1.")
        elif dec.direction == Direction.NO_TRADE:
            if risk.stop_loss is not None or risk.tp1 is not None:
                raise LiveDecisionLifecycleError("NO_TRADE decision must not carry active risk levels.")

    return True
