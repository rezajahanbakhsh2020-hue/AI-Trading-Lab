"""Comprehensive focused test suite for Canonical Live Decision Lifecycle (PR #47)."""

from datetime import datetime, timezone
import pytest
import pandas as pd

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LifecycleTransitionRecord,
    LiveDecisionLifecycleError,
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
    transition_live_decision,
    validate_live_decision_lifecycle,
)
from src.evaluation.live_decision_record import build_live_decision_record
from src.evaluation.live_decision_store import append_live_decision_to_store
from src.evaluation.live_execution_runtime import LiveExecutionRuntime, ProductionRuntimeConfig
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionRuntimeAuthorizationError,
    ProductionSignal,
    authorize_production_runtime,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_publication_store import PublicationIntegrityError, append_publication_record
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.live_trade_display import build_live_trade_display
from src.evaluation.research_store import resolve_promoted_candidate


def _data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min")
    close = pd.Series([2000.0 + i for i in range(rows)], dtype=float)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
    })


def _setup_canonical_components():
    candidate = resolve_promoted_candidate(candidate_id="cand_momentum_5m")
    assert candidate is not None
    data = _data()
    ref_now = data["timestamp"].iloc[-1]
    auth = authorize_production_runtime(candidate, symbol="XAUUSD", timeframe="5m", now=ref_now)
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    decision = evaluate_production_decision(candidate, data, reference_now=ref_now)

    now_ts = pd.to_datetime(ref_now, utc=True).isoformat()
    buy_decision = ProductionDecision(
        candidate_id=candidate.candidate_id,
        evidence_id=candidate.evidence.evidence_id,
        experiment_fingerprint=candidate.evidence.experiment_fingerprint,
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=now_ts,
        market_timestamp=now_ts,
        direction=Direction.BUY,
        reason="test_buy",
        entry_price=2000.0,
        invalidation_condition="Close below SL",
        confidence=0.85,
        parameters=candidate.parameters,
    )

    signal = ProductionSignal.from_decision(buy_decision)
    risk = calculate_production_risk_levels(buy_decision, candidate)
    return receipt, buy_decision, signal, risk, candidate


# --- Test A: Canonical Artifact Creation ---
def test_canonical_artifact_creation():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk)

    assert isinstance(cld, CanonicalLiveDecision)
    assert cld.live_decision_id == decision.decision_id
    assert cld.current_state == LiveDecisionLifecycleState.AUTHORIZED
    assert len(cld.transition_history) == 1
    assert cld.canonical_live_decision_fingerprint is not None
    assert len(cld.canonical_live_decision_fingerprint) == 64


# --- Test B: Deterministic Fingerprint ---
def test_deterministic_fingerprint():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    ts = "2026-01-01T00:00:00+00:00"

    cld1 = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc=ts)
    cld2 = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc=ts)

    assert cld1.canonical_live_decision_fingerprint == cld2.canonical_live_decision_fingerprint

    # Different decision identity produces different fingerprint
    dec_diff = ProductionDecision(
        candidate_id=decision.candidate_id,
        evidence_id=decision.evidence_id,
        experiment_fingerprint=decision.experiment_fingerprint,
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=ts,
        market_timestamp=ts,
        direction=Direction.BUY,
        reason="different_reason",
        entry_price=2010.0,
        invalidation_condition="Close below SL",
        confidence=0.85,
        parameters=decision.parameters,
    )
    sig_diff = ProductionSignal.from_decision(dec_diff)
    candidate = resolve_promoted_candidate(candidate_id="cand_momentum_5m")
    risk_diff = calculate_production_risk_levels(dec_diff, candidate)

    cld3 = create_canonical_live_decision(receipt, dec_diff, sig_diff, risk_diff, timestamp_utc=ts)
    assert cld1.canonical_live_decision_fingerprint != cld3.canonical_live_decision_fingerprint


# --- Test C: Immutability ---
def test_artifact_immutability():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk)

    with pytest.raises(AttributeError):
        cld.current_state = LiveDecisionLifecycleState.EVALUATED

    with pytest.raises(AttributeError):
        cld.live_decision_id = "hacked_id"


# --- Test D: Authorization Mismatch Rejection ---
def test_authorization_mismatch_rejection():
    receipt, decision, signal, risk, _ = _setup_canonical_components()

    # Mismatched candidate_id in receipt
    receipt_bad = ProductionAuthorizationReceipt(
        candidate_id="other_cand",
        strategy_name=receipt.strategy_name,
        strategy_version=receipt.strategy_version,
        symbol=receipt.symbol,
        timeframe=receipt.timeframe,
        promoted_artifact_fingerprint=receipt.promoted_artifact_fingerprint,
        governance_decision_fingerprint=receipt.governance_decision_fingerprint,
        campaign_selection_decision_fingerprint=receipt.campaign_selection_decision_fingerprint,
        authorization_policy_version=receipt.authorization_policy_version,
        authorized_at_utc=receipt.authorized_at_utc,
        authorization_fingerprint=receipt.authorization_fingerprint,
    )

    with pytest.raises(LiveDecisionLifecycleError, match="candidate_id"):
        create_canonical_live_decision(receipt_bad, decision, signal, risk)


# --- Test E: Lifecycle Transitions ---
def test_lifecycle_transitions_valid_and_invalid():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk)

    # Valid progression
    cld_eval = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test")
    assert cld_eval.current_state == LiveDecisionLifecycleState.EVALUATED

    cld_risk = transition_live_decision(cld_eval, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test")
    assert cld_risk.current_state == LiveDecisionLifecycleState.RISK_VALIDATED

    cld_pres = transition_live_decision(cld_risk, LiveDecisionLifecycleState.PRESENTABLE, actor="test")
    assert cld_pres.current_state == LiveDecisionLifecycleState.PRESENTABLE

    # Invalid state transition jump (PRESENTABLE -> PUBLISHED skipping PERSISTED)
    with pytest.raises(LiveDecisionLifecycleError, match="Illegal lifecycle state transition"):
        transition_live_decision(cld_pres, LiveDecisionLifecycleState.PUBLISHED, actor="test")

    # Backward transition (EVALUATED -> AUTHORIZED)
    with pytest.raises(LiveDecisionLifecycleError, match="Illegal lifecycle state transition"):
        transition_live_decision(cld_eval, LiveDecisionLifecycleState.AUTHORIZED, actor="test")


# --- Test F: Transition History Integrity ---
def test_transition_history_integrity():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk)
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test")

    assert len(cld.transition_history) == 4
    states = [tr.to_state for tr in cld.transition_history]
    assert states == [
        LiveDecisionLifecycleState.AUTHORIZED,
        LiveDecisionLifecycleState.EVALUATED,
        LiveDecisionLifecycleState.RISK_VALIDATED,
        LiveDecisionLifecycleState.PRESENTABLE,
    ]

    # Attempt to forge invalid history
    bad_history = cld.transition_history[:2] + (cld.transition_history[3],)
    bad_cld = CanonicalLiveDecision(
        live_decision_id=cld.live_decision_id,
        authorization_receipt=cld.authorization_receipt,
        decision=cld.decision,
        signal=cld.signal,
        risk_levels=cld.risk_levels,
        current_state=cld.current_state,
        transition_history=bad_history,
    )
    with pytest.raises(LiveDecisionLifecycleError, match="Transition history gap"):
        validate_live_decision_lifecycle(bad_cld)


# --- Test G: Runtime Bypass Regression ---
def test_runtime_bypass_regression():
    data = _data()
    # Unpromoted strategy must fail closed
    with pytest.raises(ValueError, match="No authoritative promoted candidate resolved"):
        build_live_runtime(data, stable_strategy="non_existent_strategy", stability_score=0.80)


# --- Test H: Display Bypass Regression ---
def test_display_bypass_regression():
    data = _data()
    # Direct display call for unpromoted strategy must fail closed
    with pytest.raises(ValueError, match="No authoritative promoted candidate resolved"):
        build_live_trade_display(data, stable_strategy="non_existent_strategy", stability_score=0.80)


# --- Test I: Runtime and Display Identity Parity ---
def test_runtime_and_display_identity_parity():
    data = _data()
    res = build_live_runtime(data, stable_strategy="momentum", stability_score=0.80)

    assert res.canonical_decision is not None
    assert res.decision["decision_id"] == res.display["decision_id"]
    assert res.decision["canonical_live_decision_fingerprint"] == res.display["canonical_live_decision_fingerprint"]
    assert res.decision["runtime_authorization_fingerprint"] == res.display["authorization_fingerprint"]


# --- Test J: Persistence Replay Idempotency & Conflict Protection ---
def test_persistence_replay_idempotency_and_conflict(tmp_path):
    receipt, decision, signal, risk, candidate = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk)
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test")

    rec = build_live_decision_record({
        "timestamp": decision.market_timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "strategy": "momentum",
        "entry_price": 2000.0,
        "stop_loss": 1980.0,
        "take_profit": 2040.0,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.85,
        "decision_id": decision.decision_id,
        "canonical_live_decision_fingerprint": cld.canonical_live_decision_fingerprint,
        "current_lifecycle_state": cld.current_state.value,
        "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
        "authorization_policy_version": receipt.authorization_policy_version,
        "authorized_at_utc": receipt.authorized_at_utc,
        "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
        "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
        "candidate_id": receipt.candidate_id,
        "strategy_name": receipt.strategy_name,
        "strategy_version": receipt.strategy_version,
    })

    store_path = tmp_path / "history.json"
    h1 = append_live_decision_to_store(rec, store_path)
    assert len(h1) == 1

    # Identical replay is idempotent
    h2 = append_live_decision_to_store(rec, store_path)
    assert len(h2) == 1

    # Conflicting fingerprint for same decision_id fails closed
    rec_conflict = dict(rec)
    rec_conflict["canonical_live_decision_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="canonical_live_decision_fingerprint mismatch"):
        append_live_decision_to_store(rec_conflict, store_path)


# --- Test K: Publication Integrity ---
def test_publication_integrity(tmp_path):
    receipt, decision, signal, risk, candidate = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk)

    # Attach canonical fingerprint to decision object for publication provenance
    object.__setattr__(decision, "canonical_live_decision_fingerprint", cld.canonical_live_decision_fingerprint)
    object.__setattr__(decision, "current_lifecycle_state", cld.current_state.value)

    pub = ProductionIntelligencePublication.from_artifacts(
        decision=decision,
        signal=signal,
        risk=risk,
        candidate=candidate,
        authorization=receipt,
    )

    assert pub.provenance.get("canonical_live_decision_fingerprint") == cld.canonical_live_decision_fingerprint
    assert pub.provenance.get("current_lifecycle_state") == cld.current_state.value

    pub_store = tmp_path / "pub_history.json"
    h1 = append_publication_record(pub.as_dict(), pub_store)
    assert len(h1) == 1

    # Identical replay is idempotent
    h2 = append_publication_record(pub.as_dict(), pub_store)
    assert len(h2) == 1

    # Conflicting canonical fingerprint in publication fails closed
    pub_bad = dict(pub.as_dict())
    pub_bad["provenance"] = dict(pub.provenance)
    pub_bad["provenance"]["canonical_live_decision_fingerprint"] = "f" * 64
    with pytest.raises(PublicationIntegrityError, match="canonical_live_decision_fingerprint mismatch"):
        append_publication_record(pub_bad, pub_store)


# --- Test L: Missing Authorization Blocks Execution ---
def test_missing_authorization_blocks_execution():
    candidate = resolve_promoted_candidate(candidate_id="cand_momentum_5m")
    assert candidate is not None

    # Mismatched symbol scope in config causes authorization failure
    runtime = LiveExecutionRuntime(
        symbol="EURUSD",
        production_config=ProductionRuntimeConfig(symbol="EURUSD", timeframe="5m", candidate_id="cand_momentum_5m"),
    )
    res = runtime.run_once(publish=False, persist=False)

    assert res["blocked"] is True
    assert res["decision"] == "NO TRADE"


# --- Test M: Missing Candidate Blocks Runtime ---
def test_missing_candidate_blocks_runtime():
    runtime = LiveExecutionRuntime(symbol="XAUUSD", production_config=ProductionRuntimeConfig(symbol="XAUUSD", timeframe="5m", candidate_id="cand_non_existent"))
    res = runtime.run_once(publish=False, persist=False)

    assert res["blocked"] is True
    assert res["decision"] == "NO TRADE"
    assert "PromotionUnavailable" in res["reason"]


# --- Test N: No Synthetic Fallback ---
def test_no_synthetic_fallback():
    data = _data()
    # Unsupported/unpromoted strategy MUST raise or block, NEVER return synthetic BUY/SELL
    with pytest.raises(ValueError, match="No authoritative promoted candidate resolved"):
        build_live_runtime(data, stable_strategy="synthetic_default_strategy", stability_score=0.99)
