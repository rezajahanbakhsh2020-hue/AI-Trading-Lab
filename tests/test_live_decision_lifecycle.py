"""Comprehensive focused test suite for Canonical Live Decision Lifecycle (PR #47 Follow-up)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
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
from src.evaluation.live_decision_store import (
    append_live_decision_to_store,
    persist_canonical_live_decision,
)
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
from src.evaluation.live_publication_store import (
    PublicationIntegrityError,
    append_publication_record,
    publish_canonical_live_decision,
)
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
    ref_now = data["timestamp"].iloc[-1] + pd.Timedelta(minutes=5)
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
    assert cld.transition_history[0].from_state is None
    assert cld.transition_history[0].to_state == LiveDecisionLifecycleState.AUTHORIZED
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
        operational_stability_score=receipt.operational_stability_score,
    )

    with pytest.raises(LiveDecisionLifecycleError, match="candidate_id"):
        create_canonical_live_decision(receipt_bad, decision, signal, risk)


# --- Test E: Lifecycle Transitions ---
def test_lifecycle_transitions_valid_and_invalid():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc="2026-01-01T10:00:00+00:00")

    # Valid progression
    cld_eval = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-01-01T10:01:00+00:00")
    assert cld_eval.current_state == LiveDecisionLifecycleState.EVALUATED

    cld_risk = transition_live_decision(cld_eval, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test", timestamp_utc="2026-01-01T10:02:00+00:00")
    assert cld_risk.current_state == LiveDecisionLifecycleState.RISK_VALIDATED

    cld_pres = transition_live_decision(cld_risk, LiveDecisionLifecycleState.PRESENTABLE, actor="test", timestamp_utc="2026-01-01T10:03:00+00:00")
    assert cld_pres.current_state == LiveDecisionLifecycleState.PRESENTABLE

    # Invalid state transition jump (PRESENTABLE -> PUBLISHED skipping PERSISTED)
    with pytest.raises(LiveDecisionLifecycleError, match="Illegal lifecycle state transition"):
        transition_live_decision(cld_pres, LiveDecisionLifecycleState.PUBLISHED, actor="test", timestamp_utc="2026-01-01T10:04:00+00:00")

    # Backward transition (EVALUATED -> AUTHORIZED)
    with pytest.raises(LiveDecisionLifecycleError, match="Illegal lifecycle state transition"):
        transition_live_decision(cld_eval, LiveDecisionLifecycleState.AUTHORIZED, actor="test", timestamp_utc="2026-01-01T10:04:00+00:00")


# --- Test F: Transition History & Chained Fingerprint Integrity ---
def test_transition_history_and_chained_fingerprint_integrity():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc="2026-01-01T10:00:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-01-01T10:01:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test", timestamp_utc="2026-01-01T10:02:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test", timestamp_utc="2026-01-01T10:03:00+00:00")

    assert len(cld.transition_history) == 4
    states = [tr.to_state for tr in cld.transition_history]
    assert states == [
        LiveDecisionLifecycleState.AUTHORIZED,
        LiveDecisionLifecycleState.EVALUATED,
        LiveDecisionLifecycleState.RISK_VALIDATED,
        LiveDecisionLifecycleState.PRESENTABLE,
    ]

    # Forged transition artifact_fingerprint fails validation
    forged_tr = LifecycleTransitionRecord(
        from_state=LiveDecisionLifecycleState.EVALUATED,
        to_state=LiveDecisionLifecycleState.RISK_VALIDATED,
        timestamp_utc="2026-01-01T10:02:00+00:00",
        actor="test",
        artifact_fingerprint="forged_fingerprint_hash_00000000000000000000000000000000000",
        reason="forged",
    )
    forged_history = (cld.transition_history[0], cld.transition_history[1], forged_tr, cld.transition_history[3])
    bad_cld = CanonicalLiveDecision(
        live_decision_id=cld.live_decision_id,
        authorization_receipt=cld.authorization_receipt,
        decision=cld.decision,
        signal=cld.signal,
        risk_levels=cld.risk_levels,
        current_state=cld.current_state,
        transition_history=forged_history,
    )
    with pytest.raises(LiveDecisionLifecycleError, match="artifact_fingerprint mismatch"):
        validate_live_decision_lifecycle(bad_cld)


# --- Test G: Temporal UTC Chronology Monotonicity ---
def test_temporal_utc_chronology_monotonicity():
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc="2026-01-01T10:05:00+00:00")

    # Backwards timestamp fails closed
    with pytest.raises(LiveDecisionLifecycleError, match="moves backwards"):
        transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-01-01T10:04:00+00:00")

    # Naive timestamp fails closed
    with pytest.raises(LiveDecisionLifecycleError, match="must be timezone-aware"):
        transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-01-01 10:06:00")


# --- Test H: Enforced Persistence Lifecycle Boundary ---
def test_enforced_persistence_lifecycle_boundary(tmp_path):
    receipt, decision, signal, risk, _ = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc="2026-01-01T10:00:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-01-01T10:01:00+00:00")

    # Attempting to persist from EVALUATED fails closed
    store_path = tmp_path / "decision_history.json"
    with pytest.raises(LiveDecisionLifecycleError, match="expected state 'PRESENTABLE'"):
        persist_canonical_live_decision(cld, store_path, timestamp_utc="2026-01-01T10:02:00+00:00")

    # Transition to PRESENTABLE first
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test", timestamp_utc="2026-01-01T10:02:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test", timestamp_utc="2026-01-01T10:03:00+00:00")

    # Now persist_canonical_live_decision succeeds and returns PERSISTED artifact
    persisted_cld = persist_canonical_live_decision(cld, store_path, timestamp_utc="2026-01-01T10:04:00+00:00")
    assert persisted_cld.current_state == LiveDecisionLifecycleState.PERSISTED
    assert len(persisted_cld.transition_history) == 5

    # Re-persisting identical PERSISTED artifact is idempotent
    re_persisted = persist_canonical_live_decision(persisted_cld, store_path)
    assert re_persisted.canonical_live_decision_fingerprint == persisted_cld.canonical_live_decision_fingerprint


# --- Test I: Enforced Publication Lifecycle Boundary ---
def test_enforced_publication_lifecycle_boundary(tmp_path):
    receipt, decision, signal, risk, candidate = _setup_canonical_components()
    cld = create_canonical_live_decision(receipt, decision, signal, risk, timestamp_utc="2026-01-01T10:00:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc="2026-01-01T10:01:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test", timestamp_utc="2026-01-01T10:02:00+00:00")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test", timestamp_utc="2026-01-01T10:03:00+00:00")

    pub_store = tmp_path / "publication_history.json"

    # Attempting to publish from PRESENTABLE without persistence fails closed
    with pytest.raises(LiveDecisionLifecycleError, match="Publication cannot bypass persistence"):
        publish_canonical_live_decision(cld, publisher=None, candidate=candidate, path=pub_store, timestamp_utc="2026-01-01T10:04:00+00:00")

    # Persist first
    store_path = tmp_path / "decision_history.json"
    persisted_cld = persist_canonical_live_decision(cld, store_path, timestamp_utc="2026-01-01T10:04:00+00:00")

    # Now publish_canonical_live_decision succeeds and returns PUBLISHED artifact
    mock_pub = MagicMock()
    mock_pub.publish.return_value = {"status": "PUBLISHED", "published": True}

    pub_cld, publication, pub_res = publish_canonical_live_decision(
        persisted_cld, publisher=mock_pub, candidate=candidate, path=pub_store, timestamp_utc="2026-01-01T10:05:00+00:00"
    )

    assert pub_cld.current_state == LiveDecisionLifecycleState.PUBLISHED
    assert publication.provenance["canonical_live_decision_fingerprint"] == pub_cld.canonical_live_decision_fingerprint
    assert publication.provenance["current_lifecycle_state"] == "PUBLISHED"


# --- Test J: Runtime Bypass Regression ---
def test_runtime_bypass_regression():
    data = _data()
    # Unpromoted strategy must fail closed
    with pytest.raises(ValueError, match="No authoritative promoted candidate resolved"):
        build_live_runtime(data, stable_strategy="non_existent_strategy", stability_score=None)


# --- Test K: Display Bypass Regression ---
def test_display_bypass_regression(monkeypatch):
    data = _data()
    # Prove build_live_trade_display does not call runtime or decision evaluation when canonical_decision is None
    monkeypatch.setattr("src.evaluation.live_runtime.build_live_runtime", pytest.fail, raising=False)
    monkeypatch.setattr("src.evaluation.live_production_decision.build_live_production_decision", pytest.fail, raising=False)
    monkeypatch.setattr("src.evaluation.live_production_decision.authorize_production_runtime", pytest.fail, raising=False)
    monkeypatch.setattr("src.evaluation.research_store.resolve_promoted_candidate", pytest.fail, raising=False)

    display = build_live_trade_display(data, canonical_decision=None, stable_strategy="non_existent_strategy", stability_score=None)
    assert display["decision"] == "BLOCKED"
    assert display["reason"] == "missing_canonical_live_decision"


# --- Test L: Runtime and Display Identity Parity ---
def test_runtime_and_display_identity_parity(tmp_path):
    data = _data()
    ref_now = pd.to_datetime(data["timestamp"], utc=True).iloc[-1].to_pydatetime() + pd.Timedelta(minutes=5)
    res = build_live_runtime(
        data,
        stable_strategy="momentum",
        stability_score=None,
        reference_now=ref_now,
        store_path=tmp_path / "store.json",
        persist=False,
    )

    assert res.canonical_decision is not None
    assert res.decision["decision_id"] == res.canonical_decision.decision.decision_id
    assert res.decision["canonical_live_decision_fingerprint"] == res.display["canonical_live_decision_fingerprint"]
    assert res.decision["runtime_authorization_fingerprint"] == res.display["authorization_fingerprint"]


# --- Test M: Persistence Replay Idempotency & Conflict Protection ---
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


# --- Test N: Missing Authorization Blocks Execution ---
def test_missing_authorization_blocks_execution():
    candidate = resolve_promoted_candidate(candidate_id="cand_momentum_5m")
    assert candidate is not None

    runtime = LiveExecutionRuntime(
        symbol="EURUSD",
        production_config=ProductionRuntimeConfig(symbol="EURUSD", timeframe="5m", candidate_id="cand_momentum_5m"),
    )
    res = runtime.run_once(publish=False, persist=False)

    assert res["blocked"] is True
    assert res["decision"] == "NO TRADE"


# --- Test O: No Synthetic Fallback ---
def test_no_synthetic_fallback():
    data = _data()
    # Unsupported/unpromoted strategy MUST raise or block, NEVER return synthetic BUY/SELL
    with pytest.raises(ValueError, match="No authoritative promoted candidate resolved"):
        build_live_runtime(data, stable_strategy="synthetic_default_strategy", stability_score=0.99)
