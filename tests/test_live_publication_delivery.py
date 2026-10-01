"""Focused tests for Canonical Live Publication Delivery Receipt & Recovery Ledger."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
    transition_live_decision,
)
from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRuntimeAuthorization,
    ProductionSignal,
    authorize_production_runtime,
    calculate_production_risk_levels,
)
from src.evaluation.live_publication_delivery import (
    DeliveryIntegrityError,
    DeliveryStatus,
    PublicationDeliveryReceipt,
    append_delivery_receipt,
    compute_delivery_receipt_fingerprint,
    load_delivery_history,
)
from src.evaluation.live_publication_store import (
    publish_canonical_live_decision,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.evaluation.research_store import resolve_promoted_candidate, save_research_candidate
from src.integration.project2_publisher import Project2Publisher


def create_test_candidate_and_receipt(tmp_path, candidate_id="cand_delivery_test", symbol="XAUUSD"):
    ds = DatasetScope(
        dataset_id=f"ds_{symbol.lower()}_5m",
        symbol=symbol,
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    spec = ResearchExperimentSpec(
        hypothesis="Delivery test candidate",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.001, slippage=0.001, latency_ms=10.0
        ),
        code_provenance=CodeProvenance(commit_sha="e52d95d1ede22cf3c8ce07dc216763ace4a4359c"),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )
    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-01-02",
        total_return=0.20,
        max_drawdown=0.05,
        sharpe_ratio=2.0,
        observations=50,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-02T00:00:00+00:00",
    )
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-03",
        end_date="2025-01-05",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=30,
        start_timestamp_utc="2025-01-03T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-01-05",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-01-05T00:00:00+00:00",
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_oos, part_wf),
        robustness_verdict={
            "passed": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )
    save_research_candidate(candidate_id=candidate_id, evidence=evidence, operational_stability_score=0.85, base_dir=tmp_path)
    candidate = resolve_promoted_candidate(candidate_id=candidate_id, base_dir=tmp_path)
    auth = authorize_production_runtime(candidate, symbol=symbol, timeframe="5m")
    auth_receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    return candidate, auth_receipt


def create_persisted_cld(candidate, auth_receipt, direction=Direction.BUY, timestamp_utc="2025-01-01T10:00:00+00:00"):
    decision = ProductionDecision(
        candidate_id=auth_receipt.candidate_id,
        evidence_id="ev_test",
        experiment_fingerprint="exp_test",
        symbol=auth_receipt.symbol,
        timeframe=auth_receipt.timeframe,
        decision_timestamp=timestamp_utc,
        market_timestamp=timestamp_utc,
        direction=direction,
        reason="test_decision",
        entry_price=2000.0 if direction == Direction.BUY else None,
        invalidation_condition="test",
        confidence=0.85,
        parameters={"momentum_window": 10},
    )
    signal = ProductionSignal.from_decision(decision)
    risk = calculate_production_risk_levels(decision, candidate)

    cld = create_canonical_live_decision(
        authorization_receipt=auth_receipt,
        decision=decision,
        signal=signal,
        risk_levels=risk,
        actor="test",
        timestamp_utc=timestamp_utc,
    )
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test", timestamp_utc=timestamp_utc)
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test", timestamp_utc=timestamp_utc)
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test", timestamp_utc=timestamp_utc)
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PERSISTED, actor="test", timestamp_utc=timestamp_utc)
    return cld


# A. Fresh publication: Valid canonical decision reaches PERSISTED -> PUBLISHED and creates DELIVERED receipt
def test_fresh_publication_delivers_and_transitions_to_published(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "PUBLISHED",
        "published": True,
        "http_code": 200,
        "event_id": "evt_123",
    }

    pub_path = tmp_path / "publication_history.json"
    del_path = tmp_path / "delivery_history.json"

    published_cld, publication, pub_res = publish_canonical_live_decision(
        canonical_decision=cld,
        publisher=mock_publisher,
        candidate=candidate,
        path=pub_path,
        delivery_path=del_path,
    )

    assert published_cld.current_state == LiveDecisionLifecycleState.PUBLISHED
    assert pub_res["delivery_status"] == "DELIVERED"
    assert pub_res["delivery_receipt_fingerprint"] is not None

    receipts = load_delivery_history(del_path)
    assert len(receipts) == 1
    assert receipts[0].delivery_status == DeliveryStatus.DELIVERED
    assert receipts[0].canonical_live_decision_fingerprint == cld.canonical_live_decision_fingerprint
    assert receipts[0].runtime_authorization_fingerprint == auth_receipt.authorization_fingerprint


# B. Stale/no-trade publication: Stale path uses SAME canonical boundary
def test_stale_no_trade_path_uses_same_canonical_publication_boundary(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)

    timestamps = pd.date_range("2025-01-01 10:00", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "openTime": timestamps,
        "open": [2000.0] * 10,
        "high": [2005.0] * 10,
        "low": [1995.0] * 10,
        "close": [2002.0] * 10,
        "timestamp": timestamps,
    })

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "SKIPPED_NO_TRADE",
        "published": False,
        "reason": "Decision is NO TRADE and skip_if_no_trade=True",
    }

    config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id=candidate.candidate_id,
        strategy_id="momentum",
        research_dir=tmp_path,
    )

    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=tmp_path / "live" / "store.json",
        snapshot_path=tmp_path / "live" / "snap.json",
        research_dir=tmp_path,
        production_config=config,
    )

    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df):
        with patch("src.evaluation.live_runtime.publish_canonical_live_decision") as mock_pub_boundary:
            mock_pub_boundary.side_effect = publish_canonical_live_decision

            ref_stale = timestamps[-1].to_pydatetime() + pd.Timedelta(seconds=1000)
            res = runtime.run_once(publish=True, skip_if_no_trade=True, reference_now=ref_stale)

            assert res["decision"] == "NO TRADE"
            assert res["current_lifecycle_state"] == "PERSISTED"  # Remains PERSISTED because delivery skipped!


# C. Publisher disabled: Result is SKIPPED_DISABLED and canonical state remains PERSISTED
def test_publisher_disabled_leaves_state_persisted(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    publisher = Project2Publisher(enabled=False)

    published_cld, publication, pub_res = publish_canonical_live_decision(
        canonical_decision=cld,
        publisher=publisher,
        candidate=candidate,
        path=tmp_path / "pub.json",
        delivery_path=tmp_path / "del.json",
    )

    assert published_cld.current_state == LiveDecisionLifecycleState.PERSISTED
    assert pub_res["delivery_status"] == "SKIPPED_DISABLED"

    receipts = load_delivery_history(tmp_path / "del.json")
    assert receipts[0].delivery_status == DeliveryStatus.SKIPPED_DISABLED


# D. No-trade skip: When skip_if_no_trade=True, delivery is explicitly recorded as SKIPPED_NO_TRADE and state remains PERSISTED
def test_skip_if_no_trade_records_skipped_no_trade_and_remains_persisted(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt, direction=Direction.NO_TRADE)

    publisher = Project2Publisher(
        publish_url="https://api.example.com/signals",
        api_key="valid_key",
        enabled=True,
    )

    published_cld, publication, pub_res = publish_canonical_live_decision(
        canonical_decision=cld,
        publisher=publisher,
        candidate=candidate,
        path=tmp_path / "pub.json",
        delivery_path=tmp_path / "del.json",
        skip_if_no_trade=True,
    )

    assert published_cld.current_state == LiveDecisionLifecycleState.PERSISTED
    assert pub_res["delivery_status"] == "SKIPPED_NO_TRADE"


# E. Publisher rejection: REJECTED leaves state PERSISTED
def test_publisher_rejection_leaves_state_persisted(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "REJECTED",
        "published": False,
        "error": "Gateway rejected payload schema",
    }

    published_cld, publication, pub_res = publish_canonical_live_decision(
        canonical_decision=cld,
        publisher=mock_publisher,
        candidate=candidate,
        path=tmp_path / "pub.json",
        delivery_path=tmp_path / "del.json",
    )

    assert published_cld.current_state == LiveDecisionLifecycleState.PERSISTED
    assert pub_res["delivery_status"] == "REJECTED"


# F. Retryable failure: FAILED_RETRYABLE leaves state PERSISTED
def test_retryable_failure_leaves_state_persisted_and_enables_retry(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {
        "status": "TIMED_OUT",
        "published": False,
        "error": "HTTP timeout",
    }

    del_path = tmp_path / "del.json"

    published_cld, publication, pub_res = publish_canonical_live_decision(
        canonical_decision=cld,
        publisher=mock_publisher,
        candidate=candidate,
        path=tmp_path / "pub.json",
        delivery_path=del_path,
    )

    assert published_cld.current_state == LiveDecisionLifecycleState.PERSISTED
    assert pub_res["delivery_status"] == "FAILED_RETRYABLE"

    receipts = load_delivery_history(del_path)
    assert receipts[0].delivery_status == DeliveryStatus.FAILED_RETRYABLE
    assert receipts[0].attempt_count == 1


# G. Successful retry: First attempt fails retryably, second attempt succeeds
def test_successful_retry_progresses_to_delivered_and_published(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    del_path = tmp_path / "del.json"
    pub_path = tmp_path / "pub.json"

    # Attempt 1: Fails retryably
    mock_pub1 = MagicMock()
    mock_pub1.publish.return_value = {"status": "UNAVAILABLE", "published": False, "error": "Connection reset"}

    res_cld1, _, res1 = publish_canonical_live_decision(
        cld, publisher=mock_pub1, candidate=candidate, path=pub_path, delivery_path=del_path
    )
    assert res_cld1.current_state == LiveDecisionLifecycleState.PERSISTED

    # Attempt 2: Succeeds
    mock_pub2 = MagicMock()
    mock_pub2.publish.return_value = {"status": "PUBLISHED", "published": True, "http_code": 200, "event_id": "evt_456"}

    res_cld2, _, res2 = publish_canonical_live_decision(
        cld, publisher=mock_pub2, candidate=candidate, path=pub_path, delivery_path=del_path
    )
    assert res_cld2.current_state == LiveDecisionLifecycleState.PUBLISHED
    assert res2["delivery_status"] == "DELIVERED"

    receipts = load_delivery_history(del_path)
    assert receipts[0].delivery_status == DeliveryStatus.DELIVERED
    assert receipts[0].attempt_count == 2


# H. Already delivered: A second invocation for an already-delivered publication does NOT invoke publisher again
def test_already_delivered_does_not_call_publisher_again(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {"status": "PUBLISHED", "published": True, "http_code": 200, "event_id": "evt_789"}

    del_path = tmp_path / "del.json"
    pub_path = tmp_path / "pub.json"

    # Invocation 1
    publish_canonical_live_decision(cld, mock_publisher, candidate, pub_path, delivery_path=del_path)
    assert mock_publisher.publish.call_count == 1

    # Invocation 2
    res_cld2, _, res2 = publish_canonical_live_decision(cld, mock_publisher, candidate, pub_path, delivery_path=del_path)
    assert mock_publisher.publish.call_count == 1  # Publisher MUST NOT BE CALLED AGAIN!
    assert res_cld2.current_state == LiveDecisionLifecycleState.PUBLISHED


# I. Conflicting replay: Same publication identity but different canonical fingerprint fails closed
def test_conflicting_replay_fails_closed(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    receipt = PublicationDeliveryReceipt(
        publication_id="pub_conflict_test",
        decision_id=cld.decision.decision_id,
        signal_id=cld.signal.signal_id,
        canonical_live_decision_fingerprint=cld.canonical_live_decision_fingerprint,
        runtime_authorization_fingerprint=auth_receipt.authorization_fingerprint,
        candidate_id=candidate.candidate_id,
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        delivery_status=DeliveryStatus.DELIVERED,
        attempt_count=1,
        first_attempt_at_utc="2025-01-01T10:00:00+00:00",
        last_attempt_at_utc="2025-01-01T10:00:00+00:00",
    )

    del_path = tmp_path / "del.json"
    append_delivery_receipt(receipt, del_path)

    # Construct conflicting receipt with different runtime_authorization_fingerprint
    conflicting_receipt = PublicationDeliveryReceipt(
        publication_id="pub_conflict_test",
        decision_id=cld.decision.decision_id,
        signal_id=cld.signal.signal_id,
        canonical_live_decision_fingerprint=cld.canonical_live_decision_fingerprint,
        runtime_authorization_fingerprint="DIFF_AUTH_FP",
        candidate_id=candidate.candidate_id,
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        delivery_status=DeliveryStatus.DELIVERED,
        attempt_count=1,
        first_attempt_at_utc="2025-01-01T10:00:00+00:00",
        last_attempt_at_utc="2025-01-01T10:00:00+00:00",
    )

    with pytest.raises(DeliveryIntegrityError, match="Conflicting delivery receipt detected"):
        append_delivery_receipt(conflicting_receipt, del_path)


# J. Receipt fingerprint determinism
def test_delivery_receipt_fingerprint_determinism():
    kwargs = dict(
        publication_id="pub_123",
        decision_id="dec_123",
        signal_id="sig_123",
        canonical_live_decision_fingerprint="cld_fp",
        runtime_authorization_fingerprint="auth_fp",
        candidate_id="cand_123",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        delivery_status="DELIVERED",
        attempt_count=1,
        first_attempt_at_utc="2025-01-01T10:00:00+00:00",
        last_attempt_at_utc="2025-01-01T10:00:00+00:00",
    )

    fp1 = compute_delivery_receipt_fingerprint(**kwargs)
    fp2 = compute_delivery_receipt_fingerprint(**kwargs)
    assert fp1 == fp2

    kwargs_diff = dict(kwargs)
    kwargs_diff["attempt_count"] = 2
    fp_diff = compute_delivery_receipt_fingerprint(**kwargs_diff)
    assert fp1 != fp_diff


# K. No secrets: Persisted delivery records and fingerprints must not contain API keys or secrets
def test_no_secrets_in_delivery_receipts(tmp_path):
    secret_key = "SECRET_API_KEY_12345"
    receipt = PublicationDeliveryReceipt(
        publication_id="pub_secret_test",
        decision_id="dec_123",
        signal_id="sig_123",
        canonical_live_decision_fingerprint="cld_fp",
        runtime_authorization_fingerprint="auth_fp",
        candidate_id="cand_123",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        delivery_status=DeliveryStatus.FAILED_PERMANENT,
        attempt_count=1,
        first_attempt_at_utc="2025-01-01T10:00:00+00:00",
        last_attempt_at_utc="2025-01-01T10:00:00+00:00",
        error=f"Error with secret X-API-Key={secret_key}",
    )

    del_path = tmp_path / "del.json"
    append_delivery_receipt(receipt, del_path)

    raw_text = del_path.read_text(encoding="utf-8")
    assert secret_key not in raw_text
    assert "[REDACTED_AUTHORIZATION_SECRET]" in raw_text


# L. Lifecycle truthfulness: publisher call != PUBLISHED, validated delivery -> PUBLISHED
def test_lifecycle_truthfulness_publisher_call_does_not_equal_published(tmp_path):
    candidate, auth_receipt = create_test_candidate_and_receipt(tmp_path)
    cld = create_persisted_cld(candidate, auth_receipt)

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {"status": "TIMED_OUT", "published": False}

    res_cld, _, pub_res = publish_canonical_live_decision(
        cld, mock_publisher, candidate, tmp_path / "pub.json", delivery_path=tmp_path / "del.json"
    )

    assert mock_publisher.publish.called
    assert res_cld.current_state != LiveDecisionLifecycleState.PUBLISHED
    assert res_cld.current_state == LiveDecisionLifecycleState.PERSISTED


# M. No alternate publication path: Monkeypatch publisher and verify fresh and stale execution paths reach same boundary
def test_fresh_and_stale_paths_both_reach_same_canonical_publication_function(tmp_path):
    candidate_fresh, _ = create_test_candidate_and_receipt(tmp_path, candidate_id="cand_fresh")
    candidate_stale, _ = create_test_candidate_and_receipt(tmp_path, candidate_id="cand_stale")

    timestamps = pd.date_range("2025-01-01 10:00", periods=100, freq="5min", tz="UTC")
    prices = [2000.0 + (i * 2.0) for i in range(100)]
    df = pd.DataFrame({
        "openTime": timestamps,
        "open": [p - 1.0 for p in prices],
        "high": [p + 3.0 for p in prices],
        "low": [p - 2.0 for p in prices],
        "close": prices,
        "timestamp": timestamps,
    })

    mock_publisher = MagicMock()
    mock_publisher.publish.return_value = {"status": "PUBLISHED", "published": True, "http_code": 200, "event_id": "evt"}

    config_fresh = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id=candidate_fresh.candidate_id,
        strategy_id="momentum",
        research_dir=tmp_path,
    )
    config_stale = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="5m",
        candidate_id=candidate_stale.candidate_id,
        strategy_id="momentum",
        research_dir=tmp_path,
    )

    # 1. Fresh path
    runtime_fresh = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=tmp_path / "live" / "store_fresh.json",
        snapshot_path=tmp_path / "live" / "snap_fresh.json",
        research_dir=tmp_path,
        production_config=config_fresh,
    )

    # 2. Stale path
    runtime_stale = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="5m",
        publisher=mock_publisher,
        store_path=tmp_path / "live" / "store_stale.json",
        snapshot_path=tmp_path / "live" / "snap_stale.json",
        research_dir=tmp_path,
        production_config=config_stale,
    )

    with patch("src.evaluation.live_execution_runtime.load_live_market_data", return_value=df):
        with patch("src.evaluation.live_runtime.publish_canonical_live_decision") as mock_runtime_pub:
            mock_runtime_pub.side_effect = publish_canonical_live_decision

            ref_fresh = timestamps[-1].to_pydatetime()
            runtime_fresh.run_once(publish=True, reference_now=ref_fresh)
            assert mock_runtime_pub.call_count == 1

            ref_stale = timestamps[-1].to_pydatetime() + pd.Timedelta(seconds=1200)
            runtime_stale.run_once(publish=True, reference_now=ref_stale)
            assert mock_runtime_pub.call_count == 2
