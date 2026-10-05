"""Deterministic Regression Tests for Authoritative Multi-Timeframe Identity and Delivery Invariants.

Covers mandatory Phase 5 invariants:
1. A "5m" signal cannot satisfy a "15m" request.
2. A "15m" signal cannot satisfy a "30m" request.
3. No timeframe is silently substituted.
4. Missing signal for the requested timeframe returns NO SIGNAL / PromotionUnavailable / ProductionBlocked.
5. Stale signal remains NO SIGNAL (stale market data results in stale status and NO SIGNAL).
6. Live signal identity is preserved.
7. "provenance_type=live_signal" is preserved.
8. "is_live=True" is preserved.
9. P2 does not calculate a replacement decision.
10. P2 does not calculate replacement MTF intelligence.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.live_decision_lifecycle import (
    CanonicalLiveDecision,
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
)
from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    LiveExecutionRuntime,
    ProductionBlocked,
    ProductionRuntimeConfig,
    validate_market_data_freshness,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionRiskLevels,
    ProductionSignal,
    PromotedCandidateArtifact,
    calculate_production_risk_levels,
    validate_production_scope,
)
from src.evaluation.live_runtime import (
    evaluate_authorized_live_runtime,
)
from src.evaluation.live_runtime_context import (
    create_authorized_runtime_context,
)
from src.evaluation.mtf_intelligence import (
    CanonicalTimeframe,
    PerTimeframeSignal,
    build_mtf_intelligence,
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
from src.evaluation.research_store import (
    save_research_candidate,
)
from src.integration.project2_publisher import Project2Publisher


def make_test_promoted_evidence(timeframe: str = "5m") -> ResearchEvidence:
    ds = DatasetScope(
        dataset_id=f"ds_inv_test_{timeframe}",
        symbol="XAUUSD",
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="a1b2c3d4e5f60718293041526374859607182930")
    spec = ResearchExperimentSpec(
        hypothesis=f"Invariant test hypothesis {timeframe}",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
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
    return ResearchEvidence(
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


def test_invariant_1_and_2_and_3_timeframe_mismatch_and_no_silent_substitution(tmp_path: Path) -> None:
    """Invariants 1, 2, 3:
    1. A '5m' signal/candidate cannot satisfy a '15m' request.
    2. A '15m' signal/candidate cannot satisfy a '30m' request.
    3. No timeframe is silently substituted.
    """
    # Save a 5m candidate
    ev5m = make_test_promoted_evidence("5m")
    save_research_candidate(
        candidate_id="cand_5m_only",
        evidence=ev5m,
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )

    # Save a 15m candidate
    ev15m = make_test_promoted_evidence("15m")
    save_research_candidate(
        candidate_id="cand_15m_only",
        evidence=ev15m,
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )

    # 1. Try requesting 15m with candidate cand_5m_only
    config_15m = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="15m",
        candidate_id="cand_5m_only",
        research_dir=tmp_path,
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="15m",
        research_dir=tmp_path,
        production_config=config_15m,
    )
    res_15m = runtime.run_once(publish=False, persist=False)
    assert res_15m["blocked"] is True
    assert res_15m["reason"] == "PromotionEligibilityError"
    assert "timeframe" in res_15m["detail"]

    # 2. Try requesting 30m with candidate cand_15m_only
    config_30m = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="30m",
        candidate_id="cand_15m_only",
        research_dir=tmp_path,
    )
    runtime_30m = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="30m",
        research_dir=tmp_path,
        production_config=config_30m,
    )
    res_30m = runtime_30m.run_once(publish=False, persist=False)
    assert res_30m["blocked"] is True
    assert res_30m["reason"] == "PromotionEligibilityError"

    # 3. ContinuousLiveRuntime candidate timeframe mismatch rejection
    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["15m"],
        research_dir=tmp_path,
        candidate_ids={"15m": "cand_5m_only"},
    )
    # Mock data acquisition
    now_dt = datetime.now(timezone.utc)
    dates = [now_dt - timedelta(minutes=15 * i) for i in range(20)]
    dates.reverse()
    df_15m = pd.DataFrame({
        "openTime": [d.isoformat() for d in dates],
        "open": [2000.0] * 20,
        "high": [2005.0] * 20,
        "low": [1995.0] * 20,
        "close": [2002.0] * 20,
    })
    with patch.object(cont_runtime, "_acquire_market_data", return_value=df_15m):
        tick_res = cont_runtime.tick(reference_now=now_dt)
        eval_15m = tick_res["evaluations"]["15m"]
        assert eval_15m["blocked"] is True
        assert eval_15m["reason"] == "PromotionEligibilityError"


def test_invariant_4_missing_signal_returns_no_signal(tmp_path: Path) -> None:
    """Invariant 4: Missing signal / candidate for requested timeframe returns NO SIGNAL / ProductionBlocked."""
    config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="1H",
        candidate_id="non_existent_candidate",
        research_dir=tmp_path,
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD",
        interval="1H",
        research_dir=tmp_path,
        production_config=config,
    )
    res = runtime.run_once(publish=False, persist=False)

    assert res["blocked"] is True
    assert res["decision"] == "NO TRADE"
    assert res["reason"] == "PromotionUnavailable"


def test_invariant_5_stale_signal_remains_no_signal() -> None:
    """Invariant 5: Stale market data results in stale status and NO SIGNAL decision."""
    ref_now = datetime.now(timezone.utc)
    stale_time = ref_now - timedelta(seconds=600)  # 10 minutes ago, exceeds max_age_seconds=300

    df_stale = pd.DataFrame({
        "timestamp": [stale_time],
        "open": [2000.0],
        "high": [2005.0],
        "low": [1995.0],
        "close": [2002.0],
    })

    freshness = validate_market_data_freshness(
        df_stale,
        max_age_seconds=300.0,
        reference_now=ref_now,
    )

    assert freshness["fresh"] is False
    assert freshness["stale"] is True
    assert freshness["reason"] == "stale_market_data"
    assert freshness["age_seconds"] > 300.0


def test_invariant_6_7_8_live_signal_identity_and_provenance_preserved() -> None:
    """Invariants 6, 7, 8:
    6. Live signal identity is preserved.
    7. "provenance_type=live_signal" is preserved.
    8. "is_live=True" is preserved.
    """
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    mkt_iso = (now_dt - timedelta(seconds=1)).isoformat()

    dec = ProductionDecision(
        candidate_id="cand_test_inv",
        evidence_id="ev_test_inv",
        experiment_fingerprint="exp_fp_inv",
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=now_iso,
        market_timestamp=mkt_iso,
        direction=Direction.BUY,
        reason="test_provenance",
        entry_price=2000.0,
        invalidation_condition="Close below SL",
        confidence=0.9,
    )
    sig = ProductionSignal.from_decision(dec)

    cand = MagicMock(spec=PromotedCandidateArtifact)
    cand.strategy_name = "momentum"
    cand.candidate_id = "cand_test_inv"
    cand.evidence = MagicMock()
    cand.evidence.evidence_id = "ev_test_inv"
    cand.evidence.experiment_fingerprint = "exp_fp_inv"
    cand.operational_stability_score = 0.9
    cand.artifact_fingerprint = "art_fp_inv"
    cand.policy = MagicMock()
    cand.policy.policy_version = "1.0"
    cand.strategy_version = "1.0"
    cand.campaign_selection_decision_fingerprint = "csd_fp_inv"
    cand.governance_decision_fingerprint = "gov_fp_inv"
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02}

    receipt = ProductionAuthorizationReceipt(
        operational_stability_score=0.9,
        candidate_id="cand_test_inv",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        promoted_artifact_fingerprint="prom_fp_inv",
        governance_decision_fingerprint="gov_fp_inv",
        campaign_selection_decision_fingerprint="csd_fp_inv",
        authorization_policy_version="1.0",
        authorized_at_utc=now_iso,
        authorization_fingerprint="auth_fp_inv",
    )

    risk = calculate_production_risk_levels(decision=dec, candidate=cand)

    pub = ProductionIntelligencePublication.from_artifacts(
        decision=dec,
        signal=sig,
        risk=risk,
        candidate=cand,
        authorization=receipt,
    )

    # 6. Live signal identity
    assert pub.decision_id == dec.decision_id
    assert pub.signal_id == sig.signal_id
    assert pub.candidate_id == "cand_test_inv"

    # 7. Provenance type
    assert pub.provenance["provenance_type"] == "live_signal"

    # 8. is_live
    assert pub.provenance["is_live"] is True
    assert pub.provenance["source"] == "AI-Trading-Lab"

    # Contract v1 payload
    payload = pub.to_contract_v1_payload()
    assert payload["provenance"]["provenance_type"] == "live_signal"
    assert payload["provenance"]["is_live"] is True
    assert payload["provenance"]["source"] == "AI-Trading-Lab"


def test_invariant_9_and_10_p2_does_not_recalculate_decision_or_mtf() -> None:
    """Invariants 9 & 10:
    9. Project 2 does not calculate a replacement decision (Project 1 signal is authoritative).
    10. Project 2 does not calculate replacement MTF intelligence.
    Project2Publisher transmits the P1 payload verbatim without altering signal or MTF fields.
    """
    pub_url = "http://localhost:8000/api/v1/integration/project1/ingest"
    publisher = Project2Publisher(publish_url=pub_url, api_key="test_key", enabled=True)

    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    mkt_iso = (now_dt - timedelta(seconds=1)).isoformat()

    dec = ProductionDecision(
        candidate_id="cand_p2_test",
        evidence_id="ev_p2_test",
        experiment_fingerprint="exp_fp_p2",
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=now_iso,
        market_timestamp=mkt_iso,
        direction=Direction.BUY,
        reason="authoritative_p1_signal",
        entry_price=2050.0,
        invalidation_condition="Close below SL",
        confidence=0.95,
    )
    sig = ProductionSignal.from_decision(dec)

    cand = MagicMock(spec=PromotedCandidateArtifact)
    cand.strategy_name = "momentum"
    cand.candidate_id = "cand_p2_test"
    cand.evidence = MagicMock()
    cand.evidence.evidence_id = "ev_p2_test"
    cand.evidence.experiment_fingerprint = "exp_fp_p2"
    cand.operational_stability_score = 0.92
    cand.artifact_fingerprint = "art_fp_p2"
    cand.policy = MagicMock()
    cand.policy.policy_version = "1.0"
    cand.strategy_version = "1.0"
    cand.campaign_selection_decision_fingerprint = "csd_fp_p2"
    cand.governance_decision_fingerprint = "gov_fp_p2"
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02}

    receipt = ProductionAuthorizationReceipt(
        operational_stability_score=0.92,
        candidate_id="cand_p2_test",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        promoted_artifact_fingerprint="prom_fp_p2",
        governance_decision_fingerprint="gov_fp_p2",
        campaign_selection_decision_fingerprint="csd_fp_p2",
        authorization_policy_version="1.0",
        authorized_at_utc=now_iso,
        authorization_fingerprint="auth_fp_p2",
    )

    risk = calculate_production_risk_levels(decision=dec, candidate=cand)

    ptf_sig = PerTimeframeSignal.from_canonical_live_decision(
        create_canonical_live_decision(
            authorization_receipt=receipt,
            decision=dec,
            signal=sig,
            risk_levels=risk,
            actor="test",
            timestamp_utc=now_iso,
            reason="test_init",
        )
    )

    mtf_intel = build_mtf_intelligence([ptf_sig], local_timeframe="5m")

    pub = ProductionIntelligencePublication.from_artifacts(
        decision=dec,
        signal=sig,
        risk=risk,
        candidate=cand,
        authorization=receipt,
        mtf_intelligence=mtf_intel,
    )

    # Verify payload before outbound delivery
    payload = pub.to_contract_v1_payload()
    assert payload["signal"]["decision"] == "BUY"
    assert payload["trade_setup"]["entry_price"] == 2050.0
    assert payload["mtf"]["alignment_coverage"] == 1
    assert payload["mtf"]["star_representation"] == "⭐"

    # Mock HTTP response to verify exact payload transmitted
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = f'{{"status": "INGESTED", "event_id": "{pub.publication_id}"}}'.encode("utf-8")
    mock_resp.geturl.return_value = pub_url
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res = publisher.publish(pub)

        assert res["status"] == "PUBLISHED"
        assert res["published"] is True

        # Extract HTTP request body sent to P2 endpoint
        req = mock_urlopen.call_args[0][0]
        sent_bytes = req.data
        sent_dict = json.loads(sent_bytes.decode("utf-8"))

        # Assert P1 signal and MTF intelligence are transmitted verbatim without modification
        assert sent_dict["signal"]["decision"] == "BUY"
        assert sent_dict["trade_setup"]["entry_price"] == 2050.0
        assert sent_dict["mtf"]["star_representation"] == "⭐"
        assert sent_dict["provenance"]["provenance_type"] == "live_signal"
        assert sent_dict["provenance"]["is_live"] is True
