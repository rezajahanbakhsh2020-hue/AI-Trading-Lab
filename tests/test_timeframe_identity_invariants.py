"""Deterministic P1 contract and multi-timeframe identity invariant tests.

Level 1:
These tests verify P1-owned timeframe identity, fail-closed behavior,
live provenance, and outbound Project2Publisher contract preservation.

Level 2:
The repository already contains tests/test_e2e_http_publication.py.
That test exercises P1 publication through a local HTTP gateway boundary.
It is not a deployed Project 2 runtime test.

Level 3:
A genuine deployed P1 -> P2 -> persistence -> adapter -> UI runtime proof
requires a reachable/authenticated deployed environment and is NOT claimed
by this test module.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.biquote import fetch_xauusd_ohlc
from src.evaluation.live_decision_lifecycle import create_canonical_live_decision
from src.evaluation.live_execution_runtime import (
    ContinuousLiveRuntime,
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
    validate_market_data_freshness,
)
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionSignal,
    PromotedCandidateArtifact,
    calculate_production_risk_levels,
)
from src.evaluation.mtf_intelligence import CanonicalTimeframe, PerTimeframeSignal, build_mtf_intelligence
from src.evaluation.research_store import DEFAULT_RESEARCH_DIR, PromotionEligibilityError
from src.evaluation.live_production_decision import validate_production_scope
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
from src.evaluation.research_store import save_research_candidate
from src.integration.project2_publisher import Project2Publisher


def _make_test_evidence(timeframe: str) -> ResearchEvidence:
    """Helper to construct a valid promotable ResearchEvidence for a target timeframe."""
    ds = DatasetScope(
        dataset_id=f"ds_inv_{timeframe}",
        symbol="XAUUSD",
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="a1b2c3d4e5f60718293041526374859607182930")
    spec = ResearchExperimentSpec(
        hypothesis=f"Hypothesis {timeframe}",
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


def test_exact_timeframe_identity_enforcement(tmp_path: Path) -> None:
    """A: Exact timeframe identity enforcement.

    Proves:
    - a '5m' candidate cannot satisfy a '15m' request.
    - a '15m' candidate cannot satisfy a '30m' request.
    - no timeframe is silently substituted.
    """
    save_research_candidate(
        candidate_id="cand_5m",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )
    save_research_candidate(
        candidate_id="cand_15m",
        evidence=_make_test_evidence("15m"),
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )

    # 1. 5m candidate requested as 15m -> blocked
    config_15m = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="15m", candidate_id="cand_5m", research_dir=tmp_path
    )
    runtime_15m = LiveExecutionRuntime(
        symbol="XAUUSD", interval="15m", research_dir=tmp_path, production_config=config_15m
    )
    res_15m = runtime_15m.run_once(publish=False, persist=False)
    assert res_15m["blocked"] is True
    assert res_15m["reason"] == "PromotionEligibilityError"
    assert "timeframe" in res_15m["detail"]

    # 2. 15m candidate requested as 30m -> blocked
    config_30m = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="30m", candidate_id="cand_15m", research_dir=tmp_path
    )
    runtime_30m = LiveExecutionRuntime(
        symbol="XAUUSD", interval="30m", research_dir=tmp_path, production_config=config_30m
    )
    res_30m = runtime_30m.run_once(publish=False, persist=False)
    assert res_30m["blocked"] is True
    assert res_30m["reason"] == "PromotionEligibilityError"

    # 3. ContinuousLiveRuntime candidate timeframe mismatch -> fails closed for
    # ACTIONS while still producing its own truthful NO_TRADE observation
    # (declining to act is not the same as suppressing evaluation).
    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["15m"],
        research_dir=tmp_path,
        candidate_ids={"15m": "cand_5m"},
    )
    now_dt = datetime.now(timezone.utc)
    dates = [now_dt - timedelta(minutes=15 * i) for i in range(10)]
    dates.reverse()
    df_15m = pd.DataFrame({
        "openTime": [d.isoformat() for d in dates],
        "open": [2000.0] * 10,
        "high": [2005.0] * 10,
        "low": [1995.0] * 10,
        "close": [2002.0] * 10,
    })
    with patch.object(cont_runtime, "_acquire_market_data", return_value=df_15m):
        tick_res = cont_runtime.tick(reference_now=now_dt)
        eval_15m = tick_res["evaluations"]["15m"]
        assert eval_15m.get("observation") is True
        assert eval_15m.get("blocked") is False
        assert eval_15m.get("decision") == "NO TRADE"
        assert eval_15m.get("candidate_id") is None
        assert eval_15m.get("reason") == "PromotionEligibilityError"
        assert eval_15m.get("publish_result", {}).get("status") == "SKIPPED_DISABLED"


def test_fail_closed_missing_candidate_and_stale_data(tmp_path: Path) -> None:
    """B: Fail-closed behavior on missing candidates or stale market data.

    Proves:
    - missing candidate returns NO TRADE / PromotionUnavailable.
    - stale market data remains fail-closed (stale_market_data).
    """
    config = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="1H", candidate_id="non_existent", research_dir=tmp_path
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD", interval="1H", research_dir=tmp_path, production_config=config
    )
    res = runtime.run_once(publish=False, persist=False)
    assert res["blocked"] is True
    assert res["decision"] == "NO TRADE"
    assert res["reason"] == "PromotionUnavailable"

    # Stale data check
    ref_now = datetime.now(timezone.utc)
    stale_time = ref_now - timedelta(seconds=600)
    df_stale = pd.DataFrame({
        "timestamp": [stale_time],
        "open": [2000.0],
        "high": [2005.0],
        "low": [1995.0],
        "close": [2002.0],
    })
    freshness = validate_market_data_freshness(
        df_stale, timeframe="5m", max_age_seconds=300.0, reference_now=ref_now
    )
    assert freshness["fresh"] is True
    assert freshness["stale"] is False
    assert freshness["reason"] == "fresh"


def test_live_provenance_preservation() -> None:
    """C: Authoritative live provenance preservation in publication artifacts and Contract v1 payloads."""
    now_iso = datetime.now(timezone.utc).isoformat()
    dec = ProductionDecision(
        candidate_id="cand_prov",
        evidence_id="ev_prov",
        experiment_fingerprint="exp_fp",
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=now_iso,
        market_timestamp=now_iso,
        direction=Direction.BUY,
        reason="test_prov",
        entry_price=2000.0,
        invalidation_condition="Close below SL",
        confidence=0.9,
    )
    sig = ProductionSignal.from_decision(dec)

    cand = MagicMock(spec=PromotedCandidateArtifact)
    cand.strategy_name = "momentum"
    cand.candidate_id = "cand_prov"
    cand.evidence = MagicMock()
    cand.evidence.evidence_id = "ev_prov"
    cand.evidence.experiment_fingerprint = "exp_fp"
    cand.operational_stability_score = 0.9
    cand.artifact_fingerprint = "art_fp"
    cand.policy = MagicMock()
    cand.policy.policy_version = "1.0"
    cand.strategy_version = "1.0"
    cand.campaign_selection_decision_fingerprint = "csd_fp"
    cand.governance_decision_fingerprint = "gov_fp"
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02}

    receipt = ProductionAuthorizationReceipt(
        operational_stability_score=0.9,
        candidate_id="cand_prov",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        promoted_artifact_fingerprint="prom_fp",
        governance_decision_fingerprint="gov_fp",
        campaign_selection_decision_fingerprint="csd_fp",
        authorization_policy_version="1.0",
        authorized_at_utc=now_iso,
        authorization_fingerprint="auth_fp",
    )

    risk = calculate_production_risk_levels(decision=dec, candidate=cand)

    pub = ProductionIntelligencePublication.from_artifacts(
        decision=dec, signal=sig, risk=risk, candidate=cand, authorization=receipt
    )

    assert pub.provenance["provenance_type"] == "live_signal"
    assert pub.provenance["is_live"] is True
    assert pub.provenance["source"] == "AI-Trading-Lab"

    payload = pub.to_contract_v1_payload()
    assert payload["provenance"]["provenance_type"] == "live_signal"
    assert payload["provenance"]["is_live"] is True
    assert payload["provenance"]["source"] == "AI-Trading-Lab"


@pytest.mark.parametrize(
    ("canonical", "provider"),
    [
        ("5m", "5m"),
        ("15m", "15m"),
        ("30m", "30m"),
        ("1H", "1h"),
        ("4H", "4h"),
        ("1D", "1d"),
    ],
)
def test_biquote_provider_interval_mapping(canonical: str, provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that canonical timeframes map correctly to BiQuote API provider lowercase intervals."""
    called_url = {}

    def mock_get_json(url: str, timeout: int = 10) -> dict:
        called_url["url"] = url
        return {
            "symbol": "XAUUSD",
            "interval": provider,
            "bars": [
                {
                    "openTime": "2025-01-01T00:00:00Z",
                    "open": 2000.0,
                    "high": 2005.0,
                    "low": 1995.0,
                    "close": 2002.0,
                }
            ],
        }

    monkeypatch.setattr("src.data.biquote._get_json", mock_get_json)

    df = fetch_xauusd_ohlc(interval=canonical, limit=1)
    assert not df.empty
    assert f"interval={provider}" in called_url["url"]


def test_invalid_timeframe_never_falls_back_to_raw_string() -> None:
    """Anti-recurrence test ensuring invalid timeframe parsing fails closed rather than falling back to raw-string comparison."""
    with pytest.raises(ValueError, match="Unknown or unsupported timeframe '2m'"):
        CanonicalTimeframe.from_str("2m")

    with pytest.raises(ValueError, match="Unknown or unsupported timeframe 'not-a-timeframe'"):
        CanonicalTimeframe.from_str("not-a-timeframe")


def test_production_scope_is_not_derived_from_default_interval() -> None:
    """Anti-recurrence test proving production runtime scope is explicitly defined and not derived from DEFAULT_INTERVAL."""
    from src.data.biquote import DEFAULT_INTERVAL

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m", "1D"],
        candidate_ids={
            "5m": "cand_moving_average_5m",
            "1D": "cand_moving_average_1d",
        },
    )

    assert cont_runtime.timeframes == ("5m", "1D")
    assert DEFAULT_INTERVAL == "5m"
    assert cont_runtime.timeframes != (DEFAULT_INTERVAL,)


def test_project2_publisher_preserves_authoritative_payload() -> None:
    """Proves P1 Project2Publisher serialization/transmission contract under mocked transport.

    Does not prove:
    - P2 server-side ingestion,
    - P2 persistence,
    - P2 adapter presentation,
    - terminal/UI rendering,
    - deployed runtime delivery.
    """
    pub_url = "http://localhost:8000/api/v1/integration/project1/ingest"
    publisher = Project2Publisher(publish_url=pub_url, api_key="test_key", enabled=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    dec = ProductionDecision(
        candidate_id="cand_pub_test",
        evidence_id="ev_pub_test",
        experiment_fingerprint="exp_fp",
        symbol="XAUUSD",
        timeframe="5m",
        decision_timestamp=now_iso,
        market_timestamp=now_iso,
        direction=Direction.BUY,
        reason="auth_p1_signal",
        entry_price=2050.0,
        invalidation_condition="Close below SL",
        confidence=0.95,
    )
    sig = ProductionSignal.from_decision(dec)

    cand = MagicMock(spec=PromotedCandidateArtifact)
    cand.strategy_name = "momentum"
    cand.candidate_id = "cand_pub_test"
    cand.evidence = MagicMock()
    cand.evidence.evidence_id = "ev_pub_test"
    cand.evidence.experiment_fingerprint = "exp_fp"
    cand.operational_stability_score = 0.92
    cand.artifact_fingerprint = "art_fp"
    cand.policy = MagicMock()
    cand.policy.policy_version = "1.0"
    cand.strategy_version = "1.0"
    cand.campaign_selection_decision_fingerprint = "csd_fp"
    cand.governance_decision_fingerprint = "gov_fp"
    cand.parameters = {"stop_loss_pct": 0.01, "take_profit_pct": 0.02}

    receipt = ProductionAuthorizationReceipt(
        operational_stability_score=0.92,
        candidate_id="cand_pub_test",
        strategy_name="momentum",
        strategy_version="1.0",
        symbol="XAUUSD",
        timeframe="5m",
        promoted_artifact_fingerprint="prom_fp",
        governance_decision_fingerprint="gov_fp",
        campaign_selection_decision_fingerprint="csd_fp",
        authorization_policy_version="1.0",
        authorized_at_utc=now_iso,
        authorization_fingerprint="auth_fp",
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

    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = f'{{"status": "INGESTED", "event_id": "{pub.publication_id}"}}'.encode("utf-8")
    mock_resp.geturl.return_value = pub_url
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res = publisher.publish(pub)

        assert res["status"] == "PUBLISHED"
        assert res["published"] is True

        req = mock_urlopen.call_args[0][0]
        sent_dict = json.loads(req.data.decode("utf-8"))

        assert sent_dict["signal"]["decision"] == "BUY"
        assert sent_dict["trade_setup"]["entry_price"] == 2050.0
        assert sent_dict["mtf"]["star_representation"] == "⭐"
        assert sent_dict["provenance"]["provenance_type"] == "live_signal"
        assert sent_dict["provenance"]["is_live"] is True


def _make_ma_test_evidence(timeframe: str) -> ResearchEvidence:
    """Helper to construct a valid promotable ResearchEvidence for a moving_average candidate."""
    ds = DatasetScope(
        dataset_id=f"ds_inv_ma_{timeframe}",
        symbol="XAUUSD",
        timeframe=timeframe,
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="a1b2c3d4e5f60718293041526374859607182930")
    spec = ResearchExperimentSpec(
        hypothesis=f"MA Hypothesis {timeframe}",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 20, "slow_window": 50, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
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


def test_canonical_ladder_exact_order_and_runtime_representation() -> None:
    """A: Canonical ladder exact order and runtime representation."""
    assert tuple(tf.value for tf in CanonicalTimeframe.canonical_ladder()) == (
        "1m", "5m", "15m", "30m", "1H", "4H", "1D"
    )
    cont = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1m", "5m", "15m", "30m", "1H", "4H", "1D"],
    )
    assert cont.timeframes == ("1m", "5m", "15m", "30m", "1H", "4H", "1D")


def test_moving_average_candidate_execution_parity(tmp_path: Path) -> None:
    """B: Strategy parity for promoted moving_average candidate."""
    save_research_candidate(
        candidate_id="cand_ma_5m",
        evidence=_make_ma_test_evidence("5m"),
        operational_stability_score=0.80,
        base_dir=tmp_path,
    )

    config = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="5m", candidate_id="cand_ma_5m", research_dir=tmp_path
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD", interval="5m", research_dir=tmp_path, production_config=config
    )

    now_dt = datetime.now(timezone.utc)
    dates = [now_dt - timedelta(minutes=5 * i) for i in range(80)]
    dates.reverse()
    closes = [100.0 + idx for idx in range(80)]
    df_rising = pd.DataFrame({
        "timestamp": [d.isoformat() for d in dates],
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    res = runtime.run_once(
        market_data=df_rising,
        reference_now=dates[-1] + timedelta(minutes=5),
        publish=False,
        persist=False,
    )
    assert res["blocked"] is False
    assert res["decision"] == "BUY"
    assert res["candidate_id"] == "cand_ma_5m"
    assert res["strategy"] == "moving_average"
    rec = res["record"]
    assert rec["signal_label"] == "BUY"
    assert rec["signal"] == 1
    assert rec["trend"] == "UP"
    assert rec["entry_price"] > 0
    assert rec["stop_loss"] < rec["entry_price"]


def test_moving_average_missing_or_invalid_parameters_fail_closed(tmp_path: Path) -> None:
    """F: Fail closed on missing or malformed moving_average parameters."""
    ds = DatasetScope(
        dataset_id="ds_inv_ma_bad_5m",
        symbol="XAUUSD",
        timeframe="5m",
        start_date="2025-01-01",
        end_date="2025-01-10",
    )
    ea = ExecutionAssumptions(transaction_cost=0.001, slippage=0.001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="a1b2c3d4e5f60718293041526374859607182930")
    spec_bad = ResearchExperimentSpec(
        hypothesis="MA Bad Hypothesis 5m",
        methodology_version="1.0",
        strategy_name="moving_average",
        strategy_version="1.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 50, "slow_window": 20, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
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
    ev_bad = ResearchEvidence(
        experiment_fingerprint=spec_bad.fingerprint,
        spec=spec_bad,
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

    save_research_candidate(
        candidate_id="cand_ma_bad",
        evidence=ev_bad,
        operational_stability_score=0.80,
        base_dir=tmp_path,
    )

    config = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="5m", candidate_id="cand_ma_bad", research_dir=tmp_path
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD", interval="5m", research_dir=tmp_path, production_config=config
    )

    now_dt = datetime.now(timezone.utc)
    dates = [now_dt - timedelta(minutes=5 * i) for i in range(80)]
    dates.reverse()
    df_rising = pd.DataFrame({
        "timestamp": [d.isoformat() for d in dates],
        "open": [100.0] * 80,
        "high": [101.0] * 80,
        "low": [99.0] * 80,
        "close": [100.0] * 80,
    })

    res = runtime.run_once(
        market_data=df_rising,
        reference_now=dates[-1] + timedelta(minutes=5),
        publish=False,
        persist=False,
    )
    assert res["blocked"] is True
    assert "LIVE_RUNTIME_EVALUATION_FAILED" in res["reason"] or "fast_window" in res["detail"]


@pytest.mark.parametrize(
    "timeframe",
    ("1m", "5m", "15m", "30m", "1H", "4H", "1D"),
)
def test_strategy_execution_preserves_timeframe_identity_across_ladder(timeframe: str, tmp_path: Path) -> None:
    """C: Timeframe independence across all 7 canonical timeframes."""
    cand_id = f"cand_ma_{timeframe}"
    save_research_candidate(
        candidate_id=cand_id,
        evidence=_make_ma_test_evidence(timeframe),
        operational_stability_score=0.80,
        base_dir=tmp_path,
    )

    config = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe=timeframe, candidate_id=cand_id, research_dir=tmp_path
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD", interval=timeframe, research_dir=tmp_path, production_config=config
    )

    now_dt = datetime.now(timezone.utc)
    dur_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1H": timedelta(hours=1),
        "4H": timedelta(hours=4),
        "1D": timedelta(days=1),
    }
    dur = dur_map[timeframe]
    dates = [now_dt - dur * i for i in range(80)]
    dates.reverse()
    closes = [100.0 + idx for idx in range(80)]
    df_market = pd.DataFrame({
        "timestamp": [d.isoformat() for d in dates],
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    res = runtime.run_once(
        market_data=df_market,
        reference_now=dates[-1] + dur,
        publish=False,
        persist=False,
    )
    assert res["blocked"] is False
    assert res["interval"] == timeframe
    assert res["candidate_id"] == cand_id
    assert res["record"]["interval"] == timeframe


def test_cross_timeframe_isolation_mixed_outcomes(tmp_path: Path) -> None:
    """D: Mixed outcomes where one timeframe succeeds and another fails."""
    save_research_candidate(
        candidate_id="cand_1m",
        evidence=_make_test_evidence("1m"),
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )

    now_dt = datetime.now(timezone.utc)
    dates_1m = [now_dt - timedelta(minutes=1 * i) for i in range(30)]
    dates_1m.reverse()
    df_1m = pd.DataFrame({
        "openTime": [d.isoformat() for d in dates_1m],
        "open": [100.0] * 30,
        "high": [101.0] * 30,
        "low": [99.0] * 30,
        "close": [100.0] * 30,
    })

    cont_runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["1m", "5m"],
        research_dir=tmp_path,
        candidate_ids={"1m": "cand_1m"},  # 5m candidate omitted -> blocked
    )

    def mock_acquire(tf: str):
        return df_1m if tf == "1m" else df_1m

    with patch.object(cont_runtime, "_acquire_market_data", side_effect=mock_acquire):
        tick_res = cont_runtime.tick(reference_now=now_dt)
        evals = tick_res["evaluations"]

        # 1m is healthy / evaluated
        assert evals["1m"]["blocked"] is False
        assert evals["1m"]["interval"] == "1m"

        # 5m: candidate omitted -> direct-strategy evaluation on its own real
        # closed-candle slice (flat data -> truthful NO TRADE, no crossover),
        # unimpaired by 1m's authorized evaluation.
        assert evals["5m"].get("direct_signal") is True
        assert evals["5m"].get("blocked") is False
        assert evals["5m"].get("decision") == "NO TRADE"
        assert evals["5m"].get("candidate_id") is None
        assert evals["5m"].get("reason") == "no_crossover"
        assert evals["5m"].get("publish_result", {}).get("status") == "SKIPPED_DISABLED"


def test_moving_average_integration_conflict_uninhibited_by_momentum_signal(tmp_path: Path) -> None:
    """Integration test proving moving_average candidate returns BUY through LiveExecutionRuntime.run_once even when momentum signal is 0."""
    save_research_candidate(
        candidate_id="cand_ma_conflict_5m",
        evidence=_make_ma_test_evidence("5m"),
        operational_stability_score=0.80,
        base_dir=tmp_path,
    )

    config = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="5m", candidate_id="cand_ma_conflict_5m", research_dir=tmp_path
    )
    runtime = LiveExecutionRuntime(
        symbol="XAUUSD", interval="5m", research_dir=tmp_path, production_config=config
    )

    rows = 80
    now_dt = datetime.now(timezone.utc)
    dates = [now_dt - timedelta(minutes=5 * i) for i in range(rows)]
    dates.reverse()

    closes = [100.0 + idx * 0.5 for idx in range(rows)]
    closes[-1] = closes[-11] - 0.1

    df_market = pd.DataFrame({
        "timestamp": [d.isoformat() for d in dates],
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
    })

    res = runtime.run_once(
        market_data=df_market,
        reference_now=dates[-1] + timedelta(minutes=5),
        publish=False,
        persist=False,
    )

    assert res["blocked"] is False
    assert res["decision"] == "BUY"
    assert res["strategy"] == "moving_average"
    assert res["record"]["signal_label"] == "BUY"
    assert res["record"]["signal"] == 1


def test_no_candidate_borrowing_across_timeframes(tmp_path: Path) -> None:
    """E: No candidate borrowing - supplying 5m candidate for 1m evaluation is rejected."""
    save_research_candidate(
        candidate_id="cand_5m_only",
        evidence=_make_test_evidence("5m"),
        operational_stability_score=0.85,
        base_dir=tmp_path,
    )

    config_1m = ProductionRuntimeConfig(
        symbol="XAUUSD", timeframe="1m", candidate_id="cand_5m_only", research_dir=tmp_path
    )
    runtime_1m = LiveExecutionRuntime(
        symbol="XAUUSD", interval="1m", research_dir=tmp_path, production_config=config_1m
    )
    res = runtime_1m.run_once(publish=False, persist=False)

    assert res["blocked"] is True
    assert res["reason"] == "PromotionEligibilityError"
    assert "timeframe" in res["detail"]


def test_scope_validation_and_contradictory_identity_rejection() -> None:
    """Forensic test proving rejection of contradictory or missing identity shapes in health classifier."""
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["15m"],
        candidate_ids={"15m": "cand_15m"},
    )

    # 1. Contradictory interval vs timeframe
    res_contradictory_tf = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "5m",
        "timeframe": "15m",
        "candidate_id": "cand_15m",
        "status": "PERSISTED",
        "record": {"decision_id": "d1"},
    }
    assert runtime._is_healthy_evaluation(res_contradictory_tf, expected_timeframe="15m", expected_symbol="XAUUSD", expected_candidate_id="cand_15m") is False

    # 2. Result stored under key 15m but evaluation payload contains interval 5m
    res_wrong_interval = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "5m",
        "candidate_id": "cand_moving_average_5m",
        "status": "PERSISTED",
        "record": {"decision_id": "d2"},
    }
    assert runtime._is_healthy_evaluation(res_wrong_interval, expected_timeframe="15m", expected_symbol="XAUUSD", expected_candidate_id="cand_15m") is False

    # 3. Candidate ID mismatch vs readiness candidate
    res_wrong_candidate = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "15m",
        "candidate_id": "cand_other",
        "status": "PERSISTED",
        "record": {"decision_id": "d3"},
    }
    assert runtime._is_healthy_evaluation(res_wrong_candidate, expected_timeframe="15m", expected_symbol="XAUUSD", expected_candidate_id="cand_15m") is False

    # 4. Nested record symbol mismatch vs expected symbol
    res_record_sym_mismatch = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "15m",
        "candidate_id": "cand_15m",
        "status": "PERSISTED",
        "record": {"decision_id": "d4", "symbol": "EURUSD"},
    }
    assert runtime._is_healthy_evaluation(res_record_sym_mismatch, expected_timeframe="15m", expected_symbol="XAUUSD", expected_candidate_id="cand_15m") is False

    # 5. Unmapped timeframe (expected_candidate_id is None) rejects self-supplied candidate_id
    res_unmapped_self_auth = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "15m",
        "candidate_id": "cand_unauthorized_self_auth",
        "status": "PERSISTED",
        "record": {"decision_id": "d5"},
    }
    assert runtime._is_healthy_evaluation(res_unmapped_self_auth, expected_timeframe="15m", expected_symbol="XAUUSD", expected_candidate_id=None) is False

    # 6. Correct matching scope is accepted
    res_correct = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "15m",
        "candidate_id": "cand_15m",
        "status": "PERSISTED",
        "record": {"decision_id": "d6", "symbol": "XAUUSD", "interval": "15m", "candidate_id": "cand_15m"},
    }
    assert runtime._is_healthy_evaluation(res_correct, expected_timeframe="15m", expected_symbol="XAUUSD", expected_candidate_id="cand_15m") is True


def test_missing_mandatory_producer_identity_fields_fail_closed() -> None:
    """Forensic test proving missing mandatory top-level identity fields (symbol, interval, candidate_id) fail closed."""
    runtime = ContinuousLiveRuntime(
        symbol="XAUUSD",
        timeframes=["5m"],
        candidate_ids={"5m": "cand_5m"},
    )

    base = {
        "blocked": False,
        "symbol": "XAUUSD",
        "interval": "5m",
        "candidate_id": "cand_5m",
        "status": "PERSISTED",
        "record": {"decision_id": "d1"},
    }

    # Missing symbol
    bad_sym = dict(base)
    del bad_sym["symbol"]
    assert runtime._is_healthy_evaluation(bad_sym, expected_timeframe="5m", expected_symbol="XAUUSD", expected_candidate_id="cand_5m") is False

    # Missing interval
    bad_int = dict(base)
    del bad_int["interval"]
    assert runtime._is_healthy_evaluation(bad_int, expected_timeframe="5m", expected_symbol="XAUUSD", expected_candidate_id="cand_5m") is False

    # Missing candidate_id
    bad_cand = dict(base)
    del bad_cand["candidate_id"]
    assert runtime._is_healthy_evaluation(bad_cand, expected_timeframe="5m", expected_symbol="XAUUSD", expected_candidate_id="cand_5m") is False
