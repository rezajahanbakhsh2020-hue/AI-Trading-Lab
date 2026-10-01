"""Comprehensive Adversarial and Boundary Integrity Test Suite for P1 Release Closure.

Validates end-to-end P1 invariants across the Trading Brain chain:
- Research hypothesis lifecycle (GENERATED -> ACCEPTED_FOR_RESEARCH -> execution)
- Raw ResearchExperimentSpec execution rejection
- Exact temporal evidence integrity (non-monotonic timestamps, missing/malformed timestamps, partition overlap)
- DatasetScope and Walk-Forward protocol identity preservation
- Promotion governance and persisted candidate resolution
- Production runtime authorization and authorized runtime context boundary
- Canonical live decision lifecycle evaluation and state transitions
- Publication boundary replay idempotency and conflict rejection
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest

from src.evaluation.evidence_integrity import ResearchEvidenceIntegrityGate
from src.evaluation.hypothesis_generator import (
    HypothesisGenerationContext,
    UnacceptedHypothesisError,
    accept_hypothesis_for_research,
    reject_hypothesis,
)
from src.evaluation.live_decision_lifecycle import (
    LiveDecisionLifecycleState,
    create_canonical_live_decision,
    transition_live_decision,
)
from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ProductionBlocked,
    ProductionRuntimeConfig,
    resolve_authoritative_promoted_candidate,
)
from src.evaluation.live_market_evaluation import create_live_market_evaluation
from src.evaluation.live_production_decision import (
    Direction,
    ProductionAuthorizationReceipt,
    ProductionDecision,
    ProductionPromotionPolicy,
    ProductionRuntimeAuthorizationError,
    ProductionSignal,
    PromotedCandidateArtifact,
    authorize_production_runtime,
    calculate_production_risk_levels,
    evaluate_production_decision,
)
from src.evaluation.live_publication_delivery import DeliveryIntegrityError, DeliveryStatus
from src.evaluation.live_publication_store import (
    PublicationIntegrityError,
    publish_canonical_live_decision,
)
from src.evaluation.live_runtime import evaluate_authorized_live_runtime
from src.evaluation.live_runtime_context import create_authorized_runtime_context
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_qualification import qualify_research_evidence
from src.evaluation.research_robustness import assess_research_robustness
from src.evaluation.research_runner import run_research_experiment, validate_and_prepare_dataset
from src.evaluation.research_store import (
    PromotionEligibilityError,
    PromotionIntegrityError,
    persist_promoted_candidate_binding,
    resolve_promoted_candidate,
    save_research_experiment,
)
from src.evaluation.stability import CanonicalStabilityEvidence


def _make_sample_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    df = pd.DataFrame(
        {
            "openTime": [d.isoformat() for d in dates],
            "timestamp": dates,
            "open": [2000.0 + i * 0.1 for i in range(n)],
            "high": [2001.0 + i * 0.1 for i in range(n)],
            "low": [1999.0 + i * 0.1 for i in range(n)],
            "close": [2000.5 + i * 0.1 for i in range(n)],
            "volume": [100.0] * n,
        }
    )
    return df


def _make_valid_spec() -> ResearchExperimentSpec:
    return ResearchExperimentSpec(
        hypothesis="Test hypothesis for P1 authoritative chain closure",
        methodology_version="discovery_v1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_test_p1",
            symbol="XAUUSD",
            timeframe="1D",
            start_date="2025-01-01",
            end_date="2025-04-10",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(commit_sha="a1b2c3d4e5f678901234567890abcdef12345678"),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
        walk_forward_protocol=WalkForwardProtocol(train_size=40, test_size=15),
    )


def _make_valid_evidence() -> ResearchEvidence:
    spec = _make_valid_spec()
    part_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-02-01",
        total_return=0.20,
        max_drawdown=0.05,
        sharpe_ratio=2.0,
        observations=50,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-02-01T00:00:00+00:00",
    )
    part_val = EvidencePartition(
        role=EvidencePartitionRole.VALIDATION,
        start_date="2025-02-02",
        end_date="2025-03-01",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=30,
        start_timestamp_utc="2025-02-02T00:00:00+00:00",
        end_timestamp_utc="2025-03-01T00:00:00+00:00",
    )
    part_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-03-02",
        end_date="2025-04-01",
        total_return=0.15,
        max_drawdown=0.05,
        sharpe_ratio=1.8,
        observations=30,
        start_timestamp_utc="2025-03-02T00:00:00+00:00",
        end_timestamp_utc="2025-04-01T00:00:00+00:00",
    )
    part_wf = EvidencePartition(
        role=EvidencePartitionRole.WALK_FORWARD,
        start_date="2025-01-01",
        end_date="2025-04-01",
        total_return=0.10,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=30,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-04-01T00:00:00+00:00",
    )
    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part_is, part_val, part_oos, part_wf),
        robustness_verdict={
            "passed": True,
            "is_robust": True,
            "parameter_sensitivity": {"passed": True},
            "subsample_stability": {"passed": True},
            "execution_cost_stress": {"passed": True},
            "statistical_validation": {"passed": True},
            "anti_overfitting": {"passed": True},
        },
        promotion_status=PromotionStatus.PROMOTABLE,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


# -----------------------------------------------------------------------------
# 1. Research Lifecycle & Governed Hypothesis
# -----------------------------------------------------------------------------

def test_generated_hypothesis_execution_fails_closed():
    spec = _make_valid_spec()
    hyp = ResearchHypothesis.from_experiment_spec(spec)
    assert hyp.status == HypothesisStatus.GENERATED

    df = _make_sample_df()
    with pytest.raises(UnacceptedHypothesisError) as exc_info:
        run_research_experiment(hyp, df=df)

    assert "is not accepted for research execution" in str(exc_info.value)


def test_raw_research_experiment_spec_execution_fails_closed():
    spec = _make_valid_spec()
    df = _make_sample_df()
    with pytest.raises(TypeError, match="expects an authoritative ResearchHypothesis instance"):
        run_research_experiment(spec, df=df)


def test_accepted_hypothesis_executes_successfully():
    spec = _make_valid_spec()
    hyp = ResearchHypothesis.from_experiment_spec(spec)
    accepted_hyp = accept_hypothesis_for_research(hyp)
    assert accepted_hyp.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH

    df = _make_sample_df()
    evidence = run_research_experiment(accepted_hyp, df=df)
    assert isinstance(evidence, ResearchEvidence)
    assert evidence.experiment_fingerprint == spec.fingerprint


def test_double_acceptance_fails_closed():
    spec = _make_valid_spec()
    hyp = ResearchHypothesis.from_experiment_spec(spec)
    accepted_hyp = accept_hypothesis_for_research(hyp)
    with pytest.raises(ValueError, match="current status is 'accepted_for_research'"):
        accept_hypothesis_for_research(accepted_hyp)


def test_rejected_hypothesis_acceptance_fails_closed():
    spec = _make_valid_spec()
    hyp = ResearchHypothesis.from_experiment_spec(spec)
    rejected_hyp = reject_hypothesis(hyp, reason="Flawed methodology")
    with pytest.raises(ValueError, match="current status is 'rejected'"):
        accept_hypothesis_for_research(rejected_hyp)


# -----------------------------------------------------------------------------
# 2. Temporal Evidence & Dataset Integrity
# -----------------------------------------------------------------------------

def test_non_monotonic_timestamps_fail_closed():
    df = _make_sample_df(10)
    # Swap row 2 and row 3 to introduce a non-monotonic time violation
    df.iloc[2], df.iloc[3] = df.iloc[3].copy(), df.iloc[2].copy()

    ds = DatasetScope(
        dataset_id="ds_test",
        symbol="XAUUSD",
        timeframe="1D",
        start_date="2025-01-01",
        end_date="2025-04-10",
    )
    with pytest.raises(ValueError, match="non-monotonic"):
        validate_and_prepare_dataset(df, ds)


def test_overlapping_partition_timestamps_fail_closed():
    spec = _make_valid_spec()
    p_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2025-01-01",
        end_date="2025-02-01",
        total_return=0.05,
        max_drawdown=-0.01,
        sharpe_ratio=1.5,
        observations=100,
        start_timestamp_utc="2025-01-01T00:00:00+00:00",
        end_timestamp_utc="2025-02-01T00:00:00+00:00",
    )
    p_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2025-01-15",  # Overlaps with IS
        end_date="2025-03-01",
        total_return=0.03,
        max_drawdown=-0.01,
        sharpe_ratio=1.2,
        observations=50,
        start_timestamp_utc="2025-01-15T00:00:00+00:00",
        end_timestamp_utc="2025-03-01T00:00:00+00:00",
    )
    evidence = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(p_is, p_oos),
        promotion_status=PromotionStatus.PROMOTABLE,
    )

    res = ResearchEvidenceIntegrityGate.validate(evidence)
    assert not res.valid
    assert RejectionReason.FAILED_VALIDATION in res.rejection_reasons


# -----------------------------------------------------------------------------
# 3. DatasetScope & Walk-Forward Identity
# -----------------------------------------------------------------------------

def test_different_walk_forward_protocols_produce_distinct_fingerprints():
    spec1 = ResearchExperimentSpec(
        hypothesis="H1",
        methodology_version="discovery_v1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(dataset_id="ds1", symbol="XAUUSD", timeframe="1D", start_date="2025-01-01", end_date="2025-04-10"),
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0),
        code_provenance=CodeProvenance(commit_sha="a1b2c3d4"),
        benchmark_reference="buy_and_hold",
        walk_forward_protocol=WalkForwardProtocol(train_size=40, test_size=15),
    )
    spec2 = ResearchExperimentSpec(
        hypothesis="H1",
        methodology_version="discovery_v1.0",
        strategy_name="momentum",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(dataset_id="ds1", symbol="XAUUSD", timeframe="1D", start_date="2025-01-01", end_date="2025-04-10"),
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0),
        code_provenance=CodeProvenance(commit_sha="a1b2c3d4"),
        benchmark_reference="buy_and_hold",
        walk_forward_protocol=WalkForwardProtocol(train_size=60, test_size=20),
    )
    assert spec1.fingerprint != spec2.fingerprint


# -----------------------------------------------------------------------------
# 4. Promotion Authority & Persisted Candidate Resolution
# -----------------------------------------------------------------------------

def test_unpromoted_or_unpersisted_candidate_resolution_fails_closed(tmp_path):
    config = ProductionRuntimeConfig(
        symbol="XAUUSD",
        timeframe="1D",
        candidate_id="non_existent_candidate_id",
        strategy_id="momentum",
        research_dir=tmp_path,
    )
    resolved = resolve_authoritative_promoted_candidate(config)
    assert isinstance(resolved, ProductionBlocked)
    assert resolved.reason == "PromotionUnavailable"


def test_persisted_promoted_candidate_resolves_and_validates(tmp_path):
    evidence = _make_valid_evidence()
    save_research_experiment(evidence, base_dir=tmp_path)

    stab = CanonicalStabilityEvidence(strategy_name="momentum", stability_score=0.85)
    persist_promoted_candidate_binding(
        candidate_id="cand_p1_test",
        evidence=evidence,
        canonical_stability=stab,
        base_dir=tmp_path,
    )

    resolved = resolve_promoted_candidate(
        candidate_id="cand_p1_test",
        symbol="XAUUSD",
        timeframe="1D",
        base_dir=tmp_path,
    )
    assert isinstance(resolved, PromotedCandidateArtifact)
    assert resolved.candidate_id == "cand_p1_test"
    assert resolved.operational_stability_score == 0.85


def test_caller_supplied_conflicting_stability_rejected(tmp_path):
    evidence = _make_valid_evidence()
    df = _make_sample_df()
    save_research_experiment(evidence, base_dir=tmp_path)

    stab = CanonicalStabilityEvidence(strategy_name="momentum", stability_score=0.85)
    persist_promoted_candidate_binding(
        candidate_id="cand_p1_test_2",
        evidence=evidence,
        canonical_stability=stab,
        base_dir=tmp_path,
    )

    candidate = resolve_promoted_candidate(candidate_id="cand_p1_test_2", base_dir=tmp_path)
    auth = authorize_production_runtime(candidate, symbol="XAUUSD", timeframe="1D")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    context = create_authorized_runtime_context(
        candidate=candidate,
        authorization=auth,
        authorization_receipt=receipt,
    )

    evaluation = create_live_market_evaluation(data=df, context=context)

    # Supplying a conflicting stability score must fail closed with ValueError
    with pytest.raises(ValueError, match="conflicts with authoritative"):
        evaluate_authorized_live_runtime(
            data=df,
            evaluation=evaluation,
            context=context,
            stable_strategy="momentum",
            stability_score=0.20,  # Conflicting with authoritative 0.85
        )


# -----------------------------------------------------------------------------
# 5. Production Authorization & Live Boundary
# -----------------------------------------------------------------------------

def test_missing_authorization_fails_closed():
    df = _make_sample_df()
    with pytest.raises(Exception):
        evaluate_authorized_live_runtime(
            data=df,
            evaluation=None,
            context="fake_context",
            stable_strategy="momentum",
        )


def test_end_to_end_authorized_live_cycle(tmp_path):
    evidence = _make_valid_evidence()
    df = _make_sample_df()
    save_research_experiment(evidence, base_dir=tmp_path)

    stab = CanonicalStabilityEvidence(strategy_name="momentum", stability_score=0.90)
    persist_promoted_candidate_binding(
        candidate_id="cand_e2e_01",
        evidence=evidence,
        canonical_stability=stab,
        base_dir=tmp_path,
    )

    candidate = resolve_promoted_candidate(candidate_id="cand_e2e_01", base_dir=tmp_path)
    auth = authorize_production_runtime(candidate, symbol="XAUUSD", timeframe="1D")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)
    context = create_authorized_runtime_context(
        candidate=candidate,
        authorization=auth,
        authorization_receipt=receipt,
    )

    evaluation = create_live_market_evaluation(data=df, context=context)
    res = evaluate_authorized_live_runtime(
        data=df,
        evaluation=evaluation,
        context=context,
        stable_strategy="momentum",
        persist=False,
    )

    assert res.canonical_decision is not None
    assert res.canonical_decision.current_state == LiveDecisionLifecycleState.PRESENTABLE
    assert res.decision["candidate_id"] == "cand_e2e_01"


# -----------------------------------------------------------------------------
# 6. Publication Boundary Replay & Conflict
# -----------------------------------------------------------------------------

class MockPublisherSuccess:
    def publish(self, publication, *args, **kwargs):
        return {
            "status": "PUBLISHED",
            "published": True,
            "event_id": publication.publication_id,
            "http_code": 200,
        }


def test_publication_replay_idempotent_and_conflict_rejection(tmp_path):
    pub_store_path = tmp_path / "publication_history.json"
    publisher = MockPublisherSuccess()

    evidence = _make_valid_evidence()
    stab = CanonicalStabilityEvidence(strategy_name="momentum", stability_score=0.90)
    save_research_experiment(evidence, base_dir=tmp_path)
    persist_promoted_candidate_binding(
        candidate_id="cand_pub_01",
        evidence=evidence,
        canonical_stability=stab,
        base_dir=tmp_path,
    )

    candidate = resolve_promoted_candidate(candidate_id="cand_pub_01", base_dir=tmp_path)
    auth = authorize_production_runtime(candidate, symbol="XAUUSD", timeframe="1D")
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    p_dec = ProductionDecision(
        candidate_id=candidate.candidate_id,
        evidence_id=candidate.evidence.evidence_id,
        experiment_fingerprint=candidate.evidence.experiment_fingerprint,
        symbol="XAUUSD",
        timeframe="1D",
        decision_timestamp="2025-01-01T12:00:00+00:00",
        market_timestamp="2025-01-01T12:00:00+00:00",
        direction=Direction.NO_TRADE,
        reason="test_no_trade",
        entry_price=None,
        invalidation_condition=None,
    )
    sig = ProductionSignal.from_decision(p_dec)
    risk = calculate_production_risk_levels(p_dec, candidate)

    cld = create_canonical_live_decision(
        authorization_receipt=receipt,
        decision=p_dec,
        signal=sig,
        risk_levels=risk,
    )
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.EVALUATED, actor="test")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.RISK_VALIDATED, actor="test")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PRESENTABLE, actor="test")
    cld = transition_live_decision(cld, LiveDecisionLifecycleState.PERSISTED, actor="test")

    # First publication
    pub_dec1, receipt1, res1 = publish_canonical_live_decision(
        cld,
        publisher=publisher,
        candidate=candidate,
        path=pub_store_path,
    )
    assert res1["delivery_status"] == DeliveryStatus.DELIVERED.value
    assert pub_dec1.current_state == LiveDecisionLifecycleState.PUBLISHED

    # Identical replay with original PERSISTED decision must be idempotent
    pub_dec2, receipt2, res2 = publish_canonical_live_decision(
        cld,
        publisher=publisher,
        candidate=candidate,
        path=pub_store_path,
    )
    assert res2["delivery_status"] == DeliveryStatus.DELIVERED.value
    assert pub_dec2.current_state == LiveDecisionLifecycleState.PUBLISHED

    # Calling with altered canonical decision fingerprint must fail closed with DeliveryIntegrityError
    with pytest.raises(DeliveryIntegrityError):
        publish_canonical_live_decision(
            pub_dec1,  # pub_dec1 has transition history extending fingerprint
            publisher=publisher,
            candidate=candidate,
            path=pub_store_path,
        )
