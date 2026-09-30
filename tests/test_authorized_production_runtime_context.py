"""Tests for AuthorizedProductionRuntimeContext immutability, fingerprinting, and fail-closed lineage validation."""

from __future__ import annotations

import datetime
from dataclasses import FrozenInstanceError

import pytest

from src.evaluation.live_production_decision import (
    ProductionAuthorizationReceipt,
    PromotedCandidateArtifact,
    authorize_production_runtime,
)
from src.evaluation.live_runtime_context import (
    AuthorizedProductionRuntimeContext,
    RuntimeContextValidationError,
    create_authorized_runtime_context,
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


def _make_dummy_evidence(
    candidate_id: str = "cand_test",
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
) -> ResearchEvidence:
    spec = ResearchExperimentSpec(
        hypothesis="Test hypothesis for production context",
        methodology_version="1.0",
        strategy_name="momentum",
        strategy_version="1.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_xauusd_5m",
            symbol=symbol,
            timeframe=timeframe,
            start_date="2024-01-01",
            end_date="2024-06-01",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=100.0,
        ),
        code_provenance=CodeProvenance(commit_sha="a1b2c3d4e5f6"),
        benchmark_reference="buy_and_hold",
        parameters={"momentum_window": 10, "stop_loss_pct": 0.01, "take_profit_pct": 0.02},
    )

    part = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2024-01-01",
        end_date="2024-06-01",
        total_return=0.1,
        max_drawdown=0.05,
        sharpe_ratio=1.5,
        observations=100,
    )

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(part,),
        promotion_status=PromotionStatus.PROMOTABLE,
    )


def _make_dummy_candidate(
    candidate_id: str = "cand_test",
    symbol: str = "XAUUSD",
    timeframe: str = "5m",
    gov_fp: str = "gov_fp_1234567890",
    cs_fp: str | None = "cs_fp_1234567890",
) -> PromotedCandidateArtifact:
    evidence = _make_dummy_evidence(candidate_id, symbol, timeframe)
    return PromotedCandidateArtifact.from_persisted_research(
        candidate_id=candidate_id,
        evidence=evidence,
        symbol=symbol,
        timeframe=timeframe,
        governance_decision_fingerprint=gov_fp,
        campaign_selection_decision_fingerprint=cs_fp,
    )


def test_runtime_context_creation_and_immutability():
    cand = _make_dummy_candidate()
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=ref_now)
    ctx = create_authorized_runtime_context(cand, auth)

    assert ctx.candidate_id == "cand_test"
    assert ctx.symbol == "XAUUSD"
    assert ctx.timeframe == "5m"
    assert ctx.strategy_id == "momentum"
    assert ctx.strategy_version == "1.0"
    assert ctx.promoted_artifact_fingerprint == cand.artifact_fingerprint
    assert ctx.governance_decision_fingerprint == "gov_fp_1234567890"
    assert ctx.campaign_selection_decision_fingerprint == "cs_fp_1234567890"
    assert ctx.authorization_fingerprint == auth.authorization_fingerprint
    assert isinstance(ctx.context_fingerprint, str) and len(ctx.context_fingerprint) == 64

    # Frozen dataclass immutability test
    with pytest.raises(FrozenInstanceError):
        ctx.symbol = "EURUSD"  # type: ignore


def test_context_fingerprint_determinism_and_sensitivity():
    cand = _make_dummy_candidate()
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    auth1 = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=ref_now)
    ctx1 = create_authorized_runtime_context(cand, auth1)

    auth2 = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=ref_now)
    ctx2 = create_authorized_runtime_context(cand, auth2)

    assert ctx1.context_fingerprint == ctx2.context_fingerprint

    # Material candidate change alters context fingerprint
    cand_alt = _make_dummy_candidate(candidate_id="cand_test_alt")
    auth_alt = authorize_production_runtime(cand_alt, symbol="XAUUSD", timeframe="5m", now=ref_now)
    ctx_alt = create_authorized_runtime_context(cand_alt, auth_alt)

    assert ctx_alt.context_fingerprint != ctx1.context_fingerprint


def test_context_validation_fails_closed_on_mismatch():
    cand = _make_dummy_candidate()
    ref_now = datetime.datetime(2025, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
    auth = authorize_production_runtime(cand, symbol="XAUUSD", timeframe="5m", now=ref_now)
    receipt = ProductionAuthorizationReceipt.from_authorization(auth)

    # 1. Candidate ID mismatch
    with pytest.raises(RuntimeContextValidationError, match="Context candidate_id 'other_id' does not match"):
        AuthorizedProductionRuntimeContext(
            candidate=cand,
            authorization=auth,
            authorization_receipt=receipt,
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id="other_id",
            strategy_id="momentum",
            strategy_version="1.0",
            promoted_artifact_fingerprint=cand.artifact_fingerprint,
            governance_decision_fingerprint=auth.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=auth.campaign_selection_decision_fingerprint,
            authorization_fingerprint=auth.authorization_fingerprint,
            authorization_policy_version=auth.authorization_policy_version,
        )

    # 2. Symbol mismatch
    with pytest.raises(RuntimeContextValidationError, match="Context symbol 'EURUSD' does not match candidate symbol"):
        AuthorizedProductionRuntimeContext(
            candidate=cand,
            authorization=auth,
            authorization_receipt=receipt,
            symbol="EURUSD",
            timeframe="5m",
            candidate_id=cand.candidate_id,
            strategy_id="momentum",
            strategy_version="1.0",
            promoted_artifact_fingerprint=cand.artifact_fingerprint,
            governance_decision_fingerprint=auth.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=auth.campaign_selection_decision_fingerprint,
            authorization_fingerprint=auth.authorization_fingerprint,
            authorization_policy_version=auth.authorization_policy_version,
        )

    # 3. Governance fingerprint mismatch
    with pytest.raises(RuntimeContextValidationError, match="governance_decision_fingerprint 'spoofed_gov' does not match"):
        AuthorizedProductionRuntimeContext(
            candidate=cand,
            authorization=auth,
            authorization_receipt=receipt,
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id=cand.candidate_id,
            strategy_id="momentum",
            strategy_version="1.0",
            promoted_artifact_fingerprint=cand.artifact_fingerprint,
            governance_decision_fingerprint="spoofed_gov",
            campaign_selection_decision_fingerprint=auth.campaign_selection_decision_fingerprint,
            authorization_fingerprint=auth.authorization_fingerprint,
            authorization_policy_version=auth.authorization_policy_version,
        )

    # 4. Authorization fingerprint mismatch
    with pytest.raises(RuntimeContextValidationError, match="Context authorization_fingerprint 'spoofed_auth' does not match"):
        AuthorizedProductionRuntimeContext(
            candidate=cand,
            authorization=auth,
            authorization_receipt=receipt,
            symbol="XAUUSD",
            timeframe="5m",
            candidate_id=cand.candidate_id,
            strategy_id="momentum",
            strategy_version="1.0",
            promoted_artifact_fingerprint=cand.artifact_fingerprint,
            governance_decision_fingerprint=auth.governance_decision_fingerprint,
            campaign_selection_decision_fingerprint=auth.campaign_selection_decision_fingerprint,
            authorization_fingerprint="spoofed_auth",
            authorization_policy_version=auth.authorization_policy_version,
        )
