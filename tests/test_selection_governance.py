"""Comprehensive unit and architectural boundary tests for Research Selection Governance.

Tests multiple-testing selection bias governance status classification,
SHA-256 fingerprint determinism and immutability, trial count retention from ledgers,
and strict architectural boundaries (no imports/calls to production decisioning,
risk geometry, publication, live execution, or Project 2 integration).
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import json
import pathlib
import pytest

from src.evaluation.discovery_engine import (
    CandidateSpec,
    DiscoveryEngine,
    ResearchSearchSpace,
    ResearchTrialRecord,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    EvidencePartition,
    EvidencePartitionRole,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchEvidence,
    ResearchExperimentSpec,
)
from src.evaluation.selection_governance import (
    ResearchSelectionAssessment,
    ResearchSelectionPolicy,
    SelectionGovernanceReason,
    SelectionGovernanceStatus,
    assess_research_selection,
    compute_selection_fingerprint,
)


def make_test_evidence(
    *,
    hypothesis: str = "Test Hypothesis",
    strategy_name: str = "momentum",
    status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
) -> ResearchEvidence:
    """Construct valid ResearchEvidence for testing."""
    ds = DatasetScope(
        dataset_id="test_ds",
        symbol="XAUUSD",
        timeframe="1h",
        start_date="2024-01-01",
        end_date="2024-06-01",
    )
    ea = ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="abcdef1234567890")

    spec = ResearchExperimentSpec(
        hypothesis=hypothesis,
        methodology_version="discovery_v1.0",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        benchmark_reference="buy_and_hold",
        parameters={"window": 14},
    )

    partitions = (
        EvidencePartition(
            role=EvidencePartitionRole.IN_SAMPLE,
            start_date="2024-01-01",
            end_date="2024-03-01",
            total_return=0.10,
            max_drawdown=-0.05,
            sharpe_ratio=1.5,
            win_rate=0.6,
            profit_factor=1.8,
            observations=100,
        ),
        EvidencePartition(
            role=EvidencePartitionRole.OUT_OF_SAMPLE,
            start_date="2024-03-01",
            end_date="2024-06-01",
            total_return=0.05,
            max_drawdown=-0.04,
            sharpe_ratio=1.2,
            win_rate=0.55,
            profit_factor=1.5,
            observations=50,
        ),
    )

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=partitions,
        robustness_verdict={"is_robust": True},
        promotion_status=status,
        rejection_reasons=rejection_reasons,
    )


def test_single_hypothesis_selection_assessment():
    """Verify single-hypothesis assessment returns SELECTION_NOT_APPLICABLE status."""
    ev = make_test_evidence()
    assessment = assess_research_selection(ev, trial_records=None)

    assert isinstance(assessment, ResearchSelectionAssessment)
    assert assessment.status == SelectionGovernanceStatus.SELECTION_NOT_APPLICABLE
    assert assessment.reason == SelectionGovernanceReason.SINGLE_HYPOTHESIS_NO_SELECTION
    assert assessment.trial_count == 1
    assert assessment.completed_trial_count == 1
    assert assessment.failed_trial_count == 0
    assert assessment.qualified_trial_count == 1
    assert assessment.rejected_trial_count == 0
    assert assessment.is_selected_winner is True
    assert len(assessment.selection_fingerprint) == 64


def test_multi_trial_selection_assessment_preserves_ledger_counts():
    """Verify multi-trial search assessment retains exact trial ledger counts."""
    ev = make_test_evidence()

    trial1 = ResearchTrialRecord(
        search_id="search_100",
        trial_id="trial_0",
        trial_index=0,
        candidate_id="cand_1",
        candidate_fingerprint="fp_cand_1",
        experiment_fingerprint=ev.experiment_fingerprint,
        evidence_fingerprint=ev.evidence_id,
        qualification_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        status="QUALIFIED",
    )
    trial2 = ResearchTrialRecord(
        search_id="search_100",
        trial_id="trial_1",
        trial_index=1,
        candidate_id="cand_2",
        candidate_fingerprint="fp_cand_2",
        experiment_fingerprint="fp_exp_2",
        evidence_fingerprint="ev_2",
        qualification_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
        status="REJECTED",
    )
    trial3 = ResearchTrialRecord(
        search_id="search_100",
        trial_id="trial_2",
        trial_index=2,
        candidate_id="cand_3",
        candidate_fingerprint="fp_cand_3",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        qualification_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.SPECIFICATION_INVALID,),
        status="FAILED",
        error_message="Execution exception",
    )

    assessment = assess_research_selection(
        evidence=ev,
        trial_records=(trial1, trial2, trial3),
        search_fingerprint="search_space_fp_123",
    )

    assert assessment.status == SelectionGovernanceStatus.SELECTION_CONTEXT_RECORDED
    assert (
        assessment.reason
        == SelectionGovernanceReason.CORRECTION_UNAVAILABLE_MISSING_DISTRIBUTIONAL_INPUTS
    )
    assert assessment.search_fingerprint == "search_space_fp_123"
    assert assessment.trial_count == 3
    assert assessment.completed_trial_count == 2
    assert assessment.failed_trial_count == 1
    assert assessment.qualified_trial_count == 1
    assert assessment.rejected_trial_count == 1
    assert assessment.candidate_id == "cand_1"
    assert assessment.candidate_fingerprint == "fp_cand_1"
    assert assessment.is_selected_winner is True


def test_selection_assessment_immutability_and_sha256_determinism():
    """Verify assessment object is frozen and fingerprinting is deterministic SHA-256."""
    ev = make_test_evidence()
    assessment1 = assess_research_selection(ev)
    assessment2 = assess_research_selection(ev)

    # Immutability check
    with pytest.raises(FrozenInstanceError):
        assessment1.status = SelectionGovernanceStatus.SELECTION_REVIEW_REQUIRED  # type: ignore

    # SHA-256 Fingerprint determinism
    assert assessment1.selection_fingerprint == assessment2.selection_fingerprint
    assert len(assessment1.selection_fingerprint) == 64


def test_architectural_boundary_no_production_or_live_imports():
    """AST test enforcing selection_governance.py does NOT import production or live modules."""
    module_path = pathlib.Path("src/evaluation/selection_governance.py")
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    forbidden_terms = (
        "ProductionDecision",
        "calculate_production_risk_levels",
        "live_execution",
        "Project2Publisher",
        "publish",
    )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for term in forbidden_terms:
                    assert term not in alias.name, f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for term in forbidden_terms:
                    assert term not in node.module, f"Forbidden from-import module found: {node.module}"
            for alias in node.names:
                for term in forbidden_terms:
                    assert term not in alias.name, f"Forbidden imported symbol found: {alias.name}"
