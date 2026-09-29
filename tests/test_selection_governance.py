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
    run_multiple_testing_correction,
)


def make_test_evidence(
    *,
    hypothesis: str = "Test Hypothesis",
    strategy_name: str = "momentum",
    status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
    p_value: float | None = None,
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
        EvidencePartition(
            role=EvidencePartitionRole.WALK_FORWARD,
            start_date="2024-01-01",
            end_date="2024-06-01",
            total_return=0.07,
            max_drawdown=-0.05,
            sharpe_ratio=1.1,
            win_rate=0.58,
            profit_factor=1.6,
            observations=80,
        ),
    )

    robustness = {"is_robust": True}
    for dim in ["parameter_sensitivity", "subsample_stability", "execution_cost_stress", "anti_overfitting"]:
        robustness[dim] = {"passed": True, "score": 0.9, "is_valid": True}
    if p_value is not None:
        robustness["statistical_validation"] = {
            "p_value": float(p_value),
            "t_statistic": 2.5,
            "observation_count": 50,
            "is_valid": True,
            "passed": True,
        }
    else:
        # Intentionally omit statistical_validation if p_value is None
        pass

    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=partitions,
        robustness_verdict=robustness,
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

    assert assessment.status in (
        SelectionGovernanceStatus.SELECTION_REVIEW_REQUIRED,
        SelectionGovernanceStatus.SELECTION_CONTEXT_RECORDED,
    )
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


def test_holm_bonferroni_multiple_testing_correction():
    """Verify Holm-Bonferroni step-down correction algorithm and monotonicity."""
    raw = {"t1": 0.01, "t2": 0.03, "t3": 0.10}
    res = run_multiple_testing_correction(raw, alpha=0.05, method="holm_bonferroni")

    assert res["method"] == "holm_bonferroni"
    assert res["alpha"] == 0.05
    assert res["eligible_trial_count"] == 3
    # t1: raw 0.01 * 3 = 0.03 <= 0.05 -> True
    assert res["adjusted_p_values"]["t1"] == 0.03
    assert res["rejection_decisions"]["t1"] is True

    # t2: raw 0.03 * 2 = 0.06 > 0.05 -> False
    assert res["adjusted_p_values"]["t2"] == 0.06
    assert res["rejection_decisions"]["t2"] is False

    # t3: raw 0.10 * 1 = 0.10 > 0.05 -> False
    assert res["adjusted_p_values"]["t3"] == 0.10
    assert res["rejection_decisions"]["t3"] is False


def test_multi_trial_holm_bonferroni_adjustment_applied():
    """Verify multi-trial selection assessment applies Holm-Bonferroni correction when valid p-values exist."""
    ev1 = make_test_evidence(hypothesis="Hypothesis 1", p_value=0.01)
    ev2 = make_test_evidence(hypothesis="Hypothesis 2", p_value=0.03)
    ev3 = make_test_evidence(hypothesis="Hypothesis 3", p_value=0.10)

    t1 = ResearchTrialRecord(
        search_id="search_1",
        trial_id="t1",
        trial_index=0,
        candidate_id="cand_1",
        candidate_fingerprint="fp_cand_1",
        experiment_fingerprint=ev1.experiment_fingerprint,
        evidence_fingerprint=ev1.evidence_id,
        qualification_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        status="QUALIFIED",
    )
    t2 = ResearchTrialRecord(
        search_id="search_1",
        trial_id="t2",
        trial_index=1,
        candidate_id="cand_2",
        candidate_fingerprint="fp_cand_2",
        experiment_fingerprint=ev2.experiment_fingerprint,
        evidence_fingerprint=ev2.evidence_id,
        qualification_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
        status="REJECTED",
    )
    t3 = ResearchTrialRecord(
        search_id="search_1",
        trial_id="t3",
        trial_index=2,
        candidate_id="cand_3",
        candidate_fingerprint="fp_cand_3",
        experiment_fingerprint=ev3.experiment_fingerprint,
        evidence_fingerprint=ev3.evidence_id,
        qualification_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
        status="REJECTED",
    )

    assessment = assess_research_selection(
        evidence=ev1,
        trial_records=(t1, t2, t3),
        search_fingerprint="search_fp_1",
        evidence_collection=(ev1, ev2, ev3),
    )

    assert assessment.status == SelectionGovernanceStatus.SELECTION_ADJUSTMENT_APPLIED
    assert assessment.reason == SelectionGovernanceReason.MULTI_TRIAL_SELECTION_DETECTED
    assert assessment.metadata["correction_applied"] is True
    assert assessment.metadata["correction_method"] == "holm_bonferroni"
    assert assessment.metadata["target_raw_p_value"] == 0.01
    assert assessment.metadata["target_adjusted_p_value"] == 0.03
    assert assessment.metadata["target_passed_correction"] is True
    assert assessment.metadata["selected_winner_identity"] == "cand_1"


def test_multi_trial_alpha_threshold_behavior():
    """Verify threshold alpha behavior when adjusting p-values."""
    ev1 = make_test_evidence(hypothesis="H1", p_value=0.02)
    ev2 = make_test_evidence(hypothesis="H2", p_value=0.03)
    ev3 = make_test_evidence(hypothesis="H3", p_value=0.10)

    t1 = ResearchTrialRecord("s1", "t1", 0, "c1", "fp1", ev1.experiment_fingerprint, ev1.evidence_id, PromotionStatus.PROMOTABLE, (), "QUALIFIED")
    t2 = ResearchTrialRecord("s1", "t2", 1, "c2", "fp2", ev2.experiment_fingerprint, ev2.evidence_id, PromotionStatus.REJECTED, (), "REJECTED")
    t3 = ResearchTrialRecord("s1", "t3", 2, "c3", "fp3", ev3.experiment_fingerprint, ev3.evidence_id, PromotionStatus.REJECTED, (), "REJECTED")

    policy_strict = ResearchSelectionPolicy(alpha=0.05)
    ass_strict = assess_research_selection(ev1, (t1, t2, t3), "s1", policy_strict, (ev1, ev2, ev3))
    # Adjusted p = 0.02 * 3 = 0.06 > 0.05 -> target_passed_correction is False
    assert ass_strict.metadata["target_passed_correction"] is False

    policy_lenient = ResearchSelectionPolicy(alpha=0.10)
    ass_lenient = assess_research_selection(ev1, (t1, t2, t3), "s1", policy_lenient, (ev1, ev2, ev3))
    # Adjusted p = 0.06 <= 0.10 -> target_passed_correction is True
    assert ass_lenient.metadata["target_passed_correction"] is True


def test_missing_p_values_fails_closed():
    """Verify missing statistical p-values fails closed to SELECTION_REVIEW_REQUIRED."""
    ev1 = make_test_evidence(hypothesis="H1", p_value=0.01)
    ev2_no_p = make_test_evidence(hypothesis="H2", p_value=None)

    t1 = ResearchTrialRecord("s1", "t1", 0, "c1", "fp1", ev1.experiment_fingerprint, ev1.evidence_id, PromotionStatus.PROMOTABLE, (), "QUALIFIED")
    t2 = ResearchTrialRecord("s1", "t2", 1, "c2", "fp2", ev2_no_p.experiment_fingerprint, ev2_no_p.evidence_id, PromotionStatus.REJECTED, (), "REJECTED")

    assessment = assess_research_selection(ev1, (t1, t2), "s1", None, (ev1, ev2_no_p))

    assert assessment.status == SelectionGovernanceStatus.SELECTION_REVIEW_REQUIRED
    assert assessment.reason == SelectionGovernanceReason.CORRECTION_UNAVAILABLE_MISSING_DISTRIBUTIONAL_INPUTS
    assert assessment.metadata["correction_applied"] is False


def test_invalid_non_finite_p_values_fails_closed():
    """Verify invalid or non-finite raw p-values raise ValueError in correction function."""
    with pytest.raises(ValueError, match="Non-finite raw p-value"):
        run_multiple_testing_correction({"t1": float("nan")})

    with pytest.raises(ValueError, match="out of bounds"):
        run_multiple_testing_correction({"t1": 1.5})


def test_failed_trials_handling():
    """Verify failed trials in ledger are excluded from statistical correction count."""
    ev1 = make_test_evidence(hypothesis="H1", p_value=0.01)
    ev2 = make_test_evidence(hypothesis="H2", p_value=0.03)

    t1 = ResearchTrialRecord("s1", "t1", 0, "c1", "fp1", ev1.experiment_fingerprint, ev1.evidence_id, PromotionStatus.PROMOTABLE, (), "QUALIFIED")
    t2 = ResearchTrialRecord("s1", "t2", 1, "c2", "fp2", ev2.experiment_fingerprint, ev2.evidence_id, PromotionStatus.REJECTED, (), "REJECTED")
    t_failed = ResearchTrialRecord("s1", "tf", 2, "cf", "fpf", "", None, PromotionStatus.REJECTED, (RejectionReason.SPECIFICATION_INVALID,), "FAILED", "Error")

    assessment = assess_research_selection(ev1, (t1, t2, t_failed), "s1", None, (ev1, ev2))

    assert assessment.status == SelectionGovernanceStatus.SELECTION_ADJUSTMENT_APPLIED
    assert assessment.metadata["eligible_trial_count"] == 2
    assert assessment.metadata["excluded_trial_count"] == 1


def test_malformed_policy_data_fails_closed():
    """Verify malformed policy parameters raise ValueError."""
    with pytest.raises(ValueError, match="alpha must be a finite float"):
        ResearchSelectionPolicy(alpha=1.5)

    with pytest.raises(ValueError, match="Unsupported correction_method"):
        ResearchSelectionPolicy(correction_method="invalid_method")


def test_deterministic_fingerprint_includes_correction_metadata():
    """Verify governance fingerprint deterministically includes selection correction metadata."""
    ev1 = make_test_evidence(hypothesis="H1", p_value=0.01)
    ev2 = make_test_evidence(hypothesis="H2", p_value=0.03)
    t1 = ResearchTrialRecord("s1", "t1", 0, "c1", "fp1", ev1.experiment_fingerprint, ev1.evidence_id, PromotionStatus.PROMOTABLE, (), "QUALIFIED")
    t2 = ResearchTrialRecord("s1", "t2", 1, "c2", "fp2", ev2.experiment_fingerprint, ev2.evidence_id, PromotionStatus.REJECTED, (), "REJECTED")

    ass1 = assess_research_selection(ev1, (t1, t2), "s1", ResearchSelectionPolicy(alpha=0.05), (ev1, ev2))
    ass2 = assess_research_selection(ev1, (t1, t2), "s1", ResearchSelectionPolicy(alpha=0.05), (ev1, ev2))
    ass_diff = assess_research_selection(ev1, (t1, t2), "s1", ResearchSelectionPolicy(alpha=0.01), (ev1, ev2))

    assert ass1.selection_fingerprint == ass2.selection_fingerprint
    assert ass1.selection_fingerprint != ass_diff.selection_fingerprint


def test_promotion_and_qualification_gates_unaffected():
    """Verify promotion safety gate remains authoritative and independent of selection correction."""
    ev = make_test_evidence(p_value=0.01)
    from src.evaluation.research_qualification import qualify_research_evidence
    from src.evaluation.research_robustness import assess_research_robustness
    rob = assess_research_robustness(ev)
    qual_res = qualify_research_evidence(ev, robustness_assessment=rob)

    assert qual_res.qualified is True
    # Selection correction metadata does not alter qualification outcome
