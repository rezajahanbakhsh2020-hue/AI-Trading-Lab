"""Comprehensive unit and integration test suite for Memory-Aware Discovery Governance.

Validates requirement matrix A through O:
A. No learning memory -> discovery behavior remains unchanged.
B. Existing informational ResearchLesson does not block a candidate.
C. Matching active DoNotRepeatConstraint blocks candidate before expensive experiment execution.
D. Non-matching active constraint does not block.
E. Multiple active constraints are evaluated deterministically.
F. Matching constraint produces an explicit auditable governance result with SHA-256 fingerprint.
G. Blocked trial remains represented in discovery lifecycle (trial ledger, registry, research candidates) and does not silently disappear.
H. Malformed/ambiguous active constraint fails closed.
I. Same inputs + same memory state produce identical governance decisions and fingerprints.
J. Learning-memory governance does not mutate production/live execution code.
K. No Project2 dependency or Project2 source modification.
L. Existing PR #24/#25 learning persistence tests remain valid.
M. Existing DiscoveryEngine behavior without applicable constraints remains compatible.
N. Blocked-by-memory trial is not recorded as an executed experiment result.
O. Existing registry idempotency/provenance guarantees remain intact.
"""

from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.discovery_engine import (
    DiscoveryCriteria,
    DiscoveryEngine,
    DiscoveryRunResult,
    ResearchSearchPolicy,
)
from src.evaluation.memory_governance import (
    DiscoveryMemoryGovernanceResult,
    MemoryConstraintMatch,
    MemoryGovernanceDecision,
    evaluate_candidate_memory_governance,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
    ResearchExperimentSpec,
)
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    LessonCategory,
    RegistryStatus,
    RegistryValidationError,
    ResearchEvidenceLineage,
    ResearchLearningRecord,
    ResearchLesson,
    ResearchOutcomeClassification,
    ResearchRegistryRecord,
    ResearchRegistryStore,
    ResearchReproducibilityDescriptor,
    StructuredObservedConditions,
    construct_learning_record_from_registry_record,
    construct_registry_record_from_evidence,
)


@pytest.fixture
def sample_market_data() -> pd.DataFrame:
    """Generate deterministic synthetic market data for discovery engine tests."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-06-01", freq="1D")
    n = len(dates)
    price = 1900.0 + np.cumsum(np.random.randn(n) * 2.0)
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price,
            "high": price + 2.0,
            "low": price - 2.0,
            "close": price + 0.5,
            "volume": 1000.0,
        }
    )
    return df


@pytest.fixture
def dataset_scope() -> DatasetScope:
    return DatasetScope("ds_test", "XAUUSD", "1D", "2023-01-01", "2023-06-01")


@pytest.fixture
def execution_assumptions() -> ExecutionAssumptions:
    return ExecutionAssumptions(transaction_cost=0.0005, slippage=0.0002, latency_ms=50.0)


@pytest.fixture
def code_provenance() -> CodeProvenance:
    return CodeProvenance(commit_sha="a1b2c3d4", repository_status="clean", author="Jules")


@pytest.fixture
def candidates() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec(
            generator_name="test_generator",
            generator_version="1.0.0",
            strategy_name="baseline_momentum",
            parameters={"fast_window": 3, "slow_window": 8},
            hypothesis_template="Test momentum fast with parameters fast={fast_window}",
        ),
        CandidateSpec(
            generator_name="test_generator",
            generator_version="1.0.0",
            strategy_name="baseline_momentum",
            parameters={"fast_window": 10, "slow_window": 30},
            hypothesis_template="Test momentum slow with parameters slow={slow_window}",
        ),
    )


def test_A_no_learning_memory_discovery_unchanged(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test A: When no learning memory exists, discovery behavior remains unchanged."""
    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=None,
    )

    assert isinstance(result, DiscoveryRunResult)
    assert result.candidates_evaluated == len(candidates)
    assert result.blocked_trial_count == 0
    assert result.executed_trial_count == len(candidates)
    assert len(result.memory_governance_results) == 0


def test_B_informational_lesson_does_not_block(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates, tmp_path
):
    """Test B: Existing informational ResearchLesson without active DoNotRepeatConstraint does not block candidates."""
    store = ResearchRegistryStore(base_dir=tmp_path)

    # Register source registry record and learning record with lessons but NO constraints
    obs_conditions = StructuredObservedConditions(
        symbol="XAUUSD",
        timeframe="1D",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope_id="ds_test",
        execution_assumptions_id="ea_test",
        code_provenance_id="cp_test",
        methodology_version="discovery_v1.0",
        out_of_sample_evaluated=True,
        walk_forward_evaluated=True,
        benchmark_status="PASSED",
        regime_status="PASSED",
        selection_status=None,
        robustness_status=None,
        rejection_reasons=(),
        metrics={},
    )
    lesson = ResearchLesson(
        lesson_id="les_info_1",
        category=LessonCategory.STRATEGY_PERFORMANCE,
        source_record_id="rec_info_1",
        experiment_fingerprint="exp_info_1",
        evidence_fingerprint="ev_info_1",
        observed_conditions=obs_conditions,
        conclusion="Strategy showed positive performance on historical data.",
        confidence_score=0.9,
        methodology_version="discovery_v1.0",
    )
    learning = ResearchLearningRecord(
        learning_id="learn_info_1",
        source_record_id="rec_info_1",
        experiment_fingerprint="exp_info_1",
        evidence_fingerprint="ev_info_1",
        candidate_id="cand_1",
        search_fingerprint="search_1",
        trial_id="trial_1",
        dataset_scope_id="ds_test",
        execution_assumptions_id="ea_test",
        code_provenance_id="cp_test",
        methodology_version="discovery_v1.0",
        classification=ResearchOutcomeClassification.SUCCESS,
        observed_conditions=obs_conditions,
        lessons=(lesson,),
        constraints=(),  # No constraints!
        confidence_score=0.9,
        rejection_reasons=(),
    )

    repro = ResearchReproducibilityDescriptor(
        experiment_fingerprint="exp_info_1",
        evidence_fingerprint="ev_info_1",
        dataset_scope_id="ds_test",
        execution_assumptions_id="ea_test",
        code_provenance_id="cp_test",
        methodology_version="discovery_v1.0",
        search_space_fingerprint=None,
        trial_id="trial_1",
        candidate_id="cand_1",
    )
    lin = ResearchEvidenceLineage(
        search_id="search_1",
        search_fingerprint=None,
        trial_id="trial_1",
        trial_index=0,
        candidate_id="cand_1",
        experiment_fingerprint="exp_info_1",
        evidence_fingerprint="ev_info_1",
        qualification_status="QUALIFIED",
        selection_assessment_id=None,
        robustness_assessment_id=None,
        promotion_status="PROMOTABLE",
    )
    dummy_reg = ResearchRegistryRecord(
        record_id="rec_info_1",
        experiment_fingerprint="exp_info_1",
        evidence_fingerprint="ev_info_1",
        candidate_id="cand_1",
        search_fingerprint="search_1",
        search_id="search_1",
        trial_id="trial_1",
        trial_index=0,
        status=RegistryStatus.QUALIFIED,
        qualification_status="QUALIFIED",
        promotion_status="PROMOTABLE",
        rejection_reasons=(),
        dataset_scope_id="ds_test",
        execution_assumptions_id="ea_test",
        code_provenance_id="cp_test",
        methodology_version="discovery_v1.0",
        selection_assessment_id=None,
        robustness_assessment_id=None,
        benchmark_status="PASSED",
        regime_status="PASSED",
        error_message="",
        reproducibility=repro,
        lineage=lin,
    )
    store.register(dummy_reg)
    store.register_learning_record(learning)

    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=store,
    )

    assert result.blocked_trial_count == 0
    assert result.executed_trial_count == len(candidates)


def test_C_matching_active_constraint_blocks_candidate(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test C & N: Matching active DoNotRepeatConstraint blocks candidate before expensive experiment execution."""
    target_candidate = candidates[0]

    # Create active constraint matching candidate 0
    constraint = DoNotRepeatConstraint(
        constraint_id="c_block_cand0",
        source_record_id="rec_source_0",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key=target_candidate.candidate_id,
        condition_description=f"Strategy {target_candidate.strategy_name} failed previously",
        reason="Repeated drawdown breach in historical backtest",
        confidence_score=0.95,
        is_active=True,
    )

    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=(constraint,),
    )

    assert result.blocked_trial_count == 1
    assert result.executed_trial_count == 1
    assert len(result.memory_governance_results) == len(candidates)

    blocked_gov = result.memory_governance_results[0]
    assert blocked_gov.decision == MemoryGovernanceDecision.BLOCKED
    assert blocked_gov.matching_constraint is not None
    assert blocked_gov.matching_constraint.constraint_id == "c_block_cand0"

    allowed_gov = result.memory_governance_results[1]
    assert allowed_gov.decision == MemoryGovernanceDecision.ALLOWED

    # N. Ensure blocked trial is NOT recorded as executed experiment result
    assert result.trial_ledger[0].status == "BLOCKED"
    assert RejectionReason.GOVERNANCE_BLOCKED in result.trial_ledger[0].rejection_reasons


def test_D_non_matching_active_constraint_does_not_block(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test D: Non-matching active constraint does not block candidate."""
    # Active constraint for a completely different strategy
    constraint = DoNotRepeatConstraint(
        constraint_id="c_other_strategy",
        source_record_id="rec_other",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key="fail_pattern:unrelated_strategy:EURUSD:1H:ds_other",
        condition_description="Unrelated strategy failed on EURUSD 1H",
        reason="Unrelated strategy failure",
        confidence_score=0.90,
        is_active=True,
    )

    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=(constraint,),
    )

    assert result.blocked_trial_count == 0
    assert result.executed_trial_count == len(candidates)


def test_E_multiple_active_constraints_evaluated_deterministically(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test E: Multiple active constraints are sorted and evaluated in deterministic order."""
    c1 = DoNotRepeatConstraint(
        constraint_id="c_01_non_match",
        source_record_id="rec_1",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key="non_matching_pattern_1",
        condition_description="Condition 1",
        reason="Reason 1",
        confidence_score=0.8,
        is_active=True,
    )
    c2 = DoNotRepeatConstraint(
        constraint_id="c_02_match",
        source_record_id="rec_2",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key=candidates[0].candidate_id,
        condition_description="Condition 2",
        reason="Reason 2",
        confidence_score=0.9,
        is_active=True,
    )

    # Pass in reverse order to verify deterministic sorting by constraint_id
    res1 = evaluate_candidate_memory_governance(
        candidate=candidates[0],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        active_constraints=(c2, c1),
    )

    res2 = evaluate_candidate_memory_governance(
        candidate=candidates[0],
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        active_constraints=(c1, c2),
    )

    assert res1.decision == MemoryGovernanceDecision.BLOCKED
    assert res2.decision == MemoryGovernanceDecision.BLOCKED
    assert res1.governance_fingerprint == res2.governance_fingerprint


def test_F_matching_constraint_produces_auditable_governance_result(
    dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test F: Matching constraint produces an explicit, auditable governance result with SHA-256 fingerprint."""
    cand = candidates[0]
    constraint = DoNotRepeatConstraint(
        constraint_id="c_audit_100",
        source_record_id="rec_audit_100",
        experiment_fingerprint="",
        evidence_fingerprint="ev_audit_100",
        pattern_key=cand.candidate_id,
        condition_description="Audit condition test",
        reason="Historical failure reason for audit",
        confidence_score=0.92,
        is_active=True,
    )

    gov_res = evaluate_candidate_memory_governance(
        candidate=cand,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        active_constraints=(constraint,),
        search_id="search_audit_1",
        search_fingerprint="search_fp_123",
    )

    assert gov_res.decision == MemoryGovernanceDecision.BLOCKED
    assert gov_res.candidate_id == cand.candidate_id
    assert gov_res.search_id == "search_audit_1"
    assert gov_res.search_fingerprint == "search_fp_123"
    assert len(gov_res.governance_fingerprint) == 64  # SHA-256
    assert gov_res.matching_constraint is not None
    assert gov_res.matching_constraint.constraint_id == "c_audit_100"
    assert gov_res.matching_constraint.source_record_id == "rec_audit_100"
    assert gov_res.matching_constraint.evidence_fingerprint == "ev_audit_100"


def test_G_blocked_trial_represented_in_discovery_lifecycle(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test G: Blocked trial remains represented in discovery lifecycle and does not silently disappear."""
    cand = candidates[0]
    constraint = DoNotRepeatConstraint(
        constraint_id="c_visible_block",
        source_record_id="rec_source_vis",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key=cand.candidate_id,
        condition_description="Visibility test condition",
        reason="Visibility test failure reason",
        confidence_score=0.95,
        is_active=True,
    )

    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=(constraint,),
    )

    # Must be represented in total trial count
    assert len(result.trial_ledger) == len(candidates)
    assert len(result.research_candidates) == len(candidates)
    assert len(result.registry_records) == len(candidates)

    # Blocked trial assertions
    tr_blocked = result.trial_ledger[0]
    assert tr_blocked.candidate_id == cand.candidate_id
    assert tr_blocked.status == "BLOCKED"
    assert RejectionReason.GOVERNANCE_BLOCKED in tr_blocked.rejection_reasons

    cand_blocked = result.research_candidates[0]
    assert cand_blocked.candidate_id == cand.candidate_id
    assert cand_blocked.promotion_status == PromotionStatus.REJECTED
    assert RejectionReason.GOVERNANCE_BLOCKED in cand_blocked.rejection_reasons

    rec_blocked = result.registry_records[0]
    assert rec_blocked.candidate_id == cand.candidate_id
    assert rec_blocked.status == RegistryStatus.REJECTED
    assert "GOVERNANCE_BLOCKED" in rec_blocked.rejection_reasons


def test_H_malformed_active_constraint_fails_closed(
    dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test H: Malformed or ambiguous active constraint fails closed."""
    cand = candidates[0]

    # Create a malformed constraint object bypassing standard init (e.g. invalid confidence_score)
    malformed_constraint = object.__new__(DoNotRepeatConstraint)
    object.__setattr__(malformed_constraint, "constraint_id", "c_malformed")
    object.__setattr__(malformed_constraint, "source_record_id", "rec_mal")
    object.__setattr__(malformed_constraint, "experiment_fingerprint", "")
    object.__setattr__(malformed_constraint, "evidence_fingerprint", None)
    object.__setattr__(malformed_constraint, "pattern_key", cand.candidate_id)
    object.__setattr__(malformed_constraint, "condition_description", "Malformed test")
    object.__setattr__(malformed_constraint, "reason", "Malformed confidence score")
    object.__setattr__(malformed_constraint, "confidence_score", 1.5)  # Invalid score (> 1.0)
    object.__setattr__(malformed_constraint, "is_active", True)
    object.__setattr__(malformed_constraint, "superseded_by_learning_id", None)
    object.__setattr__(malformed_constraint, "schema_version", "1.0")

    gov_res = evaluate_candidate_memory_governance(
        candidate=cand,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        active_constraints=(malformed_constraint,),
    )

    assert gov_res.decision == MemoryGovernanceDecision.FAIL_CLOSED
    assert "invalid confidence_score" in gov_res.reason


def test_I_same_inputs_same_memory_produce_identical_results(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test I: Identical inputs + identical memory state produce identical governance results and fingerprints."""
    constraint = DoNotRepeatConstraint(
        constraint_id="c_repeatable",
        source_record_id="rec_rep",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key=candidates[0].candidate_id,
        condition_description="Repeatability test",
        reason="Repeatability test reason",
        confidence_score=0.9,
        is_active=True,
    )

    engine = DiscoveryEngine()
    res1 = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=(constraint,),
    )

    res2 = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        memory_store=(constraint,),
    )

    assert len(res1.memory_governance_results) == len(res2.memory_governance_results)
    for g1, g2 in zip(res1.memory_governance_results, res2.memory_governance_results):
        assert g1.decision == g2.decision
        assert g1.governance_fingerprint == g2.governance_fingerprint


def test_J_K_ast_isolation_production_and_project2_unmodified():
    """Test J & K: Static AST isolation tests confirming production runtime and Project2 code are untouched."""
    root_dir = Path(__file__).resolve().parents[1]

    # J. Production / live execution files
    live_files = [
        root_dir / "app_live_trade_runtime.py",
        root_dir / "run_live_execution.py",
        root_dir / "src" / "evaluation" / "live_execution_runtime.py",
        root_dir / "src" / "evaluation" / "live_production_decision.py",
    ]
    for lf in live_files:
        if lf.exists():
            source = lf.read_text(encoding="utf-8")
            ast.parse(source)
            assert "evaluate_candidate_memory_governance" not in source, f"Memory governance leaked into production file {lf.name}"
            assert "DiscoveryMemoryGovernanceResult" not in source

    # K. Project2 integration publisher
    p2_publisher = root_dir / "src" / "integration" / "project2_publisher.py"
    if p2_publisher.exists():
        p2_source = p2_publisher.read_text(encoding="utf-8")
        ast.parse(p2_source)
        assert "evaluate_candidate_memory_governance" not in p2_source
        assert "DiscoveryMemoryGovernanceResult" not in p2_source


def test_L_M_existing_pr24_25_learning_persistence_and_engine_compatibility(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates, tmp_path
):
    """Test L & M: Existing PR #24/#25 learning persistence and DiscoveryEngine behavior remain backward-compatible."""
    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        persist_evidence=True,
        persist_registry_dir=tmp_path,
    )

    assert hasattr(result, "learning_records")
    assert hasattr(result, "registry_records")
    assert hasattr(result, "trial_ledger")
    assert hasattr(result, "memory_governance_results")
    assert len(result.learning_records) == len(candidates)


def test_O_registry_idempotency_and_provenance_guarantees(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, tmp_path
):
    """Test O: Registry idempotency and provenance guarantees remain intact with memory governance."""
    idem_candidates = (
        CandidateSpec(
            generator_name="test_idem_gen",
            generator_version="1.0.0",
            strategy_name="baseline_momentum",
            parameters={"fast_window": 3, "slow_window": 8},
            hypothesis_template="Test momentum fast with parameters fast={fast_window}",
        ),
        CandidateSpec(
            generator_name="test_idem_gen",
            generator_version="1.0.0",
            strategy_name="baseline_momentum",
            parameters={"fast_window": 10, "slow_window": 30},
            hypothesis_template="Test momentum slow with parameters slow={slow_window}",
        ),
    )

    constraint = DoNotRepeatConstraint(
        constraint_id="c_idempotent_test",
        source_record_id="rec_idem",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key=idem_candidates[0].candidate_id,
        condition_description="Idempotency test",
        reason="Idempotency test reason",
        confidence_score=0.9,
        is_active=True,
    )

    engine = DiscoveryEngine()

    run1 = engine.run_discovery(
        df=sample_market_data,
        candidates=idem_candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        persist_evidence=True,
        persist_registry_dir=tmp_path,
        memory_store=(constraint,),
    )

    run2 = engine.run_discovery(
        df=sample_market_data,
        candidates=idem_candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        persist_evidence=True,
        persist_registry_dir=tmp_path,
        memory_store=(constraint,),
    )

    assert len(run1.registry_records) == len(run2.registry_records)
    for r1, r2 in zip(run1.registry_records, run2.registry_records):
        assert r1.canonical_fingerprint == r2.canonical_fingerprint
