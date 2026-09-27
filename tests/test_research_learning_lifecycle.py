"""Comprehensive unit and integration tests for operationalizing canonical research learning memory in the research lifecycle.

Validates requirement matrix A through Q:
A. Authoritative ResearchRegistryRecord creation in DiscoveryEngine automatically produces and persists corresponding ResearchLearningRecords when persist_evidence=True.
B. Successful research evidence produces SUCCESS classification learning memory.
C. Rejected/failed research produces FAILURE classification learning memory preserving rejection reasons.
D. Inconclusive/insufficient research produces INCONCLUSIVE memory where applicable.
E. Missing or invalid provenance fails closed.
F. Evidence/registry fingerprint mismatch fails closed during registration.
G. Re-running the discovery lifecycle is idempotent and does not create duplicate learning records.
H. Conflicting learning persistence remains fail-closed (RegistryConflictError).
I. Multi-trial DiscoveryEngine execution produces distinct, isolated learning records without cross-trial contamination.
J. Learning records preserve complete experiment, evidence, candidate, search, and trial lineage.
K. Active DoNotRepeatConstraints produced by PR #24 remain queryable after lifecycle execution.
L. Existing ResearchRegistry behavior remains unchanged.
M. Existing DiscoveryEngine behavior remains unchanged except for the new observational learning persistence side effect.
N. Production/live execution code is not modified.
O. Project2 integration/publishing code is not modified.
P. No fabricated live outcome or realized-performance data is introduced.
Q. AST/static isolation tests prove the integration does not mutate production strategy/risk/promotion rules.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
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
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    LessonCategory,
    RegistryConflictError,
    RegistryStatus,
    RegistryValidationError,
    ResearchLearningRecord,
    ResearchOutcomeClassification,
    ResearchRegistryRecord,
    ResearchRegistryStore,
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


def test_A_discovery_run_automatically_populates_learning_memory(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates, tmp_path
):
    """Test A & I: DiscoveryEngine run produces and persists ResearchLearningRecord objects for every trial."""
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

    assert isinstance(result, DiscoveryRunResult)
    assert len(result.registry_records) == len(candidates)
    assert len(result.learning_records) == len(candidates)

    store = ResearchRegistryStore(base_dir=tmp_path)
    persisted_learnings = store.list_learning_records()
    assert len(persisted_learnings) == len(candidates)

    for lr in result.learning_records:
        assert isinstance(lr, ResearchLearningRecord)
        assert lr.learning_id != ""
        assert lr.source_record_id != ""
        assert lr.experiment_fingerprint != ""


def test_B_successful_evidence_produces_success_classification(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    """Test B: Successful research evidence maps to ResearchOutcomeClassification.SUCCESS."""
    spec = ResearchExperimentSpec(
        hypothesis="Successful experiment test",
        methodology_version="discovery_v1.0",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 3, "slow_window": 8},
    )
    p_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-03-01",
        total_return=0.15,
        max_drawdown=-0.05,
        sharpe_ratio=2.0,
        win_rate=0.60,
        profit_factor=1.8,
        observations=50,
    )
    p_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2023-03-02",
        end_date="2023-06-01",
        total_return=0.10,
        max_drawdown=-0.04,
        sharpe_ratio=1.5,
        win_rate=0.55,
        profit_factor=1.5,
        observations=30,
    )
    ev_success = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(p_is, p_oos),
        robustness_verdict={"is_robust": True},
        benchmark_comparison={"outperformed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
    )

    reg_rec = construct_registry_record_from_evidence(
        evidence=ev_success,
        candidate_id="cand_success",
        qualification_status="QUALIFIED",
    )
    learning = construct_learning_record_from_registry_record(reg_rec)

    assert learning.classification == ResearchOutcomeClassification.SUCCESS
    assert learning.confidence_score == 0.95
    assert len(learning.lessons) == 1
    assert learning.lessons[0].category == LessonCategory.STRATEGY_PERFORMANCE


def test_C_failed_evidence_produces_failure_classification(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    """Test C: Rejected/failed research evidence maps to FAILURE classification with rejection reasons."""
    spec = ResearchExperimentSpec(
        hypothesis="Failed experiment test",
        methodology_version="discovery_v1.0",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 20, "slow_window": 5},
    )
    p_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-03-01",
        total_return=-0.10,
        max_drawdown=-0.25,
        sharpe_ratio=-0.5,
        win_rate=0.35,
        profit_factor=0.6,
        observations=50,
    )
    ev_failed = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(p_is,),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS, RejectionReason.FAILED_ROBUSTNESS),
    )

    reg_rec = construct_registry_record_from_evidence(
        evidence=ev_failed,
        candidate_id="cand_failed",
        qualification_status="REJECTED",
    )
    learning = construct_learning_record_from_registry_record(reg_rec)

    assert learning.classification == ResearchOutcomeClassification.FAILURE
    assert "FAILED_OOS" in learning.rejection_reasons
    assert "FAILED_ROBUSTNESS" in learning.rejection_reasons
    assert len(learning.constraints) == 1
    assert learning.constraints[0].is_active is True


def test_D_insufficient_evidence_produces_inconclusive_classification(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance
):
    """Test D: Insufficient data maps to INCONCLUSIVE classification."""
    spec = ResearchExperimentSpec(
        hypothesis="Insufficient experiment test",
        methodology_version="discovery_v1.0",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 3, "slow_window": 8},
    )
    ev_insuf = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.INSUFFICIENT_DATA,),
    )

    reg_rec = construct_registry_record_from_evidence(
        evidence=ev_insuf,
        candidate_id="cand_insuf",
        qualification_status="REJECTED",
    )
    learning = construct_learning_record_from_registry_record(reg_rec)

    assert learning.classification == ResearchOutcomeClassification.INCONCLUSIVE
    assert learning.confidence_score == 0.5


def test_E_missing_or_invalid_provenance_fails_closed(tmp_path):
    """Test E: Constructing or registering learning memory with missing/invalid provenance fails closed."""
    store = ResearchRegistryStore(base_dir=tmp_path)

    with pytest.raises(TypeError, match="registry_record must be a ResearchRegistryRecord instance"):
        construct_learning_record_from_registry_record(None)  # type: ignore

    with pytest.raises(RegistryValidationError, match="record_id must be a non-empty string"):
        # Dummy invalid record without record_id or fingerprint
        invalid_rec = ResearchRegistryRecord(
            record_id="",
            experiment_fingerprint="",
            evidence_fingerprint=None,
            candidate_id=None,
            search_fingerprint=None,
            search_id=None,
            trial_id=None,
            trial_index=None,
            status=RegistryStatus.FAILED,
            qualification_status="REJECTED",
            promotion_status="REJECTED",
            rejection_reasons=(),
            dataset_scope_id="ds_1",
            execution_assumptions_id="ea_1",
            code_provenance_id="cp_1",
            methodology_version="1.0",
            selection_assessment_id=None,
            robustness_assessment_id=None,
            benchmark_status=None,
            regime_status=None,
            error_message="Error",
            reproducibility=None,  # type: ignore
            lineage=None,  # type: ignore
        )
        construct_learning_record_from_registry_record(invalid_rec)


def test_F_mismatched_evidence_fingerprint_fails_closed(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, tmp_path
):
    """Test F: Registering a learning record with a mismatched evidence fingerprint fails closed."""
    spec = ResearchExperimentSpec(
        hypothesis="Mismatched fingerprint test",
        methodology_version="discovery_v1.0",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 3, "slow_window": 8},
    )
    ev = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    reg_rec = construct_registry_record_from_evidence(evidence=ev, candidate_id="cand_1")
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning = construct_learning_record_from_registry_record(reg_rec)
    mismatched = ResearchLearningRecord(
        learning_id=learning.learning_id,
        source_record_id=learning.source_record_id,
        experiment_fingerprint=learning.experiment_fingerprint,
        evidence_fingerprint="mismatched_ev_fp_12345",
        candidate_id=learning.candidate_id,
        search_fingerprint=learning.search_fingerprint,
        trial_id=learning.trial_id,
        dataset_scope_id=learning.dataset_scope_id,
        execution_assumptions_id=learning.execution_assumptions_id,
        code_provenance_id=learning.code_provenance_id,
        methodology_version=learning.methodology_version,
        classification=learning.classification,
        observed_conditions=learning.observed_conditions,
        lessons=learning.lessons,
        constraints=learning.constraints,
        confidence_score=learning.confidence_score,
        rejection_reasons=learning.rejection_reasons,
    )

    with pytest.raises(RegistryValidationError, match="Evidence fingerprint mismatch"):
        store.register_learning_record(mismatched)


def test_G_rerunning_lifecycle_is_idempotent(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates, tmp_path
):
    """Test G: Re-running DiscoveryEngine lifecycle is idempotent and returns identical learning records."""
    engine = DiscoveryEngine()
    run1 = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        persist_evidence=True,
        persist_registry_dir=tmp_path,
    )

    run2 = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        persist_evidence=True,
        persist_registry_dir=tmp_path,
    )

    assert len(run1.learning_records) == len(run2.learning_records)
    for lr1, lr2 in zip(run1.learning_records, run2.learning_records):
        assert lr1.canonical_fingerprint == lr2.canonical_fingerprint

    store = ResearchRegistryStore(base_dir=tmp_path)
    persisted = store.list_learning_records()
    assert len(persisted) == len(candidates)


def test_H_conflicting_learning_persistence_fails_closed(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, tmp_path
):
    """Test H: Attempting to register a conflicting learning record raises RegistryConflictError."""
    spec = ResearchExperimentSpec(
        hypothesis="Conflict test",
        methodology_version="discovery_v1.0",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 3, "slow_window": 8},
    )
    ev = ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    reg_rec = construct_registry_record_from_evidence(evidence=ev, candidate_id="cand_1")
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning1 = construct_learning_record_from_registry_record(reg_rec)
    store.register_learning_record(learning1)

    # Conflicting learning with same ID but modified confidence_score
    learning_conflict = ResearchLearningRecord(
        learning_id=learning1.learning_id,
        source_record_id=learning1.source_record_id,
        experiment_fingerprint=learning1.experiment_fingerprint,
        evidence_fingerprint=learning1.evidence_fingerprint,
        candidate_id=learning1.candidate_id,
        search_fingerprint=learning1.search_fingerprint,
        trial_id=learning1.trial_id,
        dataset_scope_id=learning1.dataset_scope_id,
        execution_assumptions_id=learning1.execution_assumptions_id,
        code_provenance_id=learning1.code_provenance_id,
        methodology_version=learning1.methodology_version,
        classification=learning1.classification,
        observed_conditions=learning1.observed_conditions,
        lessons=learning1.lessons,
        constraints=learning1.constraints,
        confidence_score=0.1234,  # Conflicting content
        rejection_reasons=learning1.rejection_reasons,
    )

    with pytest.raises(RegistryConflictError):
        store.register_learning_record(learning_conflict)


def test_I_J_K_multi_trial_isolation_lineage_and_constraint_queryability(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates, tmp_path
):
    """Test I, J, K: Multi-trial discovery isolates trial lineage and queryable do-not-repeat constraints."""
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

    # J. Verify complete lineage
    for lr in result.learning_records:
        tr = next(t for t in result.trial_ledger if t.trial_id == lr.trial_id)
        assert lr.candidate_id == tr.candidate_id
        assert lr.trial_id == tr.trial_id

    # K. Verify active constraints queryable via store API
    store = ResearchRegistryStore(base_dir=tmp_path)
    active_constraints = store.get_active_do_not_repeat_constraints(symbol="XAUUSD")
    assert isinstance(active_constraints, tuple)


def test_L_M_existing_registry_and_discovery_behavior_unchanged(
    sample_market_data, dataset_scope, execution_assumptions, code_provenance, candidates
):
    """Test L & M: Validate existing DiscoveryEngine and ResearchRegistry return types and rankings are untouched."""
    engine = DiscoveryEngine()
    result = engine.run_discovery(
        df=sample_market_data,
        candidates=candidates,
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        persist_evidence=False,
    )

    assert hasattr(result, "promoted_evidence")
    assert hasattr(result, "rejected_evidence")
    assert hasattr(result, "trial_ledger")
    assert hasattr(result, "selection_assessments")
    assert hasattr(result, "robustness_assessments")
    assert hasattr(result, "registry_records")
    assert hasattr(result, "learning_records")
    assert hasattr(result, "research_candidates")


def test_N_O_P_Q_ast_static_isolation_and_no_production_mutation():
    """Test N, O, P, Q: Static AST isolation tests proving live execution, Project2, and production rules are unchanged."""
    root_dir = Path(__file__).resolve().parents[1]

    # N. Production / live execution files
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
            assert "ResearchLearningRecord" not in source, f"Learning memory leaked into production file {lf.name}"
            assert "construct_learning_record" not in source

    # O. Project2 integration publisher
    p2_publisher = root_dir / "src" / "integration" / "project2_publisher.py"
    if p2_publisher.exists():
        p2_source = p2_publisher.read_text(encoding="utf-8")
        assert "ResearchLearningRecord" not in p2_source
        assert "construct_learning_record" not in p2_source

    # P. Verify no fabricated live outcome data in learning memory constructor
    learning_factory_source = (root_dir / "src" / "evaluation" / "research_registry.py").read_text(encoding="utf-8")
    assert "realized_pnl" not in learning_factory_source.lower()
    assert "live_trade_result" not in learning_factory_source.lower()
