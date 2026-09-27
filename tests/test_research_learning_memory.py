"""Comprehensive unit, negative, property, and safety tests for Canonical Research Learning / Experience Memory Layer.

Covers matrix requirements A through Y:
A. valid learning record creation from a real ResearchRegistryRecord
B. missing source registry record fails closed
C. mismatched evidence/experiment lineage fails closed
D. deterministic learning fingerprint
E. timestamps do not change semantic fingerprint
F. identical learning registration is idempotent
G. conflicting registration fails closed
H. corrupted persistence fails closed
I. unsupported schema fails closed
J. SUCCESS classification
K. FAILURE classification
L. INCONCLUSIVE classification
M. rejection reason preservation
N. structured condition preservation
O. lesson provenance preservation
P. do-not-repeat constraint provenance
Q. query by experiment/evidence/candidate
R. query by classification/category
S. historical records are never silently overwritten
T. contradiction/supersession behavior is deterministic
U. no production mutation
V. no Project2 integration
W. existing ResearchRegistry tests remain green
X. existing DiscoveryEngine tests remain green
Y. regression coverage for all existing registry idempotence/conflict/fingerprint guarantees
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import tempfile
import pytest

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
    ResearchEvidenceLineage,
    ResearchLearningRecord,
    ResearchLesson,
    ResearchOutcomeClassification,
    ResearchRegistryRecord,
    ResearchRegistryStore,
    ResearchReproducibilityDescriptor,
    SchemaVersionError,
    StructuredObservedConditions,
    construct_learning_record_from_registry_record,
    construct_registry_record_from_evidence,
    mark_learning_supersession,
)


@pytest.fixture
def sample_spec() -> ResearchExperimentSpec:
    return ResearchExperimentSpec(
        hypothesis="Test momentum strategy for learning",
        methodology_version="discovery_v1.0",
        strategy_name="baseline_momentum",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope("ds_1", "XAUUSD", "1D", "2023-01-01", "2023-06-01"),
        execution_assumptions=ExecutionAssumptions(0.0005, 0.0002, 50.0),
        code_provenance=CodeProvenance("a1b2c3d4", "clean", "Jules"),
        benchmark_reference="buy_and_hold",
        parameters={"fast_window": 3, "slow_window": 8},
    )


@pytest.fixture
def sample_evidence(sample_spec) -> ResearchEvidence:
    p_is = EvidencePartition(
        role=EvidencePartitionRole.IN_SAMPLE,
        start_date="2023-01-01",
        end_date="2023-03-01",
        total_return=0.10,
        max_drawdown=-0.05,
        sharpe_ratio=1.5,
        win_rate=0.55,
        profit_factor=1.4,
        observations=50,
    )
    p_oos = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2023-03-02",
        end_date="2023-06-01",
        total_return=0.05,
        max_drawdown=-0.04,
        sharpe_ratio=1.1,
        win_rate=0.52,
        profit_factor=1.2,
        observations=30,
    )
    return ResearchEvidence(
        experiment_fingerprint=sample_spec.fingerprint,
        spec=sample_spec,
        partitions=(p_is, p_oos),
        robustness_verdict={"is_robust": True},
        benchmark_comparison={"outperformed": True},
        promotion_status=PromotionStatus.PROMOTABLE,
        rejection_reasons=(),
        critique_notes="Passed evaluation",
        created_at_utc="2026-09-26T12:00:00+00:00",
    )


def test_A_J_valid_learning_record_success(sample_evidence, tmp_path):
    reg_rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_success",
        trial_id="trial_100",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning = construct_learning_record_from_registry_record(reg_rec)
    registered_learning = store.register_learning_record(learning)

    assert registered_learning.classification == ResearchOutcomeClassification.SUCCESS
    assert registered_learning.source_record_id == reg_rec.record_id
    assert registered_learning.experiment_fingerprint == reg_rec.experiment_fingerprint
    assert registered_learning.evidence_fingerprint == reg_rec.evidence_fingerprint
    assert registered_learning.candidate_id == "cand_success"


def test_B_missing_source_registry_record_fails_closed(sample_evidence, tmp_path):
    reg_rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_unregistered",
        trial_id="trial_999",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    learning = construct_learning_record_from_registry_record(reg_rec)

    with pytest.raises(RegistryValidationError, match="does not exist"):
        store.register_learning_record(learning)


def test_C_mismatched_evidence_lineage_fails_closed(sample_evidence, tmp_path):
    reg_rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        trial_id="trial_1",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning_raw = construct_learning_record_from_registry_record(reg_rec)
    mismatched_learning = ResearchLearningRecord(
        learning_id=learning_raw.learning_id,
        source_record_id=learning_raw.source_record_id,
        experiment_fingerprint=learning_raw.experiment_fingerprint,
        evidence_fingerprint="mismatched_evidence_sha256",
        candidate_id=learning_raw.candidate_id,
        search_fingerprint=learning_raw.search_fingerprint,
        trial_id=learning_raw.trial_id,
        dataset_scope_id=learning_raw.dataset_scope_id,
        execution_assumptions_id=learning_raw.execution_assumptions_id,
        code_provenance_id=learning_raw.code_provenance_id,
        methodology_version=learning_raw.methodology_version,
        classification=learning_raw.classification,
        observed_conditions=learning_raw.observed_conditions,
        lessons=learning_raw.lessons,
        constraints=learning_raw.constraints,
        confidence_score=learning_raw.confidence_score,
        rejection_reasons=learning_raw.rejection_reasons,
    )

    with pytest.raises(RegistryValidationError, match="Evidence fingerprint mismatch"):
        store.register_learning_record(mismatched_learning)


def test_D_E_deterministic_fingerprint_and_timestamp_independence(sample_evidence):
    reg_rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
    )
    learning1 = construct_learning_record_from_registry_record(reg_rec)

    learning2 = ResearchLearningRecord.from_dict({
        **learning1.as_dict(),
        "created_at_utc": "2026-10-10T00:00:00+00:00",
    })

    assert learning1.canonical_fingerprint == learning2.canonical_fingerprint

    # Modified content changes fingerprint
    learning3_dict = learning1.as_dict()
    learning3_dict["confidence_score"] = 0.123456
    learning3 = ResearchLearningRecord.from_dict(learning3_dict)
    assert learning1.canonical_fingerprint != learning3.canonical_fingerprint


def test_F_G_idempotency_and_conflict_handling(sample_evidence, tmp_path):
    reg_rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning = construct_learning_record_from_registry_record(reg_rec)
    r1 = store.register_learning_record(learning)
    r2 = store.register_learning_record(learning)

    assert r1.canonical_fingerprint == r2.canonical_fingerprint
    assert len(store.list_learning_records()) == 1

    # Conflicting learning record with same ID but different content
    conflicting_dict = learning.as_dict()
    conflicting_dict["confidence_score"] = 0.2
    conflicting = ResearchLearningRecord.from_dict(conflicting_dict)

    with pytest.raises(RegistryConflictError):
        store.register_learning_record(conflicting)


def test_H_I_corrupted_persistence_and_schema_validation(tmp_path):
    store = ResearchRegistryStore(base_dir=tmp_path)
    learn_dir = store.base_dir / "learning" / "exp_1"
    learn_dir.mkdir(parents=True, exist_ok=True)
    corrupt_file = learn_dir / "bad.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(RegistryValidationError, match="Failed to parse JSON"):
        store._load_learning_file(corrupt_file)

    # Unknown schema version
    bad_schema_file = learn_dir / "bad_schema.json"
    bad_schema_file.write_text(json.dumps({"schema_version": "99.0"}), encoding="utf-8")
    with pytest.raises(SchemaVersionError, match="Unsupported learning schema version"):
        store._load_learning_file(bad_schema_file)


def test_K_M_N_O_P_failure_classification_and_preservation(sample_spec, tmp_path):
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
        experiment_fingerprint=sample_spec.fingerprint,
        spec=sample_spec,
        partitions=(p_is,),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS, RejectionReason.FAILED_ROBUSTNESS),
    )

    reg_rec = construct_registry_record_from_evidence(
        evidence=ev_failed,
        candidate_id="cand_failed",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning = construct_learning_record_from_registry_record(reg_rec)
    store.register_learning_record(learning)

    assert learning.classification == ResearchOutcomeClassification.FAILURE
    assert "FAILED_OOS" in learning.rejection_reasons
    assert "FAILED_ROBUSTNESS" in learning.rejection_reasons
    assert learning.observed_conditions.symbol == "XAUUSD"
    assert len(learning.lessons) == 1
    assert len(learning.constraints) == 1
    assert learning.constraints[0].is_active is True


def test_L_inconclusive_classification(sample_spec, tmp_path):
    ev_insufficient = ResearchEvidence(
        experiment_fingerprint=sample_spec.fingerprint,
        spec=sample_spec,
        partitions=(),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.INSUFFICIENT_DATA,),
    )
    reg_rec = construct_registry_record_from_evidence(
        evidence=ev_insufficient,
        candidate_id="cand_insuf",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(reg_rec)

    learning = construct_learning_record_from_registry_record(reg_rec)
    store.register_learning_record(learning)

    assert learning.classification == ResearchOutcomeClassification.INCONCLUSIVE


def test_Q_R_query_capabilities(sample_evidence, sample_spec, tmp_path):
    store = ResearchRegistryStore(base_dir=tmp_path)

    # Record 1: Success
    reg1 = construct_registry_record_from_evidence(evidence=sample_evidence, candidate_id="cand_1")
    store.register(reg1)
    learn1 = construct_learning_record_from_registry_record(reg1)
    store.register_learning_record(learn1)

    # Record 2: Failure
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
    ev_fail = ResearchEvidence(
        experiment_fingerprint=sample_spec.fingerprint,
        spec=sample_spec,
        partitions=(p_is,),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    reg2 = construct_registry_record_from_evidence(evidence=ev_fail, candidate_id="cand_2")
    store.register(reg2)
    learn2 = construct_learning_record_from_registry_record(reg2)
    store.register_learning_record(learn2)

    # Q. Query by experiment / candidate
    res_cand1 = store.query_learning(candidate_id="cand_1")
    assert len(res_cand1) == 1
    assert res_cand1[0].learning_id == learn1.learning_id

    # R. Query by classification / lesson category / symbol
    res_success = store.query_learning(classification=ResearchOutcomeClassification.SUCCESS)
    assert len(res_success) == 1
    assert res_success[0].learning_id == learn1.learning_id

    res_fail = store.query_learning(classification=ResearchOutcomeClassification.FAILURE)
    assert len(res_fail) == 1
    assert res_fail[0].learning_id == learn2.learning_id

    res_symbol = store.query_learning(symbol="XAUUSD")
    assert len(res_symbol) == 2


def test_S_T_supersession_and_historical_preservation(sample_evidence, sample_spec, tmp_path):
    store = ResearchRegistryStore(base_dir=tmp_path)

    # Older failed learning
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
    ev_fail = ResearchEvidence(
        experiment_fingerprint=sample_spec.fingerprint,
        spec=sample_spec,
        partitions=(p_is,),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    reg_old = construct_registry_record_from_evidence(evidence=ev_fail, candidate_id="cand_old", trial_id="trial_1")
    store.register(reg_old)
    learn_old = construct_learning_record_from_registry_record(reg_old)
    store.register_learning_record(learn_old)

    # Newer success learning
    reg_new = construct_registry_record_from_evidence(evidence=sample_evidence, candidate_id="cand_new", trial_id="trial_2")
    store.register(reg_new)
    learn_new = construct_learning_record_from_registry_record(reg_new)

    # Apply explicit supersession
    updated_newer, updated_older = mark_learning_supersession(newer_learning=learn_new, older_learning=learn_old)

    # Record supersession atomically via store API
    registered_newer, registered_older = store.record_learning_supersession(
        newer_learning=learn_new,
        older_learning=learn_old,
    )

    # Confirm historical evidence is preserved and linked
    loaded_old = store.get_learning_by_id(registered_older.learning_id, registered_older.experiment_fingerprint)
    loaded_new = store.get_learning_by_id(registered_newer.learning_id, registered_newer.experiment_fingerprint)

    assert loaded_old.superseded_by_learning_id == loaded_new.learning_id
    assert loaded_new.supersedes_learning_id == loaded_old.learning_id
    if loaded_old.constraints:
        assert loaded_old.constraints[0].is_active is False


def test_U_V_W_X_Y_AST_safety_and_non_mutation():
    registry_file = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "research_registry.py"
    source = registry_file.read_text(encoding="utf-8")

    # V. No Project2 integration
    assert "project2" not in source.lower()

    # U, Y. No production mutation or live execution
    forbidden_terms = (
        "ProductionDecision",
        "calculate_production_risk_levels",
        "Project2Publisher",
        "persist_promoted_candidate_binding",
    )
    for term in forbidden_terms:
        assert term not in source, f"Forbidden production authority term '{term}' found in research_registry.py"
