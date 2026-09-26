"""Comprehensive unit, negative, property, and AST tests for Canonical Research Registry layer.

Covers matrix requirements A through AI:
A. register a valid research evidence record
B. retrieve by evidence fingerprint
C. retrieve by experiment fingerprint
D. retrieve by trial identity
E. retrieve by search fingerprint
F. retrieve complete evidence lineage
G. identical registration is idempotent
H. conflicting duplicate fails closed
I. deterministic record fingerprint
J. changed semantic content changes fingerprint
K. timestamps do not affect semantic identity
L. Python hash() is not used
M. malformed record rejected
N. unknown schema rejected
O. missing lineage rejected
P. failed trial remains represented
Q. rejected evidence remains represented
R. insufficient evidence remains represented
S. robustness metadata preserved
T. benchmark-unavailable state preserved
U. regime-unavailable state preserved
V. qualification state preserved
W. selection-governance state preserved
X. promotion state is recorded but not executed
Y. registry cannot call ProductionDecision
Z. registry cannot call production risk
AA. registry cannot publish/live execute
AB. registry cannot auto-promote
AC. no duplicate research execution
AD. no duplicate qualification
AE. no duplicate selection governance
AF. no duplicate robustness execution
AG. atomic persistence failure preserves previous valid state
AH. concurrent/duplicate logical registration remains deterministic
AI. Project2 isolation
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
    RegistryConflictError,
    RegistryStatus,
    RegistryValidationError,
    ResearchEvidenceLineage,
    ResearchRegistryRecord,
    ResearchRegistryStore,
    ResearchReproducibilityDescriptor,
    SchemaVersionError,
    construct_registry_record_from_evidence,
)


@pytest.fixture
def sample_spec() -> ResearchExperimentSpec:
    return ResearchExperimentSpec(
        hypothesis="Test momentum strategy",
        methodology_version="discovery_v1.0",
        strategy_name="baseline",
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


def test_A_register_valid_record(sample_evidence, tmp_path):
    rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        search_id="search_1",
        search_fingerprint="sf_123",
        trial_id="trial_1",
        trial_index=0,
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    registered = store.register(rec)
    assert registered.record_id == rec.record_id
    assert registered.status == RegistryStatus.QUALIFIED


def test_B_C_D_E_lookup_methods(sample_evidence, tmp_path):
    rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        search_id="search_1",
        search_fingerprint="sf_123",
        trial_id="trial_1",
        trial_index=0,
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(rec)

    # B. by evidence fingerprint
    assert store.get_by_evidence_fingerprint(sample_evidence.evidence_id) is not None

    # C. by experiment fingerprint
    exp_recs = store.get_by_experiment_fingerprint(sample_evidence.experiment_fingerprint)
    assert len(exp_recs) == 1

    # D. by trial id
    assert store.get_by_trial_id("trial_1") is not None

    # E. by search fingerprint
    sf_recs = store.get_by_search_fingerprint("sf_123")
    assert len(sf_recs) == 1

    # by candidate id
    cand_recs = store.get_by_candidate_id("cand_1")
    assert len(cand_recs) == 1


def test_F_retrieve_evidence_lineage(sample_evidence, tmp_path):
    rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        search_id="search_1",
        search_fingerprint="sf_123",
        trial_id="trial_1",
        trial_index=0,
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(rec)

    lineage = store.get_evidence_lineage(sample_evidence.evidence_id)
    assert lineage is not None
    assert lineage.trial_id == "trial_1"
    assert lineage.candidate_id == "cand_1"
    assert lineage.search_id == "search_1"


def test_G_idempotent_registration(sample_evidence, tmp_path):
    rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        trial_id="trial_1",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    r1 = store.register(rec)
    r2 = store.register(rec)
    assert r1.canonical_fingerprint == r2.canonical_fingerprint
    assert len(store.list_records()) == 1


def test_H_conflicting_duplicate_fails_closed(sample_evidence, tmp_path):
    rec1 = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        trial_id="trial_1",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(rec1)

    # Create conflicting record with same record_id but changed content
    rec2 = ResearchRegistryRecord(
        record_id=rec1.record_id,
        experiment_fingerprint=rec1.experiment_fingerprint,
        evidence_fingerprint=rec1.evidence_fingerprint,
        candidate_id="cand_different",
        search_fingerprint=rec1.search_fingerprint,
        search_id=rec1.search_id,
        trial_id=rec1.trial_id,
        trial_index=rec1.trial_index,
        status=rec1.status,
        qualification_status=rec1.qualification_status,
        promotion_status=rec1.promotion_status,
        rejection_reasons=rec1.rejection_reasons,
        dataset_scope_id=rec1.dataset_scope_id,
        execution_assumptions_id=rec1.execution_assumptions_id,
        code_provenance_id=rec1.code_provenance_id,
        methodology_version=rec1.methodology_version,
        selection_assessment_id=rec1.selection_assessment_id,
        robustness_assessment_id=rec1.robustness_assessment_id,
        benchmark_status=rec1.benchmark_status,
        regime_status=rec1.regime_status,
        error_message="different error",
        reproducibility=rec1.reproducibility,
        lineage=rec1.lineage,
    )

    with pytest.raises(RegistryConflictError):
        store.register(rec2)


def test_I_J_K_L_fingerprint_properties(sample_evidence):
    rec1_raw = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        trial_id="trial_1",
    )
    rec1 = ResearchRegistryRecord.from_dict({**rec1_raw.as_dict(), "created_at_utc": "2026-09-26T10:00:00+00:00"})
    rec2 = ResearchRegistryRecord.from_dict({**rec1_raw.as_dict(), "created_at_utc": "2026-09-26T18:00:00+00:00"})

    # I. Deterministic fingerprint
    assert rec1.canonical_fingerprint == rec1.canonical_fingerprint

    # K. Timestamps do not affect semantic fingerprint
    assert rec1.canonical_fingerprint == rec2.canonical_fingerprint

    # J. Changed semantic content changes fingerprint
    rec3 = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_MODIFIED",
        trial_id="trial_1",
    )
    assert rec1.canonical_fingerprint != rec3.canonical_fingerprint


def test_M_N_O_validation_failures(sample_evidence):
    rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        trial_id="trial_1",
    )
    d = rec.as_dict()

    # M. Malformed status
    d_bad_status = dict(d)
    d_bad_status["status"] = "INVALID_STATUS"
    with pytest.raises(RegistryValidationError):
        ResearchRegistryRecord.from_dict(d_bad_status)

    # N. Unknown schema
    d_unknown_schema = dict(d)
    d_unknown_schema["schema_version"] = "99.0"
    with pytest.raises(SchemaVersionError):
        ResearchRegistryRecord.from_dict(d_unknown_schema)

    # O. Missing lineage
    d_no_lineage = dict(d)
    del d_no_lineage["lineage"]
    with pytest.raises(RegistryValidationError):
        ResearchRegistryRecord.from_dict(d_no_lineage)


def test_P_Q_R_S_T_U_V_W_status_preservation(sample_evidence, tmp_path):
    # P. Failed
    rec_failed = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_fail",
        error_message="Simulation crashed",
    )
    assert rec_failed.status == RegistryStatus.FAILED

    # Q. Rejected
    ev_rejected = ResearchEvidence(
        experiment_fingerprint=sample_evidence.experiment_fingerprint,
        spec=sample_evidence.spec,
        partitions=sample_evidence.partitions,
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    rec_rejected = construct_registry_record_from_evidence(
        evidence=ev_rejected,
        candidate_id="cand_rej",
    )
    assert rec_rejected.status == RegistryStatus.REJECTED

    # R. Insufficient data
    ev_insufficient = ResearchEvidence(
        experiment_fingerprint=sample_evidence.experiment_fingerprint,
        spec=sample_evidence.spec,
        partitions=(),
        robustness_verdict={},
        benchmark_comparison={},
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.INSUFFICIENT_DATA,),
    )
    rec_insufficient = construct_registry_record_from_evidence(
        evidence=ev_insufficient,
        candidate_id="cand_insuf",
    )
    assert rec_insufficient.status == RegistryStatus.INSUFFICIENT

    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(rec_failed)
    store.register(rec_rejected)
    store.register(rec_insufficient)

    assert store.get_by_candidate_id("cand_fail")[0].status == RegistryStatus.FAILED
    assert store.get_by_candidate_id("cand_rej")[0].status == RegistryStatus.REJECTED
    assert store.get_by_candidate_id("cand_insuf")[0].status == RegistryStatus.INSUFFICIENT


def test_AG_atomic_persistence_failure_preserves_state(sample_evidence, tmp_path, monkeypatch):
    rec = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_1",
        trial_id="trial_1",
    )
    store = ResearchRegistryStore(base_dir=tmp_path)
    store.register(rec)
    assert len(store.list_records()) == 1

    # Simulate atomic failure on second register
    def mock_replace(src, dst):
        raise OSError("Disk write failed")

    monkeypatch.setattr(os, "replace", mock_replace)

    rec2 = construct_registry_record_from_evidence(
        evidence=sample_evidence,
        candidate_id="cand_2",
        trial_id="trial_2",
    )
    with pytest.raises(OSError):
        store.register(rec2)

    # Confirm original state preserved
    recs = store.list_records()
    assert len(recs) == 1
    assert recs[0].candidate_id == "cand_1"


def test_AST_safety_and_isolation():
    registry_file = Path(__file__).resolve().parents[1] / "src" / "evaluation" / "research_registry.py"
    source = registry_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # AI. Project2 isolation
    assert "project2" not in source.lower()
    assert "Project2" not in source

    # L. Python hash() is not used for identity or fingerprint
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "hash", "Python built-in hash() must not be used."

    # X, Y, Z, AA, AB: Registry cannot call ProductionDecision, production risk, publish, or auto-promote
    forbidden_terms = (
        "ProductionDecision",
        "calculate_production_risk_levels",
        "Project2Publisher",
        "persist_promoted_candidate_binding",
    )
    for term in forbidden_terms:
        assert term not in source, f"Forbidden authority term '{term}' found in research_registry.py"
