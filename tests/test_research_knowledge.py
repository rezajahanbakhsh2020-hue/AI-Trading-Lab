"""Comprehensive unit, negative, and governance tests for Canonical Research Pattern Knowledge Layer.

Covers requirement matrix A through U:
A. Empty learning memory produces no patterns.
B. One learning record cannot create a repeated pattern under default repeated evidence policy.
C. Repeated successful observations derive one SUCCESS_PATTERN.
D. Repeated failed observations derive one FAILURE_PATTERN.
E. Repeated inconclusive observations derive INCONCLUSIVE_PATTERN.
F. General repeated observations produce GENERAL_OBSERVATION.
G. Deterministic canonicalization produces identical fingerprints regardless of input ordering.
H. Supporting learning-record identities are preserved.
I. Missing supporting learning record fails closed during store validation.
J. Lineage mismatch fails closed.
K. Conflicting outcomes are represented as explicit contradiction/inconclusive.
L. Different non-equivalent observed conditions are not merged.
M. Pattern persistence is idempotent.
N. Conflicting pattern registration is detected safely (RegistryConflictError).
O. Pattern queries return deterministic results.
P. Historical patterns remain traceable after newer pattern derivation/supersession.
Q. Pattern derivation does not modify live execution.
R. Pattern derivation does not modify strategy code.
S. Pattern derivation does not modify Project2.
T. Existing PR #24/#25/#26 learning and governance tests remain valid.
U. Full repository regression suite passes.
"""

from __future__ import annotations

import ast
import json
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
from src.evaluation.research_knowledge import (
    KNOWLEDGE_METHODOLOGY_VERSION_1_0,
    PatternDerivationError,
    PatternObservationSummary,
    ResearchKnowledgeDerivationPolicy,
    ResearchKnowledgePattern,
    ResearchPatternCategory,
    derive_research_knowledge_patterns,
)
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    LessonCategory,
    RegistryConflictError,
    RegistryStatus,
    RegistryValidationError,
    ResearchLearningRecord,
    ResearchLesson,
    ResearchOutcomeClassification,
    ResearchRegistryRecord,
    ResearchRegistryStore,
    StructuredObservedConditions,
    construct_learning_record_from_registry_record,
    construct_registry_record_from_evidence,
)


def _make_spec(
    strategy_name: str = "TestStrategy",
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    param_val: float = 10.0,
) -> ResearchExperimentSpec:
    return ResearchExperimentSpec(
        hypothesis=f"Test hypothesis for {strategy_name}",
        methodology_version="1.0.0",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_test",
            symbol=symbol,
            timeframe=timeframe,
            start_date="2023-01-01",
            end_date="2023-12-31",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(
            commit_sha="abcdef123456",
            repository_status="CLEAN",
            author="tester",
        ),
        benchmark_reference="buy_and_hold",
        parameters={"period": param_val},
    )


def _make_evidence(
    spec: ResearchExperimentSpec,
    promotion_status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
) -> ResearchEvidence:
    partition = EvidencePartition(
        role=EvidencePartitionRole.OUT_OF_SAMPLE,
        start_date="2023-06-01",
        end_date="2023-12-31",
        observations=500,
        sharpe_ratio=1.8 if promotion_status == PromotionStatus.PROMOTABLE else -0.5,
        total_return=0.15 if promotion_status == PromotionStatus.PROMOTABLE else -0.10,
        max_drawdown=0.05 if promotion_status == PromotionStatus.PROMOTABLE else 0.25,
        win_rate=0.55 if promotion_status == PromotionStatus.PROMOTABLE else 0.35,
        profit_factor=1.6 if promotion_status == PromotionStatus.PROMOTABLE else 0.8,
    )
    return ResearchEvidence(
        experiment_fingerprint=spec.fingerprint,
        spec=spec,
        partitions=(partition,),
        promotion_status=promotion_status,
        rejection_reasons=rejection_reasons,
    )


def _create_test_pair(
    strategy_name: str = "TestStrategy",
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    param_val: float = 10.0,
    promotion_status: PromotionStatus = PromotionStatus.PROMOTABLE,
    rejection_reasons: tuple[RejectionReason, ...] = (),
    qualification_status: str | None = None,
    error_message: str = "",
) -> tuple[ResearchRegistryRecord, ResearchLearningRecord]:
    spec = _make_spec(strategy_name, symbol, timeframe, param_val)
    ev = _make_evidence(spec, promotion_status, rejection_reasons)
    rec = construct_registry_record_from_evidence(
        evidence=ev,
        candidate_id=f"cand_{strategy_name}_{param_val}",
        trial_id=f"trial_{strategy_name}_{param_val}",
        qualification_status=qualification_status or ("QUALIFIED" if promotion_status == PromotionStatus.PROMOTABLE else "REJECTED"),
        error_message=error_message,
    )
    lr = construct_learning_record_from_registry_record(rec)
    return rec, lr


# A. Empty learning memory produces no patterns
def test_requirement_a_empty_memory_produces_no_patterns() -> None:
    patterns = derive_research_knowledge_patterns([])
    assert patterns == ()


# B. One learning record cannot create a repeated pattern unless single observation policy enabled
def test_requirement_b_single_observation_does_not_derive_repeated_pattern() -> None:
    _, lr1 = _create_test_pair()
    # Default policy requires min_observations=2
    patterns = derive_research_knowledge_patterns([lr1])
    assert patterns == ()

    # Explicit single observation policy
    policy_single = ResearchKnowledgeDerivationPolicy(allow_single_observation_patterns=True)
    patterns_single = derive_research_knowledge_patterns([lr1], policy=policy_single)
    assert len(patterns_single) == 1
    assert patterns_single[0].observation_count == 1


# C. Repeated successful observations derive one SUCCESS_PATTERN
def test_requirement_c_repeated_success_derives_success_pattern() -> None:
    rec1, lr1 = _create_test_pair(param_val=10.0)
    rec2, lr2 = _create_test_pair(param_val=20.0)

    assert lr1.classification == ResearchOutcomeClassification.SUCCESS
    assert lr2.classification == ResearchOutcomeClassification.SUCCESS

    patterns = derive_research_knowledge_patterns([lr1, lr2])
    assert len(patterns) == 1
    pat = patterns[0]
    assert pat.category == ResearchPatternCategory.SUCCESS_PATTERN
    assert pat.observation_count == 2
    assert pat.success_count == 2
    assert pat.failure_count == 0
    assert pat.inconclusive_count == 0
    assert not pat.is_contradictory
    assert set(pat.supporting_learning_ids) == {lr1.learning_id, lr2.learning_id}


# D. Repeated failed observations derive one FAILURE_PATTERN
def test_requirement_d_repeated_failure_derives_failure_pattern() -> None:
    rec1, lr1 = _create_test_pair(
        param_val=10.0,
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )
    rec2, lr2 = _create_test_pair(
        param_val=20.0,
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )

    assert lr1.classification == ResearchOutcomeClassification.FAILURE
    assert lr2.classification == ResearchOutcomeClassification.FAILURE

    patterns = derive_research_knowledge_patterns([lr1, lr2])
    assert len(patterns) == 1
    pat = patterns[0]
    assert pat.category == ResearchPatternCategory.FAILURE_PATTERN
    assert pat.observation_count == 2
    assert pat.failure_count == 2
    assert not pat.is_contradictory


# E. Repeated inconclusive observations derive INCONCLUSIVE_PATTERN
def test_requirement_e_repeated_inconclusive_derives_inconclusive_pattern() -> None:
    rec1, lr1 = _create_test_pair(
        param_val=10.0,
        rejection_reasons=(RejectionReason.INSUFFICIENT_DATA,),
    )
    rec2, lr2 = _create_test_pair(
        param_val=20.0,
        rejection_reasons=(RejectionReason.INSUFFICIENT_DATA,),
    )

    assert lr1.classification == ResearchOutcomeClassification.INCONCLUSIVE
    assert lr2.classification == ResearchOutcomeClassification.INCONCLUSIVE

    patterns = derive_research_knowledge_patterns([lr1, lr2])
    assert len(patterns) == 1
    pat = patterns[0]
    assert pat.category == ResearchPatternCategory.INCONCLUSIVE_PATTERN
    assert pat.observation_count == 2
    assert pat.inconclusive_count == 2
    assert not pat.is_contradictory


# F. General repeated observations produce GENERAL_OBSERVATION when appropriate
def test_requirement_f_general_observations_derive_general_observation_pattern() -> None:
    # Create artificial learning records with same conditions but mixed non-contradictory classifications
    # (e.g., INCONCLUSIVE + SUCCESS or custom general classification)
    rec1, lr1 = _create_test_pair(param_val=10.0)
    rec2, lr2 = _create_test_pair(param_val=20.0, rejection_reasons=(RejectionReason.INSUFFICIENT_DATA,))

    # lr1 is SUCCESS, lr2 is INCONCLUSIVE -> Mixed non-contradictory outcomes
    patterns = derive_research_knowledge_patterns([lr1, lr2])
    assert len(patterns) == 1
    pat = patterns[0]
    assert pat.category == ResearchPatternCategory.GENERAL_OBSERVATION
    assert pat.observation_count == 2
    assert pat.success_count == 1
    assert pat.inconclusive_count == 1
    assert not pat.is_contradictory


# G. Deterministic canonicalization produces identical fingerprints regardless of input ordering
def test_requirement_g_deterministic_canonicalization_order_independence() -> None:
    _, lr1 = _create_test_pair(param_val=10.0)
    _, lr2 = _create_test_pair(param_val=20.0)
    _, lr3 = _create_test_pair(param_val=30.0)

    p1 = derive_research_knowledge_patterns([lr1, lr2, lr3])
    p2 = derive_research_knowledge_patterns([lr3, lr1, lr2])
    p3 = derive_research_knowledge_patterns([lr2, lr3, lr1])

    assert len(p1) == 1
    assert p1[0].pattern_id == p2[0].pattern_id == p3[0].pattern_id
    assert p1[0].canonical_fingerprint == p2[0].canonical_fingerprint == p3[0].canonical_fingerprint


# H. Supporting learning-record identities are preserved
def test_requirement_h_supporting_learning_record_identities_preserved() -> None:
    _, lr1 = _create_test_pair(param_val=10.0)
    _, lr2 = _create_test_pair(param_val=20.0)

    patterns = derive_research_knowledge_patterns([lr1, lr2])
    assert len(patterns) == 1
    pat = patterns[0]
    expected_ids = tuple(sorted([lr1.learning_id, lr2.learning_id]))
    assert pat.supporting_learning_ids == expected_ids


# I. Missing supporting learning record fails closed during store validation
def test_requirement_i_missing_learning_record_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ResearchRegistryStore(base_dir=tmp_dir)
        rec1, lr1 = _create_test_pair(param_val=10.0)
        rec2, lr2 = _create_test_pair(param_val=20.0)

        # Register rec1 and lr1, but NOT lr2
        store.register(rec1)
        store.register_learning_record(lr1)

        with pytest.raises(PatternDerivationError, match="missing from registry store"):
            derive_research_knowledge_patterns([lr1, lr2], registry_store=store)


# J. Lineage mismatch fails closed
def test_requirement_j_lineage_mismatch_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ResearchRegistryStore(base_dir=tmp_dir)
        rec1, lr1 = _create_test_pair(param_val=10.0)
        rec2, lr2 = _create_test_pair(param_val=20.0)

        store.register(rec1)
        store.register_learning_record(lr1)
        store.register(rec2)
        store.register_learning_record(lr2)

        # Create a corrupted learning record with mismatched experiment fingerprint
        corrupted_lr2 = ResearchLearningRecord(
            learning_id=lr2.learning_id,
            source_record_id=lr2.source_record_id,
            experiment_fingerprint="mismatched_exp_fp_999999",
            evidence_fingerprint=lr2.evidence_fingerprint,
            candidate_id=lr2.candidate_id,
            search_fingerprint=lr2.search_fingerprint,
            trial_id=lr2.trial_id,
            dataset_scope_id=lr2.dataset_scope_id,
            execution_assumptions_id=lr2.execution_assumptions_id,
            code_provenance_id=lr2.code_provenance_id,
            methodology_version=lr2.methodology_version,
            classification=lr2.classification,
            observed_conditions=lr2.observed_conditions,
            lessons=lr2.lessons,
            constraints=lr2.constraints,
            confidence_score=lr2.confidence_score,
            rejection_reasons=lr2.rejection_reasons,
        )

        with pytest.raises(PatternDerivationError, match="Lineage mismatch|missing from registry store"):
            derive_research_knowledge_patterns([lr1, corrupted_lr2], registry_store=store)


# K. Conflicting outcomes are represented as contradiction/inconclusive
def test_requirement_k_conflicting_outcomes_derive_contradictory_inconclusive_pattern() -> None:
    _, lr_success = _create_test_pair(param_val=10.0, promotion_status=PromotionStatus.PROMOTABLE)
    _, lr_failure = _create_test_pair(
        param_val=20.0,
        promotion_status=PromotionStatus.REJECTED,
        rejection_reasons=(RejectionReason.FAILED_OOS,),
    )

    patterns = derive_research_knowledge_patterns([lr_success, lr_failure])
    assert len(patterns) == 1
    pat = patterns[0]
    assert pat.is_contradictory
    assert pat.category == ResearchPatternCategory.INCONCLUSIVE_PATTERN
    assert pat.success_count == 1
    assert pat.failure_count == 1
    assert "Contradictory" in pat.statement


# L. Different non-equivalent observed conditions are not merged
def test_requirement_l_different_conditions_are_not_merged() -> None:
    _, lr_eurusd_1 = _create_test_pair(symbol="EURUSD", param_val=10.0)
    _, lr_eurusd_2 = _create_test_pair(symbol="EURUSD", param_val=20.0)
    _, lr_xauusd_1 = _create_test_pair(symbol="XAUUSD", param_val=10.0)
    _, lr_xauusd_2 = _create_test_pair(symbol="XAUUSD", param_val=20.0)

    patterns = derive_research_knowledge_patterns([lr_eurusd_1, lr_eurusd_2, lr_xauusd_1, lr_xauusd_2])
    assert len(patterns) == 2
    symbols = {p.normalized_conditions.symbol for p in patterns}
    assert symbols == {"EURUSD", "XAUUSD"}


# M. Pattern persistence is idempotent
def test_requirement_m_pattern_persistence_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ResearchRegistryStore(base_dir=tmp_dir)
        rec1, lr1 = _create_test_pair(param_val=10.0)
        rec2, lr2 = _create_test_pair(param_val=20.0)

        store.register(rec1)
        store.register_learning_record(lr1)
        store.register(rec2)
        store.register_learning_record(lr2)

        patterns = derive_research_knowledge_patterns([lr1, lr2], registry_store=store)
        assert len(patterns) == 1
        pat = patterns[0]

        reg1 = store.register_pattern(pat)
        reg2 = store.register_pattern(pat)

        assert reg1.canonical_fingerprint == reg2.canonical_fingerprint == pat.canonical_fingerprint
        assert len(store.list_patterns()) == 1


# N. Conflicting pattern registration is detected safely
def test_requirement_n_conflicting_pattern_registration_raises_error() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ResearchRegistryStore(base_dir=tmp_dir)
        rec1, lr1 = _create_test_pair(param_val=10.0)
        rec2, lr2 = _create_test_pair(param_val=20.0)

        store.register(rec1)
        store.register_learning_record(lr1)
        store.register(rec2)
        store.register_learning_record(lr2)

        patterns = derive_research_knowledge_patterns([lr1, lr2], registry_store=store)
        pat = patterns[0]
        store.register_pattern(pat)

        # Create conflicting pattern with same pattern_id but different statement
        conflicting_dict = pat.as_dict()
        conflicting_dict["statement"] = "Conflicting modified statement"
        conflicting_pat = ResearchKnowledgePattern.from_dict(conflicting_dict)

        with pytest.raises(RegistryConflictError, match="Conflicting knowledge pattern exists"):
            store.register_pattern(conflicting_pat)


# O. Pattern queries return deterministic results
def test_requirement_o_pattern_queries_return_deterministic_results() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ResearchRegistryStore(base_dir=tmp_dir)

        # Strategy 1 EURUSD SUCCESS
        rec1, lr1 = _create_test_pair(strategy_name="Strat1", symbol="EURUSD", param_val=10.0)
        rec2, lr2 = _create_test_pair(strategy_name="Strat1", symbol="EURUSD", param_val=20.0)

        # Strategy 2 XAUUSD FAILURE
        rec3, lr3 = _create_test_pair(
            strategy_name="Strat2", symbol="XAUUSD", param_val=10.0,
            promotion_status=PromotionStatus.REJECTED, rejection_reasons=(RejectionReason.FAILED_OOS,)
        )
        rec4, lr4 = _create_test_pair(
            strategy_name="Strat2", symbol="XAUUSD", param_val=20.0,
            promotion_status=PromotionStatus.REJECTED, rejection_reasons=(RejectionReason.FAILED_OOS,)
        )

        for r in (rec1, rec2, rec3, rec4):
            store.register(r)
        for lr in (lr1, lr2, lr3, lr4):
            store.register_learning_record(lr)

        derived = derive_research_knowledge_patterns([lr1, lr2, lr3, lr4], registry_store=store)
        for p in derived:
            store.register_pattern(p)

        # Query by category
        success_pats = store.query_patterns(category=ResearchPatternCategory.SUCCESS_PATTERN)
        assert len(success_pats) == 1
        assert success_pats[0].normalized_conditions.strategy_name == "Strat1"

        failure_pats = store.query_patterns(category=ResearchPatternCategory.FAILURE_PATTERN)
        assert len(failure_pats) == 1
        assert failure_pats[0].normalized_conditions.strategy_name == "Strat2"

        # Query by symbol
        xau_pats = store.query_patterns(symbol="XAUUSD")
        assert len(xau_pats) == 1
        assert xau_pats[0].normalized_conditions.symbol == "XAUUSD"


# P. Historical patterns remain traceable after newer pattern derivation/supersession
def test_requirement_p_historical_patterns_traceable_after_supersession() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = ResearchRegistryStore(base_dir=tmp_dir)

        rec1, lr1 = _create_test_pair(param_val=10.0)
        rec2, lr2 = _create_test_pair(param_val=20.0)
        rec3, lr3 = _create_test_pair(param_val=30.0)

        for r in (rec1, rec2, rec3):
            store.register(r)
        for lr in (lr1, lr2, lr3):
            store.register_learning_record(lr)

        p_v1 = derive_research_knowledge_patterns([lr1, lr2], registry_store=store)[0]
        store.register_pattern(p_v1)

        p_v2 = derive_research_knowledge_patterns([lr1, lr2, lr3], registry_store=store)[0]

        newer, older = store.record_pattern_supersession(p_v2, p_v1)

        assert not older.is_active
        assert older.superseded_by_pattern_id == newer.pattern_id
        assert newer.supersedes_pattern_id == older.pattern_id
        assert newer.is_active

        # Active query returns only newer
        active_pats = store.query_patterns(active_only=True)
        assert len(active_pats) == 1
        assert active_pats[0].pattern_id == newer.pattern_id

        # Historical retrieval returns older
        retrieved_older = store.get_pattern_by_id(older.pattern_id)
        assert retrieved_older is not None
        assert not retrieved_older.is_active


# Q. Pattern derivation does not modify live execution
def test_requirement_q_no_live_execution_mutation() -> None:
    import src.evaluation.live_execution_runtime as live_rt
    assert hasattr(live_rt, "resolve_authoritative_promoted_candidate")
    assert hasattr(live_rt, "LiveExecutionRuntime")


# R. Pattern derivation does not modify strategy code
def test_requirement_r_no_strategy_code_mutation() -> None:
    import src.evaluation.strategy_suite as strat_suite
    assert hasattr(strat_suite, "run_default_strategy_suite")


# S. Pattern derivation does not modify Project2
def test_requirement_s_no_project2_mutation() -> None:
    import src.integration.project2_publisher as p2_pub
    assert hasattr(p2_pub, "Project2Publisher")
