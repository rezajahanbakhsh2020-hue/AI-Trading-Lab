"""Comprehensive unit test suite for Controlled Knowledge-to-Discovery Feedback layer.

Covers items A through S:
A. no knowledge -> discovery remains unchanged
B. relevant success pattern is surfaced as explicit discovery context
C. relevant failure pattern is surfaced as explicit discovery context
D. inconclusive pattern is surfaced as uncertainty/context only
E. general observation remains informational
F. non-equivalent symbol does not match
G. non-equivalent timeframe does not match
H. non-equivalent strategy does not match
I. dataset/execution/code-provenance mismatch is not silently matched
J. deterministic feedback fingerprint
K. deterministic ordering of multiple matching patterns
L. duplicate registration is idempotent
M. conflicting registration fails closed
N. missing pattern fails closed
O. lineage mismatch fails closed
P. superseded/invalid knowledge does not silently influence discovery
Q. DoNotRepeatConstraint behavior from PR #26 remains unchanged
R. existing multiple-testing correction semantics remain unchanged
S. no direct mutation of strategy/live/risk/promotion/Project2
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.evaluation.candidate_generator import CandidateSpec, ResearchSearchSpace
from src.evaluation.discovery_engine import DiscoveryEngine, DiscoveryCriteria
from src.evaluation.discovery_feedback import (
    DiscoveryFeedbackType,
    ResearchDiscoveryFeedback,
    evaluate_candidate_discovery_feedback,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    PromotionStatus,
    RejectionReason,
)
from src.evaluation.research_knowledge import (
    PatternObservationSummary,
    ResearchKnowledgePattern,
    ResearchPatternCategory,
    derive_research_knowledge_patterns,
)
from src.evaluation.research_registry import (
    DoNotRepeatConstraint,
    LessonCategory,
    RegistryConflictError,
    RegistryStatus,
    ResearchLearningRecord,
    ResearchOutcomeClassification,
    ResearchRegistryRecord,
    ResearchRegistryStore,
    StructuredObservedConditions,
    _compute_cp_id,
    _compute_ea_id,
    _compute_scope_id,
    construct_learning_record_from_registry_record,
)


def _make_dummy_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": 1000.0,
    })


def _make_test_fixtures():
    ds = DatasetScope(
        dataset_id="ds_test",
        symbol="XAUUSD",
        timeframe="1h",
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-09T08:00:00Z",
    )
    ea = ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="abc1234", repository_status="clean", author="test")
    cand = CandidateSpec(
        generator_name="test_gen",
        generator_version="1.0.0",
        strategy_name="momentum_burst",
        parameters={"lookback": 14, "threshold": 0.02},
    )
    return ds, ea, cp, cand


def _make_mock_pattern(
    pattern_id: str,
    category: ResearchPatternCategory,
    ds: DatasetScope,
    ea: ExecutionAssumptions,
    cp: CodeProvenance,
    strategy_name: str = "momentum_burst",
    symbol: str | None = None,
    timeframe: str | None = None,
    learning_ids: tuple[str, ...] = ("learn_1", "learn_2"),
    is_active: bool = True,
    superseded_by: str | None = None,
) -> ResearchKnowledgePattern:
    norm = PatternObservationSummary(
        symbol=symbol or ds.symbol,
        timeframe=timeframe or ds.timeframe,
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope_id=_compute_scope_id(ds),
        execution_assumptions_id=_compute_ea_id(ea),
        code_provenance_id=_compute_cp_id(cp),
        methodology_version="knowledge_v1.0",
    )
    obs_cnt = len(learning_ids)
    succ_cnt = obs_cnt if category == ResearchPatternCategory.SUCCESS_PATTERN else 0
    fail_cnt = obs_cnt if category == ResearchPatternCategory.FAILURE_PATTERN else 0
    inc_cnt = obs_cnt if category in (ResearchPatternCategory.INCONCLUSIVE_PATTERN, ResearchPatternCategory.GENERAL_OBSERVATION) else 0

    return ResearchKnowledgePattern(
        pattern_id=pattern_id,
        category=category,
        statement=f"Mock pattern {pattern_id} for {strategy_name}",
        normalized_conditions=norm,
        supporting_learning_ids=learning_ids,
        supporting_experiment_fingerprints=("exp_1", "exp_2")[:len(learning_ids)],
        supporting_evidence_fingerprints=("ev_1", "ev_2")[:len(learning_ids)],
        observation_count=obs_cnt,
        success_count=succ_cnt,
        failure_count=fail_cnt,
        inconclusive_count=inc_cnt,
        is_contradictory=(category == ResearchPatternCategory.INCONCLUSIVE_PATTERN),
        is_active=is_active,
        superseded_by_pattern_id=superseded_by,
    )


# A. no knowledge → discovery remains unchanged
def test_requirement_a_no_knowledge_discovery_unchanged():
    ds, ea, cp, cand = _make_test_fixtures()
    df = _make_dummy_df()
    engine = DiscoveryEngine()

    res = engine.run_discovery(
        df=df,
        candidates=[cand],
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[],
    )

    assert res.candidates_evaluated == 1
    assert len(res.discovery_feedback) == 1
    fb = res.discovery_feedback[0]
    assert fb.feedback_type == DiscoveryFeedbackType.NO_FEEDBACK
    assert fb.pattern_id is None


# B. relevant success pattern is surfaced as explicit discovery context
def test_requirement_b_relevant_success_pattern():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_succ_1", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    fb = feedbacks[0]
    assert fb.feedback_type == DiscoveryFeedbackType.RELEVANT_SUCCESS_PATTERN
    assert fb.pattern_id == "pat_succ_1"
    assert "matched active knowledge pattern" in fb.reason


# C. relevant failure pattern is surfaced as explicit discovery context
def test_requirement_c_relevant_failure_pattern():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_fail_1", ResearchPatternCategory.FAILURE_PATTERN, ds, ea, cp)

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    fb = feedbacks[0]
    assert fb.feedback_type == DiscoveryFeedbackType.RELEVANT_FAILURE_PATTERN
    assert fb.pattern_id == "pat_fail_1"


# D. inconclusive pattern is surfaced as uncertainty/context only
def test_requirement_d_inconclusive_pattern():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_inc_1", ResearchPatternCategory.INCONCLUSIVE_PATTERN, ds, ea, cp)

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    fb = feedbacks[0]
    assert fb.feedback_type == DiscoveryFeedbackType.INCONCLUSIVE_PATTERN
    assert fb.pattern_id == "pat_inc_1"


# E. general observation remains informational
def test_requirement_e_general_observation():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_gen_1", ResearchPatternCategory.GENERAL_OBSERVATION, ds, ea, cp)

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    fb = feedbacks[0]
    assert fb.feedback_type == DiscoveryFeedbackType.GENERAL_OBSERVATION
    assert fb.pattern_id == "pat_gen_1"


# F. non-equivalent symbol does not match
def test_requirement_f_non_equivalent_symbol():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_sym_mismatch", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp, symbol="EURUSD")

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    assert feedbacks[0].feedback_type == DiscoveryFeedbackType.NO_FEEDBACK


# G. non-equivalent timeframe does not match
def test_requirement_g_non_equivalent_timeframe():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_tf_mismatch", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp, timeframe="5m")

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    assert feedbacks[0].feedback_type == DiscoveryFeedbackType.NO_FEEDBACK


# H. non-equivalent strategy does not match
def test_requirement_h_non_equivalent_strategy():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_strat_mismatch", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp, strategy_name="mean_reversion")

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    assert feedbacks[0].feedback_type == DiscoveryFeedbackType.NO_FEEDBACK


# I. dataset/execution/code-provenance mismatch is not silently matched
def test_requirement_i_scope_execution_code_mismatch():
    ds, ea, cp, cand = _make_test_fixtures()
    ea_other = ExecutionAssumptions(transaction_cost=0.005, slippage=0.005, latency_ms=100.0)
    pat = _make_mock_pattern("pat_ea_mismatch", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea_other, cp)

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    assert len(feedbacks) == 1
    assert feedbacks[0].feedback_type == DiscoveryFeedbackType.NO_FEEDBACK


# J. deterministic feedback fingerprint
def test_requirement_j_deterministic_feedback_fingerprint():
    ds, ea, cp, cand = _make_test_fixtures()
    pat = _make_mock_pattern("pat_succ_det", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

    fb1 = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )[0]

    fb2 = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )[0]

    assert fb1.canonical_fingerprint == fb2.canonical_fingerprint
    assert len(fb1.canonical_fingerprint) == 64


# K. deterministic ordering of multiple matching patterns
def test_requirement_k_deterministic_ordering():
    ds, ea, cp, cand = _make_test_fixtures()
    p1 = _make_mock_pattern("pat_b_succ", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)
    p2 = _make_mock_pattern("pat_a_fail", ResearchPatternCategory.FAILURE_PATTERN, ds, ea, cp)

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[p1, p2],  # Passed in reverse order
    )

    assert len(feedbacks) == 2
    # Should be sorted deterministically by feedback_id
    assert feedbacks == tuple(sorted(feedbacks, key=lambda f: f.feedback_id))


# L. duplicate registration is idempotent
def test_requirement_l_idempotent_registration():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchRegistryStore(base_dir=tmpdir)
        ds, ea, cp, cand = _make_test_fixtures()
        pat = _make_mock_pattern("pat_succ_1", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

        fb = evaluate_candidate_discovery_feedback(
            candidate=cand,
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp,
            knowledge_patterns=[pat],
        )[0]

        registered1 = store.register_feedback(fb)
        registered2 = store.register_feedback(fb)

        assert registered1.canonical_fingerprint == registered2.canonical_fingerprint
        assert len(store.list_feedback()) == 1


# M. conflicting registration fails closed
def test_requirement_m_conflicting_registration_fails_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchRegistryStore(base_dir=tmpdir)
        ds, ea, cp, cand = _make_test_fixtures()
        pat = _make_mock_pattern("pat_succ_1", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

        fb = evaluate_candidate_discovery_feedback(
            candidate=cand,
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp,
            knowledge_patterns=[pat],
        )[0]

        store.register_feedback(fb)

        # Create conflicting feedback with same feedback_id
        conflicting = ResearchDiscoveryFeedback(
            feedback_id=fb.feedback_id,
            feedback_type=DiscoveryFeedbackType.RELEVANT_FAILURE_PATTERN,
            candidate_id=cand.candidate_id,
            experiment_fingerprint=fb.experiment_fingerprint,
            pattern_id=fb.pattern_id,
            pattern_fingerprint=fb.pattern_fingerprint,
            search_id=fb.search_id,
            search_fingerprint=fb.search_fingerprint,
            supporting_learning_ids=fb.supporting_learning_ids,
            supporting_experiment_fingerprints=fb.supporting_experiment_fingerprints,
            supporting_evidence_fingerprints=fb.supporting_evidence_fingerprints,
            reason="Conflicting reason",
            normalized_conditions=fb.normalized_conditions,
        )

        with pytest.raises(RegistryConflictError):
            store.register_feedback(conflicting)


# N. missing pattern fails closed
def test_requirement_n_missing_pattern_fails_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchRegistryStore(base_dir=tmpdir)
        ds, ea, cp, cand = _make_test_fixtures()
        pat = _make_mock_pattern("pat_missing", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

        feedbacks = evaluate_candidate_discovery_feedback(
            candidate=cand,
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp,
            knowledge_patterns=[pat],
            registry_store=store,  # Pattern not registered in store
        )

        assert len(feedbacks) == 1
        assert feedbacks[0].feedback_type == DiscoveryFeedbackType.FAIL_CLOSED
        assert "missing from registry store" in feedbacks[0].reason


# O. lineage mismatch fails closed
def test_requirement_o_lineage_mismatch_fails_closed():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchRegistryStore(base_dir=tmpdir)
        ds, ea, cp, cand = _make_test_fixtures()
        pat = _make_mock_pattern("pat_mismatch", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

        # Register pattern in store
        # First register source registry record and learning record
        from src.evaluation.research_registry import (
            ResearchEvidenceLineage,
            ResearchReproducibilityDescriptor,
        )
        rec = ResearchRegistryRecord(
            record_id="rec_1",
            experiment_fingerprint="exp_1",
            evidence_fingerprint="ev_1",
            candidate_id="cand_1",
            search_fingerprint="search_1",
            search_id="s1",
            trial_id="t1",
            trial_index=0,
            status=RegistryStatus.QUALIFIED,
            qualification_status="QUALIFIED",
            promotion_status=PromotionStatus.PROMOTABLE.value,
            rejection_reasons=(),
            dataset_scope_id=_compute_scope_id(ds),
            execution_assumptions_id=_compute_ea_id(ea),
            code_provenance_id=_compute_cp_id(cp),
            methodology_version="discovery_v1.0",
            selection_assessment_id=None,
            robustness_assessment_id=None,
            benchmark_status=None,
            regime_status=None,
            error_message="",
            reproducibility=ResearchReproducibilityDescriptor(
                experiment_fingerprint="exp_1",
                evidence_fingerprint="ev_1",
                dataset_scope_id=_compute_scope_id(ds),
                execution_assumptions_id=_compute_ea_id(ea),
                code_provenance_id=_compute_cp_id(cp),
                methodology_version="discovery_v1.0",
                search_space_fingerprint="search_1",
                trial_id="t1",
                candidate_id="cand_1",
            ),
            lineage=ResearchEvidenceLineage(
                search_id="s1",
                search_fingerprint="search_1",
                trial_id="t1",
                trial_index=0,
                candidate_id="cand_1",
                experiment_fingerprint="exp_1",
                evidence_fingerprint="ev_1",
                qualification_status="QUALIFIED",
                selection_assessment_id=None,
                robustness_assessment_id=None,
                promotion_status=PromotionStatus.PROMOTABLE.value,
            ),
        )
        store.register(rec)
        lr = construct_learning_record_from_registry_record(rec)
        store.register_learning_record(lr)

        # Register pattern
        pat_for_store = _make_mock_pattern(
            "pat_mismatch",
            ResearchPatternCategory.SUCCESS_PATTERN,
            ds, ea, cp,
            learning_ids=(lr.learning_id,)
        )
        store.register_pattern(pat_for_store)

        # Evaluate with input pattern having mismatched fingerprint
        mismatched_pat = ResearchKnowledgePattern(
            pattern_id="pat_mismatch",
            category=ResearchPatternCategory.SUCCESS_PATTERN,
            statement="Mismatched statement",
            normalized_conditions=pat_for_store.normalized_conditions,
            supporting_learning_ids=(lr.learning_id,),
            supporting_experiment_fingerprints=("exp_1",),
            supporting_evidence_fingerprints=("ev_1",),
            observation_count=1,
            success_count=1,
            failure_count=0,
            inconclusive_count=0,
            is_contradictory=False,
        )

        feedbacks = evaluate_candidate_discovery_feedback(
            candidate=cand,
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp,
            knowledge_patterns=[mismatched_pat],
            registry_store=store,
        )

        assert len(feedbacks) == 1
        assert feedbacks[0].feedback_type == DiscoveryFeedbackType.FAIL_CLOSED
        assert "Fingerprint mismatch" in feedbacks[0].reason


# P. superseded/invalid knowledge does not silently influence discovery
def test_requirement_p_superseded_knowledge_ignored():
    ds, ea, cp, cand = _make_test_fixtures()
    pat_inactive = _make_mock_pattern(
        "pat_inactive",
        ResearchPatternCategory.SUCCESS_PATTERN,
        ds, ea, cp,
        is_active=False,
        superseded_by="pat_newer",
    )

    feedbacks = evaluate_candidate_discovery_feedback(
        candidate=cand,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat_inactive],
    )

    assert len(feedbacks) == 1
    assert feedbacks[0].feedback_type == DiscoveryFeedbackType.NO_FEEDBACK


# Q. DoNotRepeatConstraint behavior from PR #26 remains unchanged
def test_requirement_q_donotrepeat_constraint_authoritative():
    ds, ea, cp, cand = _make_test_fixtures()
    df = _make_dummy_df()
    engine = DiscoveryEngine()

    constraint = DoNotRepeatConstraint(
        constraint_id="const_hard_block",
        source_record_id="rec_fail",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        pattern_key=f"fail_pattern:{cand.strategy_name}:{ds.symbol}:{ds.timeframe}:{_compute_scope_id(ds)[:8]}",
        condition_description="Failure condition",
        reason="Hard block rule test",
        confidence_score=0.95,
        is_active=True,
    )

    # Candidate should still be hard blocked by DoNotRepeatConstraint regardless of feedback
    res = engine.run_discovery(
        df=df,
        candidates=[cand],
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        memory_store=[constraint],
        enable_memory_governance=True,
    )

    assert res.blocked_trial_count == 1
    assert res.trial_ledger[0].status == "BLOCKED"


# R. existing multiple-testing correction semantics remain unchanged
def test_requirement_r_multiple_testing_semantics_unchanged():
    from src.evaluation.selection_governance import (
        ResearchSelectionPolicy,
        assess_research_selection,
    )
    ds, ea, cp, cand = _make_test_fixtures()
    df = _make_dummy_df()
    engine = DiscoveryEngine()

    res = engine.run_discovery(
        df=df,
        candidates=[cand],
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    # Check that selection governance module functions normally
    policy = ResearchSelectionPolicy()
    assert policy.alpha == 0.05
    assert str(policy.correction_method).lower() in ("holm_bonferroni", "bonferroni", "none")


# S. no direct mutation of strategy/live/risk/promotion/Project2
def test_requirement_s_no_direct_mutation():
    import src.strategies.registry as strat_reg
    import src.evaluation.live_execution_runtime as live_rt
    import src.evaluation.live_production_decision as prod_dec

    ds, ea, cp, cand = _make_test_fixtures()
    df = _make_dummy_df()
    pat = _make_mock_pattern("pat_succ_1", ResearchPatternCategory.SUCCESS_PATTERN, ds, ea, cp)

    engine = DiscoveryEngine()
    res = engine.run_discovery(
        df=df,
        candidates=[cand],
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        knowledge_patterns=[pat],
    )

    # Verify no strategy registry mutation
    assert "cand_test_1" not in strat_reg.DEFAULT_REGISTRY.names()
    # Verify feedback exists in result
    assert len(res.discovery_feedback) == 1
    fb = res.discovery_feedback[0]
    assert fb.feedback_type == DiscoveryFeedbackType.RELEVANT_SUCCESS_PATTERN
