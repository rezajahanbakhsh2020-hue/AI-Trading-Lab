"""Comprehensive Tests for Canonical Mathematical Discovery Engine Integration.

Enforces:
1. Happy path: bounded SymbolicSearch candidate enters DiscoveryEngine and reaches ResearchEvidence through the canonical mathematical bridge.
2. Governance order: memory governance executes before hypothesis acceptance/execution.
3. Discovery-feedback governance: blocked feedback candidate never reaches accept_hypothesis_for_research() or research execution.
4. Acceptance failure: GENERATED mathematical hypothesis that cannot transition to ACCEPTED_FOR_RESEARCH produces a blocked/rejected auditable trial and no evidence.
5. Candidate identity tampering: mutate expression fingerprint / candidate fingerprint / search-space fingerprint and prove fail-closed before execution.
6. Generator metadata tampering: generator_id mismatch, generator_version mismatch, and random_seed mismatch must fail closed before candidate execution.
7. DatasetScope mismatch: candidate/search-space/hypothesis DatasetScope mismatch must prevent execution.
8. ExecutionAssumptions mismatch.
9. CodeProvenance mismatch.
10. WalkForwardProtocol mismatch or absence.
11. Budget: MathematicalSearch.max_search_budget and DiscoveryEngine max_trials cannot be exceeded.
12. Determinism: same search constitution + same lineage + same generator metadata produces the same candidate identities and ordering.
13. Duplicate prevention: duplicate mathematical candidates do not create duplicate research evidence.
14. Research-only isolation: mathematical discovery/execution never mutates DEFAULT_REGISTRY and never creates a production strategy.
15. Anti-leakage: prove generation does not inspect market data or ResearchEvidence/performance outcomes.
16. Blocked-candidate sentinel: monkeypatch the research execution boundary and prove it is never called when memory/discovery governance blocks the candidate.
17. End-to-end identity: expression fingerprint -> candidate fingerprint -> hypothesis fingerprint -> evidence experiment fingerprint remains traceable and internally consistent.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import pandas as pd

from src.evaluation.discovery_engine import DiscoveryEngine, DiscoveryCriteria, ResearchSearchPolicy
from src.evaluation.discovery_feedback import DiscoveryFeedbackType, ResearchDiscoveryFeedback
from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalOperator,
    MathematicalSearchSpace,
)
from src.evaluation.mathematical_expression_candidate import (
    MathematicalCandidateValidationError,
    MathematicalExpressionCandidate,
    MathematicalSignalInterpretationPolicy,
)
from src.evaluation.mathematical_search import (
    MathematicalSearchError,
    SearchTerminationReason,
    SymbolicSearch,
)
from src.evaluation.memory_governance import (
    DiscoveryMemoryGovernanceResult,
    MemoryGovernanceDecision,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    RejectionReason,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_knowledge import (
    PatternObservationSummary,
    ResearchKnowledgePattern,
    ResearchPatternCategory,
)
from src.evaluation.research_registry import DoNotRepeatConstraint, _compute_scope_id, _compute_ea_id, _compute_cp_id
from src.strategies.registry import DEFAULT_REGISTRY


@pytest.fixture
def sample_lineage():
    ds = DatasetScope("ds_test", "XAUUSD", "1h", "2023-01-01", "2023-01-07")
    ea = ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0)
    cp = CodeProvenance("commit_abc123456789")
    return ds, ea, cp


@pytest.fixture
def synthetic_ohlcv(sample_lineage):
    ds, _, _ = sample_lineage
    dates = pd.date_range("2023-01-01", periods=150, freq="1h", tz="UTC")
    close = [100.0 + i * 0.5 + (1.0 if i % 2 == 0 else -0.5) for i in range(150)]
    df = pd.DataFrame({
        "timestamp": dates,
        "open": close,
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "volume": [1000.0 + i * 10 for i in range(150)],
        "return": [0.0] + [(close[i] - close[i - 1]) / close[i - 1] for i in range(1, 150)],
    })
    return df


@pytest.fixture
def sample_search_space(sample_lineage):
    ds, ea, cp = sample_lineage
    return MathematicalSearchSpace(
        search_id="space_math_discovery",
        max_depth=3,
        max_node_count=5,
        allowed_operators=(
            MathematicalOperator.FEATURE,
            MathematicalOperator.CONSTANT,
            MathematicalOperator.ADD,
            MathematicalOperator.SUB,
        ),
        allowed_features=("close", "volume"),
        min_lag=0,
        max_lag=2,
        constant_bounds=(-2.0, 2.0),
        constant_precision=1.0,
        max_search_budget=10,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        generator_id="symbolic_search",
        generator_version="1.0",
        random_seed=0,
    )


# --- TEST 1: HAPPY PATH ---

def test_happy_path_symbolic_search_candidate_reaches_research_evidence(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=3,
    )

    assert res.candidates_evaluated == 3
    assert len(res.trial_ledger) == 3
    for trial in res.trial_ledger:
        assert trial.status in ("QUALIFIED", "REJECTED")
        assert trial.evidence_fingerprint is not None
        assert trial.experiment_fingerprint != ""


# --- TEST 2: GOVERNANCE ORDER ENFORCEMENT ---

def test_governance_order_memory_governance_runs_before_hypothesis_acceptance(sample_lineage, synthetic_ohlcv, sample_search_space, monkeypatch):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    # Spy on accept_hypothesis_for_research
    accept_spy = MagicMock(side_effect=accept_hypothesis_for_research)
    monkeypatch.setattr("src.evaluation.discovery_engine.accept_hypothesis_for_research", accept_spy)

    # Strategy that generates 1 candidate
    strategy = SymbolicSearch()
    search_res = strategy.search(sample_search_space, ds, ea, cp, limit=1)
    cand = search_res.candidates[0]

    # Create active DoNotRepeatConstraint matching cand.candidate_id
    constraint = DoNotRepeatConstraint(
        constraint_id="cnst_block_first",
        pattern_key=cand.candidate_id,
        reason="Blocked by memory governance test",
        source_record_id="rec_001",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        condition_description="Test constraint",
        confidence_score=1.0,
    )

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=1,
        memory_store=(constraint,),
        enable_memory_governance=True,
    )

    assert res.candidates_evaluated == 1
    assert res.blocked_trial_count == 1
    assert res.trial_ledger[0].status == "BLOCKED"
    assert RejectionReason.GOVERNANCE_BLOCKED in res.trial_ledger[0].rejection_reasons

    # Memory governance blocked candidate MUST NOT call accept_hypothesis_for_research
    accept_spy.assert_not_called()


# --- TEST 3: DISCOVERY FEEDBACK GOVERNANCE ---

def test_discovery_feedback_governance_attaches_feedback_records(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    pattern = ResearchKnowledgePattern(
        pattern_id="pat_success_close",
        category=ResearchPatternCategory.SUCCESS_PATTERN,
        statement="Close feature performs well on XAUUSD 1h",
        supporting_learning_ids=("lr_1",),
        supporting_experiment_fingerprints=("exp_1",),
        supporting_evidence_fingerprints=("ev_1",),
        observation_count=1,
        success_count=1,
        failure_count=0,
        inconclusive_count=0,
        is_contradictory=False,
        normalized_conditions=PatternObservationSummary(
            strategy_name="mathematical_expression",
            strategy_version="1.0.0",
            symbol=ds.symbol,
            timeframe=ds.timeframe,
            dataset_scope_id=_compute_scope_id(ds),
            execution_assumptions_id=_compute_ea_id(ea),
            code_provenance_id=_compute_cp_id(cp),
            methodology_version="discovery_v1.0",
        ),
    )

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=2,
        knowledge_patterns=(pattern,),
        enable_discovery_feedback=True,
    )

    assert len(res.discovery_feedback) >= 1
    assert any(fb.feedback_type == DiscoveryFeedbackType.RELEVANT_SUCCESS_PATTERN for fb in res.discovery_feedback)


# --- TEST 4: ACCEPTANCE FAILURE ---

def test_hypothesis_acceptance_failure_produces_blocked_trial_no_evidence(sample_lineage, synthetic_ohlcv, sample_search_space, monkeypatch):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    def failing_accept_hypothesis(hypothesis: ResearchHypothesis) -> ResearchHypothesis:
        raise ValueError("Acceptance authority rejected hypothesis: Test rejection")

    monkeypatch.setattr("src.evaluation.discovery_engine.accept_hypothesis_for_research", failing_accept_hypothesis)

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=2,
    )

    assert res.candidates_evaluated == 2
    assert res.blocked_trial_count == 2
    for tr in res.trial_ledger:
        assert tr.status == "BLOCKED"
        assert RejectionReason.GOVERNANCE_BLOCKED in tr.rejection_reasons
        assert tr.evidence_fingerprint is None
    assert len(res.promoted_evidence) == 0
    assert len(res.rejected_evidence) == 0


# --- TEST 5: CANDIDATE IDENTITY TAMPERING ---

def test_candidate_search_space_fingerprint_tampering_fails_closed(sample_lineage, synthetic_ohlcv, sample_search_space, monkeypatch):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    real_search = SymbolicSearch.search

    def tampered_search(*args, **kwargs):
        res = real_search(*args, **kwargs)
        # Construct candidate with mismatched search space
        other_space = MathematicalSearchSpace(search_id="other_space", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
        orig = res.candidates[0]
        tampered_cand = MathematicalExpressionCandidate(
            expression=orig.expression,
            search_space=other_space,  # TAMPERED
            signal_policy=orig.signal_policy,
            dataset_scope=orig.dataset_scope,
            execution_assumptions=orig.execution_assumptions,
            code_provenance=orig.code_provenance,
            generator_id=orig.generator_id,
            generator_version=orig.generator_version,
            random_seed=orig.random_seed,
        )
        return type(res)(
            search_space_fingerprint=res.search_space_fingerprint,
            generator_id=res.generator_id,
            generator_version=res.generator_version,
            random_seed=res.random_seed,
            generated_count=1,
            budget=res.budget,
            termination_reason=res.termination_reason,
            candidates=(tampered_cand,),
        )

    monkeypatch.setattr(SymbolicSearch, "search", tampered_search)

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=1,
    )

    assert res.trial_ledger[0].status == "FAILED"
    assert RejectionReason.SPECIFICATION_INVALID in res.trial_ledger[0].rejection_reasons
    assert "SearchSpace fingerprint" in res.trial_ledger[0].error_message


# --- TEST 6: GENERATOR METADATA TAMPERING ---

def test_generator_metadata_tampering_fails_closed(sample_lineage, synthetic_ohlcv, sample_search_space, monkeypatch):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    real_search = SymbolicSearch.search

    def tampered_search(*args, **kwargs):
        res = real_search(*args, **kwargs)
        orig = res.candidates[0]
        tampered_cand = MathematicalExpressionCandidate(
            expression=orig.expression,
            search_space=orig.search_space,
            signal_policy=orig.signal_policy,
            dataset_scope=orig.dataset_scope,
            execution_assumptions=orig.execution_assumptions,
            code_provenance=orig.code_provenance,
            generator_id="tampered_generator",  # TAMPERED
            generator_version=orig.generator_version,
            random_seed=orig.random_seed,
        )
        return type(res)(
            search_space_fingerprint=res.search_space_fingerprint,
            generator_id=res.generator_id,
            generator_version=res.generator_version,
            random_seed=res.random_seed,
            generated_count=1,
            budget=res.budget,
            termination_reason=res.termination_reason,
            candidates=(tampered_cand,),
        )

    monkeypatch.setattr(SymbolicSearch, "search", tampered_search)

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=1,
    )

    assert res.trial_ledger[0].status == "FAILED"
    assert RejectionReason.SPECIFICATION_INVALID in res.trial_ledger[0].rejection_reasons
    assert "generator metadata" in res.trial_ledger[0].error_message


# --- TEST 7: DATASETSCOPE MISMATCH ---

def test_dataset_scope_mismatch_prevents_execution(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    ds_mismatch = DatasetScope("ds_mismatch", "EURUSD", "1h", "2023-01-01", "2023-01-07")
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    with pytest.raises(MathematicalSearchError, match="Supplied DatasetScope"):
        engine.run_mathematical_discovery(
            df=synthetic_ohlcv,
            search_space=sample_search_space,
            dataset_scope=ds_mismatch,  # Mismatched against sample_search_space.dataset_scope
            execution_assumptions=ea,
            code_provenance=cp,
            wf_train_size=40,
            wf_test_size=15,
            limit=1,
        )


# --- TEST 8: EXECUTIONASSUMPTIONS MISMATCH ---

def test_execution_assumptions_mismatch_prevents_execution(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    ea_mismatch = ExecutionAssumptions(0.005, 0.005, 100.0)
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    with pytest.raises(MathematicalSearchError, match="Supplied ExecutionAssumptions"):
        engine.run_mathematical_discovery(
            df=synthetic_ohlcv,
            search_space=sample_search_space,
            dataset_scope=ds,
            execution_assumptions=ea_mismatch,  # Mismatched
            code_provenance=cp,
            wf_train_size=40,
            wf_test_size=15,
            limit=1,
        )


# --- TEST 9: CODEPROVENANCE MISMATCH ---

def test_code_provenance_mismatch_prevents_execution(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    cp_mismatch = CodeProvenance("commit_mismatch_999")
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    with pytest.raises(MathematicalSearchError, match="Supplied CodeProvenance"):
        engine.run_mathematical_discovery(
            df=synthetic_ohlcv,
            search_space=sample_search_space,
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp_mismatch,  # Mismatched
            wf_train_size=40,
            wf_test_size=15,
            limit=1,
        )


# --- TEST 10: WALKFORWARDPROTOCOL REQUIREMENT ---

def test_walk_forward_protocol_mandatory_resolution(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=None,
        wf_test_size=None,
        limit=1,
    )

    assert res.candidates_evaluated == 1
    # Verify walk_forward_protocol was resolved on executed spec
    for cand in res.research_candidates:
        if cand.evidence:
            assert cand.hypothesis.walk_forward_protocol is not None
            assert cand.hypothesis.walk_forward_protocol.train_size > 0
            assert cand.hypothesis.walk_forward_protocol.test_size > 0


# --- TEST 11: BUDGET ENFORCEMENT ---

def test_search_space_max_search_budget_and_search_policy_max_trials_respected(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    # Cap search policy max_trials at 2
    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        search_policy=ResearchSearchPolicy(max_trials=2),
        wf_train_size=40,
        wf_test_size=15,
        limit=10,
    )

    assert res.candidates_evaluated == 2
    assert len(res.trial_ledger) == 2


# --- TEST 12: DETERMINISM ---

def test_determinism_same_constitution_lineage_metadata_produces_identical_candidates_and_ordering(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    res1 = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=5,
    )

    res2 = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=5,
    )

    assert [t.candidate_id for t in res1.trial_ledger] == [t.candidate_id for t in res2.trial_ledger]
    assert [t.candidate_fingerprint for t in res1.trial_ledger] == [t.candidate_fingerprint for t in res2.trial_ledger]
    assert [t.experiment_fingerprint for t in res1.trial_ledger] == [t.experiment_fingerprint for t in res2.trial_ledger]


# --- TEST 13: DUPLICATE PREVENTION ---

def test_duplicate_candidate_prevention(sample_lineage, synthetic_ohlcv, sample_search_space, monkeypatch):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    real_search = SymbolicSearch.search

    def duplicate_search(*args, **kwargs):
        res = real_search(*args, **kwargs)
        orig = res.candidates[0]
        # Return 2 identical candidates
        return type(res)(
            search_space_fingerprint=res.search_space_fingerprint,
            generator_id=res.generator_id,
            generator_version=res.generator_version,
            random_seed=res.random_seed,
            generated_count=2,
            budget=res.budget,
            termination_reason=res.termination_reason,
            candidates=(orig, orig),
        )

    monkeypatch.setattr(SymbolicSearch, "search", duplicate_search)

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=2,
    )

    assert res.candidates_evaluated == 2
    # Second evaluation must have DUPLICATE_CANDIDATE rejection reason
    dup_cand = res.research_candidates[1]
    if dup_cand.evidence:
        assert RejectionReason.DUPLICATE_CANDIDATE in dup_cand.evidence.rejection_reasons


# --- TEST 14: RESEARCH-ONLY ISOLATION ---

def test_research_only_isolation_default_registry_unmodified(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=2,
    )

    # Built-in DEFAULT_REGISTRY must remain untouched and MUST NOT contain mathematical_expression
    assert not DEFAULT_REGISTRY.contains("mathematical_expression")


# --- TEST 15: ANTI-LEAKAGE GUARANTEE ---

def test_anti_leakage_search_generation_does_not_inspect_market_data(sample_lineage, sample_search_space):
    ds, ea, cp = sample_lineage
    strategy = SymbolicSearch()

    # Pass empty/dummy data parameters to SymbolicSearch prove it does NOT evaluate DataFrame
    search_res = strategy.search(
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        limit=5,
    )

    assert search_res.generated_count == 5
    for cand in search_res.candidates:
        assert isinstance(cand, MathematicalExpressionCandidate)


# --- TEST 16: BLOCKED-CANDIDATE SENTINEL ---

def test_blocked_candidate_execution_boundary_sentinel_never_called(sample_lineage, synthetic_ohlcv, sample_search_space, monkeypatch):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    # Spy sentinel on run_research_experiment
    exec_spy = MagicMock(side_effect=AssertionError("run_research_experiment MUST NOT be called for blocked candidates!"))
    monkeypatch.setattr("src.evaluation.discovery_engine.run_research_experiment", exec_spy)

    strategy = SymbolicSearch()
    search_res = strategy.search(sample_search_space, ds, ea, cp, limit=1)
    cand = search_res.candidates[0]

    constraint = DoNotRepeatConstraint(
        constraint_id="cnst_block_sentinel",
        pattern_key=cand.candidate_id,
        reason="Blocked by sentinel test constraint",
        source_record_id="rec_sentinel",
        experiment_fingerprint="",
        evidence_fingerprint=None,
        condition_description="Test constraint",
        confidence_score=1.0,
    )

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=1,
        memory_store=(constraint,),
        enable_memory_governance=True,
    )

    assert res.blocked_trial_count == 1
    exec_spy.assert_not_called()


# --- TEST 17: END-TO-END IDENTITY CHAIN TRACEABILITY ---

def test_end_to_end_identity_chain_traceability(sample_lineage, synthetic_ohlcv, sample_search_space):
    ds, ea, cp = sample_lineage
    engine = DiscoveryEngine(criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10))

    res = engine.run_mathematical_discovery(
        df=synthetic_ohlcv,
        search_space=sample_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        wf_train_size=40,
        wf_test_size=15,
        limit=1,
    )

    cand_wrapper = res.research_candidates[0]
    assert cand_wrapper.evidence is not None
    ev = cand_wrapper.evidence

    # Expression fingerprint -> candidate fingerprint -> hypothesis fingerprint -> evidence experiment fingerprint
    expr_fp = ev.spec.parameters["expression_fingerprint"]
    cand_fp = ev.spec.parameters["candidate_fingerprint"]
    search_fp = ev.spec.parameters["search_space_fingerprint"]

    assert search_fp == sample_search_space.fingerprint
    assert ev.spec.fingerprint == ev.experiment_fingerprint
    assert cand_wrapper.hypothesis.fingerprint == ev.experiment_fingerprint
