"""Comprehensive unit and integration test suite for Knowledge -> Hypothesis boundary.

Tests:
1. Valid knowledge produces a valid hypothesis.
2. Multiple source knowledge records preserve complete lineage.
3. Source evidence lineage is preserved.
4. Missing knowledge fails closed.
5. Missing evidence lineage fails closed.
6. Missing DatasetScope fails closed.
7. Missing ExecutionAssumptions fails closed.
8. Incomplete CodeProvenance fails closed.
9. Hypothesis identity is deterministic.
10. Equivalent hypotheses are deduplicated.
11. Generator version/provenance is persisted.
12. Hypothesis lifecycle transitions behave correctly.
13. Rejected hypotheses cannot enter research as valid.
14. Generated hypotheses cannot create live signals.
15. Generated hypotheses cannot execute trades.
16. Generated hypotheses cannot bypass governance.
17. Existing research/evidence behavior remains compatible.
18. All existing relevant tests remain green.
"""

import tempfile
import pytest

from src.evaluation.hypothesis_generator import (
    HypothesisGenerationContext,
    HypothesisGenerationError,
    KnowledgeHypothesisGenerator,
    KnowledgeHypothesisGeneratorPolicy,
    accept_hypothesis_for_research,
    reject_hypothesis,
    supersede_hypothesis,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    ResearchCandidate,
    ResearchHypothesis,
)
from src.evaluation.research_knowledge import (
    PatternObservationSummary,
    ResearchKnowledgePattern,
    ResearchPatternCategory,
)
from src.evaluation.research_registry import ResearchRegistryStore


def _make_context(
    dataset_id: str = "ds_xauusd_1h",
    symbol: str = "XAUUSD",
    timeframe: str = "1h",
    commit_sha: str = "a1b2c3d4",
) -> HypothesisGenerationContext:
    dataset_scope = DatasetScope(
        dataset_id=dataset_id,
        symbol=symbol,
        timeframe=timeframe,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    execution_assumptions = ExecutionAssumptions(
        transaction_cost=0.0001,
        slippage=0.0002,
        latency_ms=10.0,
    )
    code_provenance = CodeProvenance(
        commit_sha=commit_sha,
        repository_status="clean",
        author="researcher@lab",
    )
    return HypothesisGenerationContext(
        dataset_scope=dataset_scope,
        execution_assumptions=execution_assumptions,
        code_provenance=code_provenance,
        benchmark_reference="BUY_AND_HOLD",
        parameters_override={"lookback": 20},
    )


def _make_pattern(
    pattern_id: str = "pat_001",
    strategy_name: str = "momentum",
    category: ResearchPatternCategory = ResearchPatternCategory.SUCCESS_PATTERN,
    supporting_learning_ids: tuple[str, ...] = ("learn_001",),
    supporting_evidence_fps: tuple[str, ...] = ("ev_fp_001",),
    is_contradictory: bool = False,
    is_active: bool = True,
) -> ResearchKnowledgePattern:
    norm_summary = PatternObservationSummary(
        symbol="XAUUSD",
        timeframe="1h",
        strategy_name=strategy_name,
        strategy_version="1.0.0",
        dataset_scope_id="scope_001",
        execution_assumptions_id="ea_001",
        code_provenance_id="cp_001",
        methodology_version="knowledge_v1.0",
    )
    return ResearchKnowledgePattern(
        pattern_id=pattern_id,
        category=category,
        statement=f"Validated knowledge statement for pattern {pattern_id}",
        normalized_conditions=norm_summary,
        supporting_learning_ids=supporting_learning_ids,
        supporting_experiment_fingerprints=("exp_fp_001",),
        supporting_evidence_fingerprints=supporting_evidence_fps,
        observation_count=2,
        success_count=2 if category == ResearchPatternCategory.SUCCESS_PATTERN else 0,
        failure_count=2 if category == ResearchPatternCategory.FAILURE_PATTERN else 0,
        inconclusive_count=0,
        is_contradictory=is_contradictory,
        is_active=is_active,
    )


# Test 1: Valid knowledge produces a valid hypothesis
def test_1_valid_knowledge_produces_valid_hypothesis():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    pattern = _make_pattern()

    hypotheses = generator.generate([pattern], context=ctx)

    assert len(hypotheses) == 1
    hyp = hypotheses[0]
    assert isinstance(hyp, ResearchHypothesis)
    assert hyp.strategy_name == "momentum"
    assert hyp.status == HypothesisStatus.GENERATED
    assert "pat_001" in hyp.source_knowledge_ids
    assert "ev_fp_001" in hyp.source_evidence_ids


# Test 2: Multiple source knowledge records preserve complete lineage
def test_2_multiple_source_knowledge_records_preserve_complete_lineage():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p1 = _make_pattern("pat_001", supporting_evidence_fps=("ev_fp_001",))
    p2 = _make_pattern("pat_002", supporting_evidence_fps=("ev_fp_002",))

    hypotheses = generator.generate([p1, p2], context=ctx)

    assert len(hypotheses) == 1
    hyp = hypotheses[0]
    assert hyp.source_knowledge_ids == ("pat_001", "pat_002")
    assert hyp.source_evidence_ids == ("ev_fp_001", "ev_fp_002")


# Test 3: Source evidence lineage is preserved
def test_3_source_evidence_lineage_is_preserved():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern(supporting_evidence_fps=("ev_abc", "ev_xyz"))

    hypotheses = generator.generate([p], context=ctx)

    assert hypotheses[0].source_evidence_ids == ("ev_abc", "ev_xyz")


# Test 4: Missing knowledge fails closed
def test_4_missing_knowledge_fails_closed():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()

    with pytest.raises(HypothesisGenerationError, match="Cannot generate hypothesis from empty knowledge sequence"):
        generator.generate([], context=ctx)


# Test 5: Missing evidence lineage fails closed
def test_5_missing_evidence_lineage_fails_closed():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern(supporting_evidence_fps=())

    with pytest.raises(HypothesisGenerationError, match="lacks required supporting evidence lineage"):
        generator.generate([p], context=ctx)


# Test 6: Missing DatasetScope fails closed
def test_6_missing_dataset_scope_fails_closed():
    with pytest.raises(TypeError, match="dataset_scope must be a DatasetScope instance"):
        HypothesisGenerationContext(
            dataset_scope=None,  # type: ignore
            execution_assumptions=ExecutionAssumptions(0.0001, 0.0002, 10.0),
            code_provenance=CodeProvenance("a1b2c3d4"),
        )


# Test 7: Missing ExecutionAssumptions fails closed
def test_7_missing_execution_assumptions_fails_closed():
    with pytest.raises(TypeError, match="execution_assumptions must be an ExecutionAssumptions instance"):
        HypothesisGenerationContext(
            dataset_scope=DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31"),
            execution_assumptions=None,  # type: ignore
            code_provenance=CodeProvenance("a1b2c3d4"),
        )


# Test 8: Incomplete CodeProvenance fails closed
def test_8_incomplete_code_provenance_fails_closed():
    with pytest.raises(ValueError, match="commit_sha must be a non-empty string"):
        CodeProvenance(commit_sha="")


# Test 9: Hypothesis identity is deterministic
def test_9_hypothesis_identity_is_deterministic():
    generator = KnowledgeHypothesisGenerator()
    ctx1 = _make_context()
    ctx2 = _make_context()
    p1 = _make_pattern()
    p2 = _make_pattern()

    hyp1 = generator.generate([p1], context=ctx1)[0]
    hyp2 = generator.generate([p2], context=ctx2)[0]

    assert hyp1.fingerprint == hyp2.fingerprint
    assert hyp1.hypothesis_id == hyp2.hypothesis_id


# Test 10: Equivalent hypotheses are deduplicated
def test_10_equivalent_hypotheses_are_deduplicated():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p1 = _make_pattern("pat_001")
    p2 = _make_pattern("pat_001")

    hypotheses = generator.generate([p1, p2], context=ctx)

    assert len(hypotheses) == 1


# Test 11: Generator version/provenance is persisted
def test_11_generator_version_provenance_persisted():
    policy = KnowledgeHypothesisGeneratorPolicy(generator_version="2.1.0")
    generator = KnowledgeHypothesisGenerator(policy=policy)
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]

    assert hyp.generator_version == "2.1.0"
    assert hyp.as_dict()["generator_version"] == "2.1.0"


# Test 12: Hypothesis lifecycle transitions behave correctly
def test_12_hypothesis_lifecycle_transitions():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]
    assert hyp.status == HypothesisStatus.GENERATED

    accepted = accept_hypothesis_for_research(hyp)
    assert accepted.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH

    rejected = reject_hypothesis(hyp, reason="Poor sample size")
    assert rejected.status == HypothesisStatus.REJECTED
    assert rejected.constraints["rejection_reason"] == "Poor sample size"

    superseded = supersede_hypothesis(hyp, superseding_id="hyp_next_999")
    assert superseded.status == HypothesisStatus.SUPERSEDED
    assert superseded.constraints["superseded_by"] == "hyp_next_999"


# Test 13: Rejected hypotheses cannot enter research as valid
def test_13_rejected_hypotheses_cannot_enter_research_as_valid():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]
    rejected = reject_hypothesis(hyp, reason="Failed initial review")

    with pytest.raises(ValueError, match="Cannot accept REJECTED hypothesis"):
        accept_hypothesis_for_research(rejected)


# Test 14: Generated hypotheses cannot create live signals
def test_14_generated_hypotheses_cannot_create_live_signals():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]

    # Confirm hyp has no production/live methods or signal attributes
    assert not hasattr(hyp, "to_live_signal")
    assert not hasattr(hyp, "to_production_decision")
    assert not hasattr(hyp, "execute_trade")


# Test 15: Generated hypotheses cannot execute trades
def test_15_generated_hypotheses_cannot_execute_trades():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]

    with pytest.raises(AttributeError):
        hyp.place_order()  # type: ignore


# Test 16: Generated hypotheses cannot bypass governance
def test_16_generated_hypotheses_cannot_bypass_governance():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]

    # Create candidate from hypothesis without evidence
    cand = ResearchCandidate(
        candidate_id="cand_hyp_001",
        hypothesis=hyp,
        promotion_status=PromotionStatus.PROPOSED,
    )

    # Candidate promotion check must return False
    assert cand.can_promote() is False

    # Attempting to convert unvalidated candidate to production artifact must fail closed
    with pytest.raises(ValueError, match="evidence is missing"):
        cand.promote_to_production_artifact("XAUUSD", "1h")


# Test 17: Persistence and store integration with ResearchRegistryStore
def test_17_hypothesis_registry_store_integration():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ResearchRegistryStore(base_dir=tmpdir)

        # Register hypothesis
        registered = store.register_hypothesis(hyp)
        assert registered.hypothesis_id == hyp.hypothesis_id

        # Retrieve hypothesis
        retrieved = store.get_hypothesis_by_id(hyp.hypothesis_id)
        assert retrieved is not None
        assert retrieved.fingerprint == hyp.fingerprint

        # Query hypothesis
        queries = store.query_hypotheses(strategy_name="momentum")
        assert len(queries) == 1
        assert queries[0].hypothesis_id == hyp.hypothesis_id


# Test 18: Existing research/evidence behavior remains compatible
def test_18_existing_research_evidence_behavior_remains_compatible():
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_context()
    p = _make_pattern()

    hyp = generator.generate([p], context=ctx)[0]
    spec = hyp.to_experiment_spec()

    assert spec.hypothesis == hyp.statement
    assert spec.strategy_name == hyp.strategy_name
    assert len(spec.fingerprint) == 64
    assert len(hyp.fingerprint) == 64
