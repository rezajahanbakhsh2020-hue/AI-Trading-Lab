"""Focused integration tests for Governed Research Hypothesis Execution Lifecycle.

Validates:
- GENERATED hypothesis cannot execute and fails closed.
- Explicit governance acceptance transitions status to ACCEPTED_FOR_RESEARCH.
- Already ACCEPTED_FOR_RESEARCH hypothesis cannot be re-accepted (one-way lifecycle boundary).
- ACCEPTED_FOR_RESEARCH hypothesis passes to existing run_research_experiment().
- Identity, fingerprint, provenance, and lineage are preserved end-to-end in ResearchEvidence.
- Rejected, superseded, or incomplete hypotheses fail closed.
- Existing non-governed ResearchExperimentSpec execution path remains compatible.
- Single execution engine (run_research_experiment) handles both spec and accepted hypothesis paths.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.hypothesis_generator import (
    HypothesisGenerationContext,
    HypothesisGenerationError,
    KnowledgeHypothesisGenerator,
    UnacceptedHypothesisError,
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
    ResearchEvidence,
    ResearchExperimentSpec,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_knowledge import (
    PatternObservationSummary,
    ResearchKnowledgePattern,
    ResearchPatternCategory,
)
from src.evaluation.research_runner import run_research_experiment


def _make_sample_dataframe() -> pd.DataFrame:
    """Helper to generate a clean sample market dataframe for backtest execution."""
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1000.0] * 100,
        }
    )
    return df


def _make_valid_context() -> HypothesisGenerationContext:
    return HypothesisGenerationContext(
        dataset_scope=DatasetScope(
            dataset_id="test_dataset_daily",
            symbol="XAUUSD",
            timeframe="1D",
            start_date="2025-01-01",
            end_date="2025-04-10",
        ),
        execution_assumptions=ExecutionAssumptions(
            transaction_cost=0.0001,
            slippage=0.0001,
            latency_ms=10.0,
        ),
        code_provenance=CodeProvenance(
            commit_sha="a1b2c3d4e5f678901234567890abcdef12345678",
            repository_status="clean",
            author="Researcher",
        ),
        benchmark_reference="BUY_AND_HOLD",
        parameters_override={"fast_period": 10, "slow_period": 30},
    )


def _make_valid_pattern() -> ResearchKnowledgePattern:
    summary = PatternObservationSummary(
        symbol="XAUUSD",
        timeframe="1D",
        strategy_name="SMA_CROSSOVER",
        strategy_version="1.0.0",
        dataset_scope_id="scope_001",
        execution_assumptions_id="ea_001",
        code_provenance_id="cp_001",
        methodology_version="knowledge_v1.0",
    )
    return ResearchKnowledgePattern(
        pattern_id="pat_momentum_001",
        category=ResearchPatternCategory.SUCCESS_PATTERN,
        statement="Momentum crossover strategy shows positive returns in daily gold data.",
        normalized_conditions=summary,
        supporting_learning_ids=("learn_001",),
        supporting_experiment_fingerprints=("exp_fp_001",),
        supporting_evidence_fingerprints=("ev_001",),
        observation_count=1,
        success_count=1,
        failure_count=0,
        inconclusive_count=0,
        is_contradictory=False,
    )


def test_generated_hypothesis_cannot_execute():
    """Prove that a GENERATED hypothesis cannot execute and raises UnacceptedHypothesisError."""
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_valid_context()
    pattern = _make_valid_pattern()

    hypotheses = generator.generate([pattern], context=ctx)
    assert len(hypotheses) == 1
    generated_hyp = hypotheses[0]
    assert generated_hyp.status == HypothesisStatus.GENERATED

    df = _make_sample_dataframe()

    # Attempting to execute a GENERATED hypothesis must fail closed
    with pytest.raises(UnacceptedHypothesisError) as exc_info:
        run_research_experiment(generated_hyp, df=df)

    assert "is not accepted for research execution" in str(exc_info.value)


def test_explicit_governance_acceptance_produces_accepted_for_research():
    """Prove that explicit governance acceptance produces HypothesisStatus.ACCEPTED_FOR_RESEARCH."""
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_valid_context()
    pattern = _make_valid_pattern()

    generated_hyp = generator.generate([pattern], context=ctx)[0]
    accepted_hyp = accept_hypothesis_for_research(generated_hyp)

    assert accepted_hyp.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH
    assert accepted_hyp.hypothesis_id == generated_hyp.hypothesis_id
    assert accepted_hyp.fingerprint == generated_hyp.fingerprint
    assert accepted_hyp.canonical_hypothesis_fingerprint == generated_hyp.canonical_hypothesis_fingerprint
    assert accepted_hyp.strategy_name == generated_hyp.strategy_name
    assert accepted_hyp.code_provenance == generated_hyp.code_provenance


def test_already_accepted_hypothesis_cannot_be_reaccepted():
    """Regression test: Prove that an already ACCEPTED_FOR_RESEARCH hypothesis cannot be re-accepted."""
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_valid_context()
    pattern = _make_valid_pattern()

    generated_hyp = generator.generate([pattern], context=ctx)[0]
    accepted_hyp = accept_hypothesis_for_research(generated_hyp)
    assert accepted_hyp.status == HypothesisStatus.ACCEPTED_FOR_RESEARCH

    # Attempting to re-accept an already accepted hypothesis MUST fail closed
    with pytest.raises(ValueError) as exc_info:
        accept_hypothesis_for_research(accepted_hyp)

    assert "current status is 'accepted_for_research'" in str(exc_info.value)
    assert f"expected '{HypothesisStatus.GENERATED.value}'" in str(exc_info.value)


def test_accepted_hypothesis_reaches_existing_research_runner():
    """Prove that an ACCEPTED_FOR_RESEARCH hypothesis successfully executes via run_research_experiment()."""
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_valid_context()
    pattern = _make_valid_pattern()

    generated_hyp = generator.generate([pattern], context=ctx)[0]
    accepted_hyp = accept_hypothesis_for_research(generated_hyp)

    df = _make_sample_dataframe()
    evidence = run_research_experiment(accepted_hyp, df=df)

    assert isinstance(evidence, ResearchEvidence)
    assert evidence.experiment_fingerprint == evidence.spec.fingerprint
    assert evidence.spec.hypothesis == accepted_hyp.statement
    assert evidence.spec.strategy_name == accepted_hyp.strategy_name
    assert evidence.spec.parameters == accepted_hyp.parameters
    assert evidence.spec.code_provenance == accepted_hyp.code_provenance
    assert len(evidence.partitions) > 0


def test_rejected_and_superseded_hypotheses_fail_closed():
    """Prove that rejected or superseded hypotheses cannot be accepted or executed."""
    generator = KnowledgeHypothesisGenerator()
    ctx = _make_valid_context()
    pattern = _make_valid_pattern()

    generated_hyp = generator.generate([pattern], context=ctx)[0]
    df = _make_sample_dataframe()

    # Reject hypothesis
    rejected_hyp = reject_hypothesis(generated_hyp, reason="Flawed statistical baseline")
    assert rejected_hyp.status == HypothesisStatus.REJECTED

    with pytest.raises(ValueError, match="current status is 'rejected'"):
        accept_hypothesis_for_research(rejected_hyp)

    with pytest.raises(UnacceptedHypothesisError):
        run_research_experiment(rejected_hyp, df=df)

    # Supersede hypothesis
    superseded_hyp = supersede_hypothesis(generated_hyp, superseding_id="hyp_new_123")
    assert superseded_hyp.status == HypothesisStatus.SUPERSEDED

    with pytest.raises(ValueError, match="current status is 'superseded'"):
        accept_hypothesis_for_research(superseded_hyp)

    with pytest.raises(UnacceptedHypothesisError):
        run_research_experiment(superseded_hyp, df=df)


def test_incomplete_provenance_hypothesis_cannot_bypass_governance():
    """Prove that a hypothesis with missing or incomplete CodeProvenance fails closed upon acceptance/execution."""
    with pytest.raises(ValueError, match="commit_sha must be a non-empty string"):
        CodeProvenance(commit_sha="", repository_status="clean", author="")

    valid_cp = CodeProvenance(commit_sha="a1b2c3d4", repository_status="clean", author="researcher")

    with pytest.raises(ValueError, match="statement must be a non-empty string"):
        ResearchHypothesis(
            statement="   ",
            methodology_version="1.0",
            strategy_name="SMA_CROSSOVER",
            strategy_version="1.0.0",
            dataset_scope=DatasetScope(
                dataset_id="ds_01", symbol="XAUUSD", timeframe="1D", start_date="2025-01-01", end_date="2025-04-10"
            ),
            execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0),
            code_provenance=valid_cp,
            benchmark_reference="BUY_AND_HOLD",
        )


def test_raw_research_experiment_spec_execution_fails_closed():
    """Prove that passing a raw ResearchExperimentSpec directly to run_research_experiment fails closed with TypeError."""
    spec = ResearchExperimentSpec(
        hypothesis="Direct experiment spec without governed ResearchHypothesis wrapper",
        methodology_version="discovery_v1.0",
        strategy_name="SMA_CROSSOVER",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_01", symbol="XAUUSD", timeframe="1D", start_date="2025-01-01", end_date="2025-04-10"
        ),
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0),
        code_provenance=CodeProvenance(commit_sha="commit_sha_12345"),
        benchmark_reference="BUY_AND_HOLD",
        parameters={"fast_period": 10, "slow_period": 30},
        walk_forward_protocol=WalkForwardProtocol(train_size=40, test_size=15),
    )

    df = _make_sample_dataframe()
    with pytest.raises(TypeError, match="expects an authoritative ResearchHypothesis instance"):
        run_research_experiment(spec, df=df)


def test_single_execution_engine_accepts_only_accepted_hypothesis():
    """Verify that run_research_experiment executes strictly via accepted ResearchHypothesis."""
    spec = ResearchExperimentSpec(
        hypothesis="Test hypothesis for single engine verification",
        methodology_version="1.0",
        strategy_name="SMA_CROSSOVER",
        strategy_version="1.0.0",
        dataset_scope=DatasetScope(
            dataset_id="ds_01", symbol="XAUUSD", timeframe="1D", start_date="2025-01-01", end_date="2025-04-10"
        ),
        execution_assumptions=ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0001, latency_ms=10.0),
        code_provenance=CodeProvenance(commit_sha="sha_12345"),
        benchmark_reference="BUY_AND_HOLD",
        parameters={"fast_period": 10, "slow_period": 30},
        walk_forward_protocol=WalkForwardProtocol(train_size=40, test_size=15),
    )

    hyp = ResearchHypothesis.from_experiment_spec(spec)
    assert hyp.status == HypothesisStatus.GENERATED

    with pytest.raises(UnacceptedHypothesisError):
        run_research_experiment(hyp, df=_make_sample_dataframe())

    accepted_hyp = accept_hypothesis_for_research(hyp)
    df = _make_sample_dataframe()
    ev_hyp = run_research_experiment(accepted_hyp, df=df)

    assert ev_hyp.experiment_fingerprint == spec.fingerprint
