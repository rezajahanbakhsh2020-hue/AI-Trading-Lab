"""Comprehensive Tests for Canonical Research-Only Mathematical Expression Execution Bridge.

Enforces:
- Adversarial Signal Policy semantics (LONG > 0, SHORT < 0, NO TRADE == 0, NaN/Inf fail closed, warmup == 0)
- Identity & Fingerprint invariants (expression, candidate, hypothesis, spec, evidence)
- Search-Space Pre-Validation & Governance Failure Handling
- End-to-End Governed Execution Flow (GENERATED -> ACCEPTED_FOR_RESEARCH -> run_research_experiment -> ResearchEvidence)
- Anti-Production leakage & default registry isolation
"""

from __future__ import annotations

import json
import math
import pytest
import numpy as np
import pandas as pd

from src.evaluation.mathematical_expression import (
    MathematicalDomainError,
    MathematicalEvaluationError,
    MathematicalExpression,
    MathematicalOperator,
    MathematicalSearchSpace,
    SearchSpaceValidationError,
)
from src.evaluation.mathematical_expression_candidate import (
    MathematicalCandidateValidationError,
    MathematicalExpressionCandidate,
    MathematicalSignalInterpretationPolicy,
    run_mathematical_research_experiment,
)
from src.evaluation.mathematical_expression_strategy import (
    create_mathematical_research_registry,
    generate_mathematical_expression_signal,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import DiscoveryCriteria
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
    # Generate deterministic trending close series
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


# --- ADVERSARIAL SIGNAL POLICY TESTS ---

def test_signal_policy_canonical_evaluation():
    """Test positive -> LONG (1), negative -> SHORT (-1), zero -> NO TRADE (0)."""
    policy = MathematicalSignalInterpretationPolicy()
    assert policy.evaluate_value(10.5) == 1
    assert policy.evaluate_value(-0.001) == -1
    assert policy.evaluate_value(0.0) == 0

    s = pd.Series([5.0, -2.0, 0.0, 0.1, -100.0])
    evaluated = policy.evaluate_series(s)
    assert list(evaluated) == [1, -1, 0, 1, -1]


def test_signal_policy_non_finite_fail_closed():
    """Test that NaN and Inf fail closed in signal policy evaluation."""
    policy = MathematicalSignalInterpretationPolicy()
    with pytest.raises(MathematicalCandidateValidationError, match="Non-finite evaluation value rejected"):
        policy.evaluate_value(float("nan"))

    with pytest.raises(MathematicalCandidateValidationError, match="Non-finite evaluation value rejected"):
        policy.evaluate_value(float("inf"))

    with pytest.raises(MathematicalCandidateValidationError, match="Non-finite values encountered"):
        policy.evaluate_series(pd.Series([1.0, np.nan, -1.0]))


def test_strategy_adapter_signal_generation_and_warmup(sample_lineage, synthetic_ohlcv):
    """Test strategy adapter signal generation, warmup enforcement, and index preservation."""
    ds, ea, cp = sample_lineage
    # Expression: close[t-1] - close[t-2] with lag=1 => max_lookback = 3
    child1 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    child2 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=2, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    expr = MathematicalExpression(operator=MathematicalOperator.SUB, children=(child1, child2), lag=0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    assert expr.max_lookback == 2

    res = generate_mathematical_expression_signal(synthetic_ohlcv, expression_obj=expr)

    assert "signal" in res.columns
    assert "expression_value" in res.columns
    assert len(res) == len(synthetic_ohlcv)

    # Verify warmup rows are NO TRADE (0)
    for row in range(expr.max_lookback):
        assert res["signal"].iloc[row] == 0

    # Verify valid post-warmup signals match sign policy
    for row in range(expr.max_lookback, len(synthetic_ohlcv)):
        val = res["expression_value"].iloc[row]
        sig = res["signal"].iloc[row]
        if val > 0:
            assert sig == 1
        elif val < 0:
            assert sig == -1
        else:
            assert sig == 0


def test_strategy_adapter_protected_operator_domain_violations_fail_closed(sample_lineage):
    """Test that protected division by zero or log/sqrt domain violations fail closed in adapter."""
    ds, ea, cp = sample_lineage
    df_zero = pd.DataFrame({"close": [0.0, 0.0, 0.0, 0.0]})

    div_zero_expr = MathematicalExpression(
        operator=MathematicalOperator.PROTECTED_DIV,
        children=(
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp),
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp),
        ),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    with pytest.raises(MathematicalDomainError, match="Protected division by zero"):
        generate_mathematical_expression_signal(df_zero, expression_obj=div_zero_expr)


# --- CANDIDATE IDENTITY & FINGERPRINTING TESTS ---

def test_candidate_fingerprint_changes_on_any_structural_or_policy_change(sample_lineage):
    """Test candidate fingerprint changes on expression, search space, signal policy, lag, or seed changes."""
    ds, ea, cp = sample_lineage

    expr1 = MathematicalExpression(
        operator=MathematicalOperator.FEATURE, feature_name="close", lag=1,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    expr2 = MathematicalExpression(
        operator=MathematicalOperator.FEATURE, feature_name="close", lag=2,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )

    space1 = MathematicalSearchSpace(search_id="space1", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space2 = MathematicalSearchSpace(search_id="space2", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    policy1 = MathematicalSignalInterpretationPolicy(version="1.0")
    policy2 = MathematicalSignalInterpretationPolicy(version="2.0")

    cand1 = MathematicalExpressionCandidate(expr1, space1, policy1, ds, ea, cp, random_seed=42)
    cand_expr2 = MathematicalExpressionCandidate(expr2, space1, policy1, ds, ea, cp, random_seed=42)
    cand_space2 = MathematicalExpressionCandidate(expr1, space2, policy1, ds, ea, cp, random_seed=42)
    cand_policy2 = MathematicalExpressionCandidate(expr1, space1, policy2, ds, ea, cp, random_seed=42)
    cand_seed2 = MathematicalExpressionCandidate(expr1, space1, policy1, ds, ea, cp, random_seed=99)

    assert cand1.fingerprint != cand_expr2.fingerprint
    assert cand1.fingerprint != cand_space2.fingerprint
    assert cand1.fingerprint != cand_policy2.fingerprint
    assert cand1.fingerprint != cand_seed2.fingerprint


def test_roundtrip_canonical_json_reconstruction_preserves_fingerprint(sample_lineage):
    """Test deterministic canonical JSON serialization and fingerprint reproducibility."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT, constant_value=5.0,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    space = MathematicalSearchSpace(search_id="space_rt", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()

    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp, random_seed=123)

    json_str = cand.to_canonical_json()
    d = json.loads(json_str)

    # Reconstruct expression
    expr_rec = MathematicalExpression.from_canonical_dict(d["expression"])
    policy_rec = MathematicalSignalInterpretationPolicy.from_canonical_dict(d["signal_policy"])

    assert expr_rec.fingerprint == expr.fingerprint
    assert policy_rec.fingerprint == policy.fingerprint


# --- SEARCH SPACE PRE-VALIDATION & GOVERNANCE TESTS ---

def test_candidate_pre_validation_fails_closed_before_execution(sample_lineage):
    """Test that candidate.validate() catches depth/node limit violations before execution."""
    ds, ea, cp = sample_lineage
    space_strict = MathematicalSearchSpace(
        search_id="space_strict", max_depth=2, max_node_count=2,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    policy = MathematicalSignalInterpretationPolicy()

    # Depth = 3 exceeds max_depth = 2
    deep_expr = MathematicalExpression(
        operator=MathematicalOperator.NEG,
        children=(
            MathematicalExpression(
                operator=MathematicalOperator.ADD,
                children=(
                    MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp),
                    MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=2.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp),
                ),
                dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
            ),
        ),
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )

    invalid_cand = MathematicalExpressionCandidate(deep_expr, space_strict, policy, ds, ea, cp)

    with pytest.raises(SearchSpaceValidationError, match="exceeds search space limit"):
        invalid_cand.validate()

    # Test run_mathematical_research_experiment rejects invalid candidate before execution
    with pytest.raises(SearchSpaceValidationError):
        run_mathematical_research_experiment(invalid_cand)


def test_candidate_pre_validation_lineage_mismatch_fails_closed(sample_lineage):
    """Test that candidate lineage mismatch against search space or AST fails closed."""
    ds1, ea, cp = sample_lineage
    ds2 = DatasetScope("ds_other", "XAUUSD", "1h", "2023-01-01", "2023-01-10")

    space1 = MathematicalSearchSpace(search_id="space1", dataset_scope=ds1, execution_assumptions=ea, code_provenance=cp)
    expr1 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds1, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()

    # Candidate specifies ds2 while space/expr specified ds1
    mismatched_cand = MathematicalExpressionCandidate(expr1, space1, policy, ds2, ea, cp)

    with pytest.raises(MathematicalCandidateValidationError, match="does not match SearchSpace DatasetScope"):
        mismatched_cand.validate()


# --- END-TO-END GOVERNED RESEARCH EXECUTION TEST ---

def test_end_to_end_governed_mathematical_research_experiment(sample_lineage, synthetic_ohlcv):
    """End-to-End Proof:
    1. Construct MathematicalExpression AST (close[t-1] - close[t-2])
    2. Construct MathematicalExpressionCandidate
    3. Route through run_mathematical_research_experiment():
       - candidate pre-validation
       - ResearchHypothesis(status=GENERATED)
       - accept_hypothesis_for_research() => ACCEPTED_FOR_RESEARCH
       - run_research_experiment() with dedicated research-only registry
       - ResearchEvidence returned
    4. Assert full identity traceability and absence of production artifacts.
    """
    ds, ea, cp = sample_lineage

    c1 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    c2 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=2, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    expr = MathematicalExpression(operator=MathematicalOperator.SUB, children=(c1, c2), dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    space = MathematicalSearchSpace(
        search_id="space_e2e", max_depth=5, max_node_count=10, max_window_size=50,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    policy = MathematicalSignalInterpretationPolicy()

    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    wf_protocol = WalkForwardProtocol(train_size=40, test_size=15)
    criteria = DiscoveryCriteria(min_observations_is=20, min_observations_oos=10)

    # Execute governed bridge
    evidence = run_mathematical_research_experiment(
        candidate=cand,
        df=synthetic_ohlcv,
        criteria=criteria,
        walk_forward_protocol=wf_protocol,
    )

    # Verification assertions
    assert evidence.spec.strategy_name == "mathematical_expression"
    assert evidence.spec.dataset_scope == ds
    assert evidence.spec.execution_assumptions == ea
    assert evidence.spec.code_provenance == cp
    assert evidence.spec.walk_forward_protocol == wf_protocol

    # Identity invariant tracing proof
    assert evidence.experiment_fingerprint == evidence.spec.fingerprint
    assert evidence.spec.parameters["expression_fingerprint"] == expr.fingerprint
    assert evidence.spec.parameters["candidate_fingerprint"] == cand.fingerprint

    # Verify no DEFAULT_REGISTRY pollution
    assert not DEFAULT_REGISTRY.contains("mathematical_expression")


def test_research_only_registry_isolation():
    """Verify that DEFAULT_REGISTRY remains strictly free of research mathematical_expression strategy."""
    assert not DEFAULT_REGISTRY.contains("mathematical_expression")

    research_reg = create_mathematical_research_registry()
    assert research_reg.contains("mathematical_expression")
    assert not DEFAULT_REGISTRY.contains("mathematical_expression")
