"""Comprehensive Tests for Canonical Research-Only Mathematical Expression Execution Bridge.

Enforces:
- Walk-Forward Protocol mandatory authority (no implicit fallbacks)
- Tamper-Evident Parameter Boundary validation & Anti-Recurrence Sentinel checks
- Point vs Series Temporal Invariant equivalence
- Complete Protected Operator Coverage (PROTECTED_DIV, PROTECTED_LOG, PROTECTED_SQRT) through the adapter
- Non-Finite Fail-Closed adapter coverage (NaN, +Inf, -Inf)
- Complete Identity Chain preservation
- Search-Space Constitutional Pre-Validation failure handling
- Governance lifecycle and acceptance state enforcement
- Research-Only Registry isolation
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.evaluation.hypothesis_generator import accept_hypothesis_for_research
from src.evaluation.mathematical_expression import (
    MathematicalDomainError,
    MathematicalEvaluationError,
    MathematicalExpression,
    MathematicalExpressionError,
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
    validate_mathematical_hypothesis_parameters,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    HypothesisStatus,
    PromotionStatus,
    ResearchHypothesis,
    WalkForwardProtocol,
)
from src.evaluation.research_runner import DiscoveryCriteria, run_research_experiment
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


# --- 1. WALK-FORWARD AUTHORITY TESTS ---

def test_walk_forward_protocol_mandatory_authority(sample_lineage):
    """Test that creating a hypothesis or running a mathematical experiment without an explicit WalkForwardProtocol fails closed."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_wf", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    # 1. to_hypothesis without walk_forward_protocol -> FAIL
    with pytest.raises(MathematicalCandidateValidationError, match="requires an explicit WalkForwardProtocol"):
        cand.to_hypothesis(walk_forward_protocol=None)  # type: ignore

    # 2. run_mathematical_research_experiment without walk_forward_protocol -> FAIL
    with pytest.raises(MathematicalCandidateValidationError, match="requires an explicit WalkForwardProtocol"):
        run_mathematical_research_experiment(candidate=cand, walk_forward_protocol=None)  # type: ignore


# --- 2. TAMPER-EVIDENT HYPOTHESIS -> ADAPTER BOUNDARY TESTS ---

def test_tamper_evident_boundary_expression_dict_tampering(sample_lineage):
    """Adversarial test: Mutate expression_dict while keeping expression_fingerprint unchanged -> Fail Closed."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_tamp", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))
    params = dict(hyp.parameters)

    # Tamper with expression_dict (change constant value from 1.0 to 99.0) without updating fingerprint
    tampered_expr_dict = expr.to_canonical_dict()
    tampered_expr_dict["constant_value"] = "0x1.8cp+6"  # 99.0 in hex float
    params["expression_dict"] = tampered_expr_dict

    with pytest.raises(MathematicalExpressionError, match="Mathematical identity mismatch"):
        validate_mathematical_hypothesis_parameters(params, strategy_name="mathematical_expression")


def test_tamper_evident_boundary_expression_fingerprint_tampering(sample_lineage):
    """Adversarial test: Mutate expression_fingerprint while keeping expression_dict unchanged -> Fail Closed."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_tamp", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))
    params = dict(hyp.parameters)

    # Tamper with fingerprint string
    params["expression_fingerprint"] = "a" * 64

    with pytest.raises(MathematicalExpressionError, match="Mathematical identity mismatch"):
        validate_mathematical_hypothesis_parameters(params, strategy_name="mathematical_expression")


def test_tamper_evident_boundary_signal_policy_tampering(sample_lineage):
    """Adversarial test: Mutate signal_policy_dict while keeping signal_policy_fingerprint unchanged -> Fail Closed."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_tamp", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy(version="1.0")
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))
    params = dict(hyp.parameters)

    # Tamper with policy dict
    tampered_policy_dict = policy.to_canonical_dict()
    tampered_policy_dict["version"] = "99.0"
    params["signal_policy_dict"] = tampered_policy_dict

    with pytest.raises(MathematicalExpressionError, match="Mathematical signal policy identity mismatch"):
        validate_mathematical_hypothesis_parameters(params, strategy_name="mathematical_expression")


def test_tamper_evident_boundary_missing_required_fields(sample_lineage):
    """Adversarial test: Remove a required mathematical identity field -> Fail Closed."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_tamp", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))

    for required in ("expression_dict", "expression_fingerprint", "candidate_fingerprint", "search_space_fingerprint", "signal_policy_dict", "signal_policy_fingerprint"):
        params = dict(hyp.parameters)
        del params[required]
        with pytest.raises(MathematicalExpressionError, match="Tampered or incomplete mathematical hypothesis"):
            validate_mathematical_hypothesis_parameters(params, strategy_name="mathematical_expression")


def test_tamper_evident_boundary_wrong_strategy_name(sample_lineage):
    """Adversarial test: Supply non-mathematical strategy name with mathematical parameters -> Fail Closed."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_tamp", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))

    with pytest.raises(MathematicalExpressionError, match="not the canonical research-only mathematical strategy"):
        validate_mathematical_hypothesis_parameters(hyp.parameters, strategy_name="momentum")


# --- 3. POINT VS SERIES TEMPORAL INVARIANT TEST FOR ADAPTER ---

def test_adapter_point_vs_series_temporal_invariant(sample_lineage, synthetic_ohlcv):
    """Adversarial proof: expression.evaluate(df, t) == expression.evaluate(df, t=None).iloc[t] for all valid t and signals match sign policy."""
    ds, ea, cp = sample_lineage
    # Expression: (close[t-1] - close[t-3]) * volume[t-2]
    c1 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    c2 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=3, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    diff = MathematicalExpression(operator=MathematicalOperator.SUB, children=(c1, c2), lag=0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    vol = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="volume", lag=2, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    nested_expr = MathematicalExpression(operator=MathematicalOperator.MUL, children=(diff, vol), lag=0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    assert nested_expr.max_lookback == 3

    policy = MathematicalSignalInterpretationPolicy()
    df_res = generate_mathematical_expression_signal(synthetic_ohlcv, expression_obj=nested_expr, signal_policy_obj=policy)

    expr_series = nested_expr.evaluate(synthetic_ohlcv, t=None)

    # Warmup period check
    for row in range(nested_expr.max_lookback):
        assert df_res["signal"].iloc[row] == 0

    # Valid post-warmup rows check
    for row in range(nested_expr.max_lookback, len(synthetic_ohlcv)):
        pt_val = nested_expr.evaluate(synthetic_ohlcv, t=row)
        series_val = expr_series.iloc[row]

        assert pt_val == series_val
        assert df_res["expression_value"].iloc[row] == series_val

        expected_sig = policy.evaluate_value(series_val)
        assert df_res["signal"].iloc[row] == expected_sig


# --- 4. ADAPTER-LEVEL PROTECTED OPERATOR COVERAGE TESTS ---

def test_adapter_protected_div_domain_violation_fails_closed(sample_lineage, synthetic_ohlcv):
    """Test PROTECTED_DIV with near-zero denominator through real adapter -> Fail Closed."""
    ds, ea, cp = sample_lineage
    df_bad = synthetic_ohlcv.copy()
    df_bad.loc[10, "volume"] = 0.0

    c_close = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    c_vol = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="volume", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    div_expr = MathematicalExpression(operator=MathematicalOperator.PROTECTED_DIV, children=(c_close, c_vol), dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    with pytest.raises(MathematicalDomainError, match="Protected division by zero"):
        generate_mathematical_expression_signal(df_bad, expression_obj=div_expr)


def test_adapter_protected_log_domain_violation_fails_closed(sample_lineage, synthetic_ohlcv):
    """Test PROTECTED_LOG with non-positive value (<= 0) through real adapter -> Fail Closed."""
    ds, ea, cp = sample_lineage
    df_bad = synthetic_ohlcv.copy()
    df_bad.loc[10, "close"] = -5.0

    c_close = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    log_expr = MathematicalExpression(operator=MathematicalOperator.PROTECTED_LOG, children=(c_close,), dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    with pytest.raises(MathematicalDomainError, match="Protected log domain violation"):
        generate_mathematical_expression_signal(df_bad, expression_obj=log_expr)


def test_adapter_protected_sqrt_domain_violation_fails_closed(sample_lineage, synthetic_ohlcv):
    """Test PROTECTED_SQRT with negative value (< 0) through real adapter -> Fail Closed."""
    ds, ea, cp = sample_lineage
    df_bad = synthetic_ohlcv.copy()
    df_bad.loc[10, "close"] = -10.0

    c_close = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    sqrt_expr = MathematicalExpression(operator=MathematicalOperator.PROTECTED_SQRT, children=(c_close,), dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    with pytest.raises(MathematicalDomainError, match="Protected sqrt domain violation"):
        generate_mathematical_expression_signal(df_bad, expression_obj=sqrt_expr)


# --- 5. ADAPTER-LEVEL NON-FINITE FAIL-CLOSED TESTS ---

def test_adapter_non_finite_outputs_fail_closed(sample_lineage, synthetic_ohlcv):
    """Test NaN, +Inf, -Inf in post-warmup evaluation fail closed through adapter."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    # 1. NaN in post-warmup close -> Fail Closed
    df_nan = synthetic_ohlcv.copy()
    df_nan.loc[5, "close"] = np.nan
    with pytest.raises(MathematicalEvaluationError, match="Non-finite value"):
        generate_mathematical_expression_signal(df_nan, expression_obj=expr)

    # 2. +Inf in post-warmup close -> Fail Closed
    df_inf = synthetic_ohlcv.copy()
    df_inf.loc[5, "close"] = np.inf
    with pytest.raises(MathematicalEvaluationError, match="Non-finite value"):
        generate_mathematical_expression_signal(df_inf, expression_obj=expr)

    # 3. -Inf in post-warmup close -> Fail Closed
    df_neginf = synthetic_ohlcv.copy()
    df_neginf.loc[5, "close"] = -np.inf
    with pytest.raises(MathematicalEvaluationError, match="Non-finite value"):
        generate_mathematical_expression_signal(df_neginf, expression_obj=expr)


# --- 6. COMPLETE IDENTITY CHAIN TEST ---

def test_complete_identity_chain_preservation(sample_lineage, synthetic_ohlcv):
    """Test full identity chain preservation across AST, Candidate, Hypothesis, Spec, and Evidence."""
    ds, ea, cp = sample_lineage

    c1 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=1, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    c2 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", lag=2, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    expr = MathematicalExpression(operator=MathematicalOperator.SUB, children=(c1, c2), dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    space = MathematicalSearchSpace(
        search_id="space_chain", max_depth=5, max_node_count=10, max_window_size=50,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)
    wf_protocol = WalkForwardProtocol(train_size=40, test_size=15)

    evidence = run_mathematical_research_experiment(
        candidate=cand,
        df=synthetic_ohlcv,
        walk_forward_protocol=wf_protocol,
        criteria=DiscoveryCriteria(min_observations_is=20, min_observations_oos=10),
    )

    # Verify complete identity chain
    assert expr.fingerprint == evidence.spec.parameters["expression_fingerprint"]
    assert cand.fingerprint == evidence.spec.parameters["candidate_fingerprint"]
    assert space.fingerprint == evidence.spec.parameters["search_space_fingerprint"]
    assert policy.fingerprint == evidence.spec.parameters["signal_policy_fingerprint"]
    assert evidence.spec.fingerprint == evidence.experiment_fingerprint


# --- 7. SEARCH-SPACE CONSTITUTION PRE-VALIDATION FAILURES ---

def test_search_space_pre_validation_failures_before_execution(sample_lineage):
    """Adversarial tests: Invalid candidates are rejected before strategy adapter execution or evidence creation."""
    ds, ea, cp = sample_lineage

    # Allowed features strictly ('close',)
    space = MathematicalSearchSpace(
        search_id="space_gov", max_depth=3, max_node_count=5, allowed_features=("close",),
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    policy = MathematicalSignalInterpretationPolicy()

    # Disallowed feature reference 'volume'
    expr_disallowed_feat = MathematicalExpression(
        operator=MathematicalOperator.FEATURE, feature_name="volume", lag=1,
        dataset_scope=ds, execution_assumptions=ea, code_provenance=cp
    )
    cand_invalid = MathematicalExpressionCandidate(expr_disallowed_feat, space, policy, ds, ea, cp)

    wf_protocol = WalkForwardProtocol(40, 15)

    with pytest.raises(SearchSpaceValidationError, match="not in search space allowed features"):
        run_mathematical_research_experiment(candidate=cand_invalid, walk_forward_protocol=wf_protocol)


# --- 8. GOVERNANCE LIFECYCLE AND ACCEPTANCE TESTS ---

def test_unaccepted_hypothesis_execution_fails_closed(sample_lineage, synthetic_ohlcv):
    """Test that a ResearchHypothesis in status GENERATED cannot enter run_research_experiment directly."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_gov", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    raw_hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))
    assert raw_hyp.status == HypothesisStatus.GENERATED

    research_reg = create_mathematical_research_registry()

    # Direct execution of GENERATED hypothesis must fail closed
    with pytest.raises(Exception, match="is not accepted for research execution"):
        run_research_experiment(spec=raw_hyp, df=synthetic_ohlcv, registry=research_reg)


# --- 9. RESEARCH-ONLY REGISTRY ISOLATION TESTS ---

def test_research_registry_isolation_and_builtin_strategy_preservation():
    """Verify built-in DEFAULT_REGISTRY remains pristine while research registry has mathematical_expression."""
    assert not DEFAULT_REGISTRY.contains("mathematical_expression")
    assert DEFAULT_REGISTRY.contains("baseline")
    assert DEFAULT_REGISTRY.contains("breakout")
    assert DEFAULT_REGISTRY.contains("momentum")

    research_reg = create_mathematical_research_registry()
    assert research_reg.contains("mathematical_expression")
    assert not DEFAULT_REGISTRY.contains("mathematical_expression")


# --- 10. ANTI-RECURRENCE SENTINEL PRE-EVALUATION TEST ---

def test_anti_recurrence_tampered_candidate_fails_before_ast_evaluation(sample_lineage, synthetic_ohlcv, monkeypatch):
    """Anti-Recurrence Gate Test: Prove that tampered candidates fail closed BEFORE MathematicalExpression.evaluate is ever called."""
    ds, ea, cp = sample_lineage
    expr = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    space = MathematicalSearchSpace(search_id="space_tamp", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)
    policy = MathematicalSignalInterpretationPolicy()
    cand = MathematicalExpressionCandidate(expr, space, policy, ds, ea, cp)

    hyp = cand.to_hypothesis(walk_forward_protocol=WalkForwardProtocol(40, 15))

    # Tamper with hypothesis parameters (corrupt expression_fingerprint)
    tampered_params = dict(hyp.parameters)
    tampered_params["expression_fingerprint"] = "corrupted_fp_123456"

    # Sentinel spy on MathematicalExpression.evaluate
    eval_spy = MagicMock(side_effect=AssertionError("MathematicalExpression.evaluate should NOT be called on tampered inputs!"))
    monkeypatch.setattr(MathematicalExpression, "evaluate", eval_spy)

    research_reg = create_mathematical_research_registry()

    # Attempt execution with tampered parameters passed to generate_mathematical_expression_signal
    with pytest.raises(MathematicalExpressionError, match="Mathematical identity mismatch"):
        generate_mathematical_expression_signal(synthetic_ohlcv, **tampered_params)

    # Assert sentinel evaluate method was never invoked
    eval_spy.assert_not_called()
