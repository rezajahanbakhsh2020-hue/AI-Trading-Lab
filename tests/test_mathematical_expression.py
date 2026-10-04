"""Tests for Canonical Mathematical Expression + Search-Space Constitution Layer.

Covers Four-Layer Beta Plus Testing Requirements:
- Stage 1: Local correctness (construction, AST nodes, canonical serialization, fingerprinting, search space, search budget)
- Stage 2: Temporal & Adversarial (negative lag, lookback propagation, safe operators, domain violations, NaN/Inf, mutation, lossless numeric precision, constant_precision enforcement, missing/mismatched lineage fail-closed)
- Stage 3: Research Integration (DatasetScope, ExecutionAssumptions, CodeProvenance compatibility, roundtrip serialization with hex floats)
- Stage 4: Boundary & Anti-Recurrence (Zero coupling to ProductionDecision/live execution/risk/P2, fail-closed governance)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import pytest
import numpy as np
import pandas as pd

from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
)
from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalExpressionError,
    MathematicalOperator,
    MathematicalSearchSpace,
    MathematicalDomainError,
    MathematicalEvaluationError,
    SearchSpaceValidationError,
)


# --- STAGE 1: LOCAL CORRECTNESS TESTS ---

def test_valid_expression_construction():
    """Test valid construction of leaf and nested AST nodes."""
    const_node = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=1.5,
    )
    assert const_node.operator == MathematicalOperator.CONSTANT
    assert const_node.constant_value == 1.5
    assert const_node.node_count == 1
    assert const_node.depth == 1
    assert const_node.feature_references == ()

    feat_node = MathematicalExpression(
        operator=MathematicalOperator.FEATURE,
        feature_name="close",
        lag=2,
    )
    assert feat_node.operator == MathematicalOperator.FEATURE
    assert feat_node.feature_name == "close"
    assert feat_node.lag == 2
    assert feat_node.max_lookback == 2
    assert feat_node.feature_references == ("close",)

    add_node = MathematicalExpression(
        operator=MathematicalOperator.ADD,
        children=(const_node, feat_node),
    )
    assert add_node.node_count == 3
    assert add_node.depth == 2
    assert add_node.feature_references == ("close",)
    assert add_node.max_lookback == 2


def test_immutable_expression_mutation_attempt():
    """Test that expression AST nodes are frozen and reject attribute assignment."""
    node = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=10.0,
    )
    with pytest.raises(Exception):  # FrozenInstanceError / TypeError
        node.constant_value = 20.0  # type: ignore


def test_deterministic_canonical_serialization_and_fingerprint():
    """Test that equivalent structural representations serialize deterministically and yield identical SHA-256 fingerprints."""
    node1 = MathematicalExpression(
        operator=MathematicalOperator.ADD,
        children=(
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close"),
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=5.0),
        ),
        random_seed=42,
    )

    node2 = MathematicalExpression(
        operator=MathematicalOperator.ADD,
        children=(
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close"),
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=5.0),
        ),
        random_seed=42,
    )

    assert node1.to_canonical_json() == node2.to_canonical_json()
    assert node1.fingerprint == node2.fingerprint
    assert len(node1.fingerprint) == 64


def test_structurally_different_expressions_have_different_identities():
    """Test that structural changes alter fingerprint."""
    node1 = MathematicalExpression(
        operator=MathematicalOperator.ADD,
        children=(
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close"),
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=5.0),
        ),
    )

    node2 = MathematicalExpression(
        operator=MathematicalOperator.ADD,
        children=(
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close"),
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=5.1),
        ),
    )

    assert node1.fingerprint != node2.fingerprint


def test_lossless_numeric_fingerprint_precision():
    """Test that distinct finite constants that differ beyond 12 decimal places receive different fingerprints."""
    c1 = 1.0000000000000002
    c2 = 1.0000000000000004

    node1 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=c1)
    node2 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=c2)

    assert node1.fingerprint != node2.fingerprint


def test_malformed_expression_rejection():
    """Test rejection of malformed or empty nodes."""
    with pytest.raises(MathematicalExpressionError, match="CONSTANT node requires constant_value"):
        MathematicalExpression(operator=MathematicalOperator.CONSTANT)

    with pytest.raises(MathematicalExpressionError, match="FEATURE node requires non-empty feature_name"):
        MathematicalExpression(operator=MathematicalOperator.FEATURE)

    with pytest.raises(MathematicalExpressionError, match="requires arity 2"):
        MathematicalExpression(operator=MathematicalOperator.ADD, children=())


def test_invalid_arity_rejection():
    """Test rejection when child count does not match operator arity."""
    c = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0)

    # ADD requires 2 children, give 1
    with pytest.raises(MathematicalExpressionError, match="requires arity 2"):
        MathematicalExpression(operator=MathematicalOperator.ADD, children=(c,))

    # NEG requires 1 child, give 2
    with pytest.raises(MathematicalExpressionError, match="requires arity 1"):
        MathematicalExpression(operator=MathematicalOperator.NEG, children=(c, c))


def test_unknown_operator_rejection():
    """Test rejection of unknown operator string."""
    with pytest.raises(MathematicalExpressionError, match="Unknown operator"):
        MathematicalExpression(operator="INVALID_OP", children=())  # type: ignore


def test_invalid_constants_rejection():
    """Test rejection of NaN and infinity constants."""
    with pytest.raises(MathematicalExpressionError, match="rejects NaN/Inf value"):
        MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=float("nan"))

    with pytest.raises(MathematicalExpressionError, match="rejects NaN/Inf value"):
        MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=float("inf"))


def test_invalid_search_space_rejection():
    """Test rejection of invalid search space parameters."""
    with pytest.raises(SearchSpaceValidationError, match="search_id must be a non-empty string"):
        MathematicalSearchSpace(search_id="")

    with pytest.raises(SearchSpaceValidationError, match="max_depth must be an integer >= 1"):
        MathematicalSearchSpace(search_id="s1", max_depth=0)

    with pytest.raises(SearchSpaceValidationError, match="constant_bounds min_val"):
        MathematicalSearchSpace(search_id="s1", constant_bounds=(100.0, -100.0))

    with pytest.raises(SearchSpaceValidationError, match="max_search_budget must be a positive integer"):
        MathematicalSearchSpace(search_id="s1", max_search_budget=0)


def test_deterministic_search_space_fingerprint_and_search_budget():
    """Test deterministic fingerprinting and max_search_budget participation in search space identity."""
    s1 = MathematicalSearchSpace(search_id="space1", max_depth=3, max_search_budget=500, random_seed=123)
    s2 = MathematicalSearchSpace(search_id="space1", max_depth=3, max_search_budget=500, random_seed=123)
    s3 = MathematicalSearchSpace(search_id="space1", max_depth=3, max_search_budget=1000, random_seed=123)

    assert s1.to_canonical_json() == s2.to_canonical_json()
    assert s1.fingerprint == s2.fingerprint
    assert s1.fingerprint != s3.fingerprint


# --- STAGE 2: TEMPORAL & ADVERSARIAL TESTS ---

def test_future_negative_lag_rejection():
    """Test strict rejection of negative lags."""
    with pytest.raises(MathematicalExpressionError, match="lag cannot be negative"):
        MathematicalExpression(
            operator=MathematicalOperator.FEATURE,
            feature_name="close",
            lag=-1,
        )


def test_invalid_lookback_warmup_rejection_at_evaluation():
    """Test that evaluating at point t prior to max_lookback warmup fails closed."""
    expr = MathematicalExpression(
        operator=MathematicalOperator.FEATURE,
        feature_name="close",
        lag=5,
    )
    df = pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]})

    # t=4 with lag=5 requires t - lag = -1, which is invalid
    with pytest.raises(MathematicalEvaluationError, match="Insufficient lookback history"):
        expr.evaluate(df, t=4)

    # t=5 with lag=5 evaluates at index 0 (value 10.0)
    assert expr.evaluate(df, t=5) == 10.0


def test_temporal_dependency_propagation_through_nested_expressions():
    """Test max_lookback computation through nested AST tree."""
    leaf1 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="high", lag=2)
    leaf2 = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="low", lag=4)
    parent = MathematicalExpression(
        operator=MathematicalOperator.ADD,
        children=(leaf1, leaf2),
        lag=3,
    )

    # parent lag 3 + max(child lag 2, child lag 4) = 7
    assert parent.max_lookback == 7
    assert parent.warmup_requirement == 7


def test_constant_precision_enforcement_in_search_space():
    """Test that constant_precision is enforced strictly during search space validation."""
    space = MathematicalSearchSpace(
        search_id="s_prec",
        constant_precision=0.001,
    )

    # 1.502 aligns with 0.001 precision -> valid
    e_valid = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.502)
    space.validate_expression(e_valid)

    # 1.5025 does NOT align with 0.001 precision -> invalid
    e_invalid = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.5025)
    with pytest.raises(SearchSpaceValidationError, match="violates search space constant_precision constraint"):
        space.validate_expression(e_invalid)


def test_fail_closed_lineage_validation_in_search_space():
    """Test that search space validation fails closed if required lineage is missing or mismatched."""
    ds1 = DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31")
    ds2 = DatasetScope("ds2", "XAUUSD", "1h", "2023-01-01", "2023-12-31")
    ea1 = ExecutionAssumptions(0.0001, 0.0002, 10.0)
    cp1 = CodeProvenance("commit_sha_123")

    space = MathematicalSearchSpace(
        search_id="s_lineage",
        dataset_scope=ds1,
        execution_assumptions=ea1,
        code_provenance=cp1,
    )

    # 1. Valid matching lineage -> passes
    e_valid = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=1.0,
        dataset_scope=ds1,
        execution_assumptions=ea1,
        code_provenance=cp1,
    )
    space.validate_expression(e_valid)

    # 2. Missing DatasetScope -> fails closed
    e_missing_ds = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=1.0,
        execution_assumptions=ea1,
        code_provenance=cp1,
    )
    with pytest.raises(SearchSpaceValidationError, match="missing DatasetScope required by search space"):
        space.validate_expression(e_missing_ds)

    # 3. Mismatched DatasetScope -> fails closed
    e_mismatch_ds = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=1.0,
        dataset_scope=ds2,
        execution_assumptions=ea1,
        code_provenance=cp1,
    )
    with pytest.raises(SearchSpaceValidationError, match="DatasetScope does not match search space DatasetScope"):
        space.validate_expression(e_mismatch_ds)


def test_unsafe_protected_division_domain_violation():
    """Test fail-closed behavior on division by zero or near-zero denominator."""
    c10 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=10.0)
    czero = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=0.0)
    div_expr = MathematicalExpression(operator=MathematicalOperator.PROTECTED_DIV, children=(c10, czero))

    df = pd.DataFrame({"dummy": [1.0, 2.0]})

    with pytest.raises(MathematicalDomainError, match="Protected division by zero"):
        div_expr.evaluate(df, t=0)

    with pytest.raises(MathematicalDomainError, match="Protected division by zero"):
        div_expr.evaluate(df, t=None)


def test_invalid_log_domain():
    """Test fail-closed behavior on log of zero or negative numbers."""
    cneg = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=-5.0)
    log_expr = MathematicalExpression(operator=MathematicalOperator.PROTECTED_LOG, children=(cneg,))

    df = pd.DataFrame({"dummy": [1.0]})

    with pytest.raises(MathematicalDomainError, match="Protected log domain violation"):
        log_expr.evaluate(df, t=0)


def test_invalid_sqrt_domain():
    """Test fail-closed behavior on sqrt of negative numbers."""
    cneg = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=-1.0)
    sqrt_expr = MathematicalExpression(operator=MathematicalOperator.PROTECTED_SQRT, children=(cneg,))

    df = pd.DataFrame({"dummy": [1.0]})

    with pytest.raises(MathematicalDomainError, match="Protected sqrt domain violation"):
        sqrt_expr.evaluate(df, t=0)


def test_nan_inf_propagation_rejection():
    """Test rejection when evaluation encounters NaN/Inf values in input data."""
    feat = MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close")
    df_nan = pd.DataFrame({"close": [10.0, float("nan"), 12.0]})

    with pytest.raises(MathematicalEvaluationError, match="Non-finite value"):
        feat.evaluate(df_nan, t=1)

    with pytest.raises(MathematicalEvaluationError, match="Non-finite values encountered"):
        feat.evaluate(df_nan, t=None)


def test_cyclic_expression_rejection():
    """Test that self-referencing / cyclic structures fail closed during validation."""
    n1 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0)
    object.__setattr__(n1, "children", (n1,))

    with pytest.raises(MathematicalExpressionError, match="Cycle detected"):
        n1._check_cycles()


def test_semantically_relevant_lineage_changes_altering_fingerprint():
    """Test that changing DatasetScope, ExecutionAssumptions, or CodeProvenance alters fingerprint."""
    ds1 = DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31")
    ds2 = DatasetScope("ds2", "XAUUSD", "1h", "2023-01-01", "2023-12-31")

    e1 = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=5.0,
        dataset_scope=ds1,
    )
    e2 = MathematicalExpression(
        operator=MathematicalOperator.CONSTANT,
        constant_value=5.0,
        dataset_scope=ds2,
    )

    assert e1.fingerprint != e2.fingerprint


def test_dataset_scope_mismatch_cannot_be_silently_accepted():
    """Test that combining AST nodes with mismatched DatasetScopes fails closed."""
    ds1 = DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31")
    ds2 = DatasetScope("ds2", "XAUUSD", "1h", "2023-01-01", "2023-12-31")

    child1 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds1)
    child2 = MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=2.0, dataset_scope=ds2)

    with pytest.raises(MathematicalExpressionError, match="DatasetScope mismatch"):
        MathematicalExpression(
            operator=MathematicalOperator.ADD,
            children=(child1, child2),
            dataset_scope=ds1,
        )


# --- STAGE 3: RESEARCH INTEGRATION TESTS ---

def test_research_lineage_types_compatibility():
    """Test integration with existing research identity classes."""
    ds = DatasetScope("ds_xau", "XAUUSD", "1h", "2023-01-01", "2023-06-01")
    ea = ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0002, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="a1b2c3d4e5f6", repository_status="clean", author="researcher")

    expr = MathematicalExpression(
        operator=MathematicalOperator.FEATURE,
        feature_name="close",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    assert expr.dataset_scope == ds
    assert expr.execution_assumptions == ea
    assert expr.code_provenance == cp

    space = MathematicalSearchSpace(
        search_id="s_integrated",
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    assert space.dataset_scope == ds
    assert space.execution_assumptions == ea
    assert space.code_provenance == cp


def test_reproducibility_across_repeated_construction():
    """Test identical identity across repeated separate constructions."""
    ds = DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31")

    e1 = MathematicalExpression(
        operator=MathematicalOperator.MUL,
        children=(
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="volume", dataset_scope=ds),
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=0.5, dataset_scope=ds),
        ),
        dataset_scope=ds,
        random_seed=999,
    )

    e2 = MathematicalExpression(
        operator=MathematicalOperator.MUL,
        children=(
            MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="volume", dataset_scope=ds),
            MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=0.5, dataset_scope=ds),
        ),
        dataset_scope=ds,
        random_seed=999,
    )

    assert e1 == e2
    assert e1.fingerprint == e2.fingerprint


def test_roundtrip_canonical_serialization_deserialization():
    """Test lossless reconstruction via from_canonical_dict."""
    ds = DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31")
    ea = ExecutionAssumptions(transaction_cost=0.0001, slippage=0.0002, latency_ms=10.0)
    cp = CodeProvenance(commit_sha="fedcba9876543210")

    original = MathematicalExpression(
        operator=MathematicalOperator.PROTECTED_LOG,
        children=(
            MathematicalExpression(
                operator=MathematicalOperator.FEATURE,
                feature_name="close",
                lag=3,
                dataset_scope=ds,
                execution_assumptions=ea,
                code_provenance=cp,
            ),
        ),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    d = original.to_canonical_dict()
    reconstructed = MathematicalExpression.from_canonical_dict(d)

    assert reconstructed == original
    assert reconstructed.fingerprint == original.fingerprint


# --- STAGE 4: BOUNDARY & ANTI-RECURRENCE TESTS ---

def test_zero_production_live_project2_coupling():
    """Verify in a clean process that importing mathematical_expression does NOT import ProductionDecision, live execution, or Project2 modules."""
    check_code = """
import sys
import src.evaluation.mathematical_expression

loaded_modules = set(sys.modules.keys())
forbidden = [
    'src.evaluation.live_production_decision',
    'src.evaluation.live_execution_runtime',
    'src.integration.project2_publisher',
    'src.evaluation.live_runtime',
    'src.evaluation.live_decision_lifecycle',
]
for mod in forbidden:
    if mod in loaded_modules:
        print(f"FORBIDDEN:{mod}")
        sys.exit(1)
print("CLEAN")
"""
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Coupling check failed output:\n{result.stdout}\n{result.stderr}"
    assert "CLEAN" in result.stdout


def test_search_space_validation_fails_closed_before_execution():
    """Test that search space validation catches violations strictly before any evaluation can occur."""
    ds = DatasetScope("ds1", "XAUUSD", "1h", "2023-01-01", "2023-12-31")
    space = MathematicalSearchSpace(
        search_id="space_strict",
        max_depth=2,
        allowed_features=("close",),
        dataset_scope=ds,
    )

    deep_expr = MathematicalExpression(
        operator=MathematicalOperator.NEG,
        children=(
            MathematicalExpression(
                operator=MathematicalOperator.ADD,
                children=(
                    MathematicalExpression(operator=MathematicalOperator.FEATURE, feature_name="close", dataset_scope=ds),
                    MathematicalExpression(operator=MathematicalOperator.CONSTANT, constant_value=1.0, dataset_scope=ds),
                ),
                dataset_scope=ds,
            ),
        ),
        dataset_scope=ds,
    )

    with pytest.raises(SearchSpaceValidationError, match="depth 3 exceeds search space limit of 2"):
        space.validate_expression(deep_expr)
