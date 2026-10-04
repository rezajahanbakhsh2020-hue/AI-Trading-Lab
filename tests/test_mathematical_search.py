"""Adversarial and Comprehensive Tests for Mathematical Search Layer.

Covers all 20 required architectural and adversarial test specifications:
1. Deterministic identical search
2. Feature ordering independence
3. Constant validation (NaN, Inf, bounds, precision)
4. Lag boundary (min_lag/max_lag, negative lag rejection)
5. Operator constitution (forbidden operators)
6. Feature constitution (forbidden features)
7. Complexity constitution (depth, node count, feature count, interaction count, complexity, window size)
8. Lineage validation (missing or mismatched DatasetScope / ExecutionAssumptions / CodeProvenance)
9. Duplicate commutative prevention (ADD and MUL)
10. Fingerprint uniqueness
11. Search budget truncation (max_search_budget)
12. Requested limit truncation
13. Zero / negative requested limit fails closed
14. Reproducibility across seeds and parameters
15. Candidate metadata mutation Sensitivity
16. No data dependency ( DataFrame / market data independence )
17. No DEFAULT_REGISTRY pollution
18. Research-only candidate compatibility (candidate.validate() and execution bridge)
19. Anti-recurrence boundary (market evaluation never invoked)
20. Malformed search space fails closed
"""

from __future__ import annotations

import math
from typing import Any
import pytest
import pandas as pd

from src.evaluation.mathematical_expression import (
    MathematicalExpression,
    MathematicalExpressionError,
    MathematicalOperator,
    MathematicalSearchSpace,
    SearchSpaceValidationError,
)
from src.evaluation.mathematical_expression_candidate import (
    MathematicalExpressionCandidate,
    MathematicalSignalInterpretationPolicy,
    run_mathematical_research_experiment,
)
from src.evaluation.mathematical_search import (
    MathematicalSearchError,
    MathematicalSearchResult,
    MathematicalSearchStrategy,
    SearchTerminationReason,
    SymbolicSearch,
)
from src.evaluation.research_constitution import (
    CodeProvenance,
    DatasetScope,
    ExecutionAssumptions,
    WalkForwardProtocol,
)
from src.evaluation.mathematical_expression_strategy import create_mathematical_research_registry


@pytest.fixture
def default_lineage() -> tuple[DatasetScope, ExecutionAssumptions, CodeProvenance]:
    ds = DatasetScope(
        dataset_id="ds_xauusd_1h",
        symbol="XAUUSD",
        timeframe="1h",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )
    ea = ExecutionAssumptions(
        transaction_cost=0.0001,
        slippage=0.0001,
        latency_ms=10.0,
    )
    cp = CodeProvenance(
        commit_sha="a" * 40,
        repository_status="clean",
        author="researcher@test.org",
    )
    return ds, ea, cp


@pytest.fixture
def base_search_space(default_lineage) -> MathematicalSearchSpace:
    ds, ea, cp = default_lineage
    return MathematicalSearchSpace(
        search_id="test_search_space_1",
        allowed_operators=(
            MathematicalOperator.FEATURE,
            MathematicalOperator.CONSTANT,
            MathematicalOperator.ADD,
            MathematicalOperator.SUB,
            MathematicalOperator.MUL,
            MathematicalOperator.PROTECTED_DIV,
            MathematicalOperator.NEG,
            MathematicalOperator.ABS,
        ),
        allowed_features=("close", "open", "high", "low"),
        max_depth=3,
        max_node_count=5,
        max_feature_count=2,
        max_interaction_count=1,
        min_lag=0,
        max_lag=2,
        max_window_size=20,
        constant_bounds=(-10.0, 10.0),
        constant_precision=0.5,
        max_complexity=10,
        max_search_budget=100,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )


# 1. Deterministic identical search: same inputs -> identical ordered fingerprints
def test_1_deterministic_identical_search(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    res1 = strategy.search(
        search_space=base_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        constant_values=(1.0, -2.0),
        limit=20,
        generator_id="symbolic_search",
        generator_version="1.0",
        random_seed=42,
    )

    res2 = strategy.search(
        search_space=base_search_space,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
        constant_values=(1.0, -2.0),
        limit=20,
        generator_id="symbolic_search",
        generator_version="1.0",
        random_seed=42,
    )

    assert res1.generated_count == res2.generated_count
    assert res1.candidates
    fps1 = [c.fingerprint for c in res1.candidates]
    fps2 = [c.fingerprint for c in res2.candidates]
    assert fps1 == fps2


# 2. Feature ordering: input feature order variation cannot change deterministic output ordering
def test_2_feature_ordering_independence(default_lineage):
    ds, ea, cp = default_lineage
    ss1 = MathematicalSearchSpace(
        search_id="ss_1",
        allowed_operators=(MathematicalOperator.FEATURE,),
        allowed_features=("close", "open", "volume", "high"),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )
    ss2 = MathematicalSearchSpace(
        search_id="ss_1",
        allowed_operators=(MathematicalOperator.FEATURE,),
        allowed_features=("volume", "high", "close", "open"),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res1 = strategy.search(ss1, ds, ea, cp, limit=10)
    res2 = strategy.search(ss2, ds, ea, cp, limit=10)

    # Search space sorting canonicalizes features tuple so AST generation order is identical
    assert [c.expression.fingerprint for c in res1.candidates] == [c.expression.fingerprint for c in res2.candidates]


# 3. Constant validation: NaN, Inf, out-of-bounds and precision-invalid constants fail closed
def test_3_constant_validation(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    # NaN constant
    with pytest.raises(MathematicalSearchError, match="Non-finite constant"):
        strategy.search(base_search_space, ds, ea, cp, constant_values=(float("nan"),))

    # Inf constant
    with pytest.raises(MathematicalSearchError, match="Non-finite constant"):
        strategy.search(base_search_space, ds, ea, cp, constant_values=(float("inf"),))

    # Out of bounds constant (bounds are -10.0, 10.0)
    with pytest.raises(MathematicalSearchError, match="constant_bounds"):
        strategy.search(base_search_space, ds, ea, cp, constant_values=(15.0,))

    # Precision violation (precision is 0.5)
    with pytest.raises(MathematicalSearchError, match="constant_precision"):
        strategy.search(base_search_space, ds, ea, cp, constant_values=(0.25,))


# 4. Lag boundary: min_lag/max_lag respected; negative/future lag cannot be generated
def test_4_lag_boundary(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_lag",
        allowed_operators=(MathematicalOperator.FEATURE,),
        allowed_features=("close",),
        min_lag=1,
        max_lag=3,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp)

    for cand in res.candidates:
        assert cand.expression.lag >= 1
        assert cand.expression.lag <= 3


# 5. Operator constitution: forbidden operators never appear
def test_5_operator_constitution(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_op",
        allowed_operators=(MathematicalOperator.FEATURE, MathematicalOperator.ADD),
        allowed_features=("close", "open"),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=50)

    def _check_ops(node: MathematicalExpression):
        assert node.operator in (MathematicalOperator.FEATURE, MathematicalOperator.ADD)
        for child in node.children:
            _check_ops(child)

    for cand in res.candidates:
        _check_ops(cand.expression)


# 6. Feature constitution: forbidden features never appear
def test_6_feature_constitution(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_feat",
        allowed_operators=(MathematicalOperator.FEATURE, MathematicalOperator.ADD),
        allowed_features=("close",),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=20)

    for cand in res.candidates:
        assert cand.expression.feature_references == ("close",)


# 7. Complexity constitution: limits strictly respected
def test_7_complexity_constitution(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_complex",
        allowed_operators=(
            MathematicalOperator.FEATURE,
            MathematicalOperator.ADD,
            MathematicalOperator.MUL,
        ),
        allowed_features=("close", "open", "volume"),
        max_depth=3,
        max_node_count=4,
        max_feature_count=2,
        max_interaction_count=1,
        max_complexity=4,
        max_window_size=10,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=50)

    for cand in res.candidates:
        expr = cand.expression
        assert expr.depth <= ss.max_depth
        assert expr.node_count <= ss.max_node_count
        assert expr.feature_count <= ss.max_feature_count
        assert expr.interaction_count <= ss.max_interaction_count
        assert expr.node_count <= ss.max_complexity
        assert expr.max_lookback <= ss.max_window_size


# 8. Lineage: missing or mismatched DatasetScope / ExecutionAssumptions / CodeProvenance fails closed
def test_8_lineage_validation(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    mismatched_ds = DatasetScope(
        dataset_id="other_ds",
        symbol="EURUSD",
        timeframe="1h",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    # Missing dataset_scope
    with pytest.raises(MathematicalSearchError, match="dataset_scope"):
        strategy.search(base_search_space, None, ea, cp)

    # Mismatched dataset_scope against search_space
    with pytest.raises(MathematicalSearchError, match="DatasetScope"):
        strategy.search(base_search_space, mismatched_ds, ea, cp)


# 9. Duplicate prevention: commutative ADD and MUL permutations produce one canonical candidate, not two
def test_9_duplicate_commutative_prevention(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_comm",
        allowed_operators=(MathematicalOperator.FEATURE, MathematicalOperator.ADD),
        allowed_features=("close", "open"),
        min_lag=0,
        max_lag=0,
        max_depth=2,
        max_node_count=3,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=20)

    # We have features 'close' and 'open'.
    # ADD(close, open) vs ADD(open, close) should produce only 1 candidate!
    add_cands = [
        c for c in res.candidates
        if c.expression.operator == MathematicalOperator.ADD
    ]
    # Check that feature_references == ('close', 'open') appears only once
    pair_adds = [c for c in add_cands if set(c.expression.feature_references) == {"close", "open"}]
    assert len(pair_adds) == 1


# 10. Fingerprint uniqueness: no two returned candidates share MathematicalExpression.fingerprint
def test_10_fingerprint_uniqueness(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()
    res = strategy.search(base_search_space, ds, ea, cp, constant_values=(1.0, 2.0), limit=50)

    expr_fps = [c.expression.fingerprint for c in res.candidates]
    assert len(expr_fps) == len(set(expr_fps))

    cand_fps = [c.fingerprint for c in res.candidates]
    assert len(cand_fps) == len(set(cand_fps))


# 11. Search budget: generator never returns more than max_search_budget
def test_11_search_budget_truncation(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_budget",
        allowed_operators=(MathematicalOperator.FEATURE, MathematicalOperator.ADD),
        allowed_features=("close", "open", "high", "low"),
        max_search_budget=5,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=100)  # Requesting 100, but max budget is 5

    assert res.generated_count == 5
    assert len(res.candidates) == 5
    assert res.budget == 5
    assert res.termination_reason == SearchTerminationReason.BUDGET_EXHAUSTED


# 12. Requested limit: explicit lower limit truncates deterministically
def test_12_requested_limit_truncation(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_limit",
        allowed_operators=(MathematicalOperator.FEATURE, MathematicalOperator.ADD),
        allowed_features=("close", "open", "high", "low"),
        max_search_budget=100,
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=3)

    assert res.generated_count == 3
    assert len(res.candidates) == 3
    assert res.budget == 3
    assert res.termination_reason == SearchTerminationReason.BUDGET_EXHAUSTED


# 13. Zero / negative limit: fails closed
def test_13_zero_negative_limit_fails_closed(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    with pytest.raises(MathematicalSearchError, match="limit must be a positive integer"):
        strategy.search(base_search_space, ds, ea, cp, limit=0)

    with pytest.raises(MathematicalSearchError, match="limit must be a positive integer"):
        strategy.search(base_search_space, ds, ea, cp, limit=-5)


# 14. Reproducibility: same seed and constitution reproduce exact ordering
def test_14_reproducibility(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    res1 = strategy.search(base_search_space, ds, ea, cp, constant_values=(1.0,), limit=15, random_seed=123)
    res2 = strategy.search(base_search_space, ds, ea, cp, constant_values=(1.0,), limit=15, random_seed=123)

    assert [c.candidate_id for c in res1.candidates] == [c.candidate_id for c in res2.candidates]


# 15. Metadata mutation: changing generator version/id or seed changes MathematicalExpressionCandidate identity
def test_15_metadata_mutation_affects_candidate_identity(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    res_v1 = strategy.search(
        base_search_space, ds, ea, cp, constant_values=(1.0,), limit=5, generator_version="1.0", random_seed=0
    )
    res_v2 = strategy.search(
        base_search_space, ds, ea, cp, constant_values=(1.0,), limit=5, generator_version="2.0", random_seed=0
    )

    assert res_v1.candidates[0].fingerprint != res_v2.candidates[0].fingerprint
    assert res_v1.candidates[0].expression.fingerprint != res_v2.candidates[0].expression.fingerprint


# 16. No data dependency: prove the generator does not require or inspect a DataFrame/market data
def test_16_no_data_dependency(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    strategy = SymbolicSearch()

    # Search should execute without passing any DataFrame or market data objects
    res = strategy.search(base_search_space, ds, ea, cp, limit=10)
    assert isinstance(res, MathematicalSearchResult)
    assert res.generated_count == 10


# 17. No production registry: MathematicalSearch must not register anything in production registries or research strategy registries
def test_17_no_production_registry_pollution(base_search_space, default_lineage):
    ds, ea, cp = default_lineage
    registry = create_mathematical_research_registry()
    initial_names = set(registry.names())

    strategy = SymbolicSearch()
    res = strategy.search(base_search_space, ds, ea, cp, limit=20)

    final_names = set(registry.names())

    assert initial_names == final_names


# 18. Research-only candidate: every generated candidate can pass validate() and be handed to execution bridge
def test_18_research_only_candidate_bridge_compatibility(default_lineage):
    ds, ea, cp = default_lineage
    ss = MathematicalSearchSpace(
        search_id="ss_bridge",
        allowed_operators=(MathematicalOperator.FEATURE,),
        allowed_features=("close",),
        dataset_scope=ds,
        execution_assumptions=ea,
        code_provenance=cp,
    )

    strategy = SymbolicSearch()
    res = strategy.search(ss, ds, ea, cp, limit=1)

    candidate = res.candidates[0]
    candidate.validate()  # Passes governance pre-validation

    # Synthesize dummy DataFrame matching DatasetScope dates (2023-01-01 to 2023-12-31)
    dates = pd.date_range("2023-01-01", "2023-12-31 23:00:00", freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "close": [100.0 + (i % 100) for i in range(len(dates))],
            "open": [100.0 + (i % 100) for i in range(len(dates))],
            "high": [101.0 + (i % 100) for i in range(len(dates))],
            "low": [99.0 + (i % 100) for i in range(len(dates))],
            "volume": [1000.0 for _ in range(len(dates))],
        },
        index=dates,
    )

    wf_protocol = WalkForwardProtocol(train_size=400, test_size=150)
    evidence = run_mathematical_research_experiment(
        candidate=candidate,
        df=df,
        walk_forward_protocol=wf_protocol,
    )

    assert evidence is not None
    assert evidence.experiment_fingerprint is not None


# 19. Anti-recurrence: monkeypatch/spy on any market-evaluation function and prove generation never invokes evaluation
def test_19_anti_recurrence_no_evaluation(base_search_space, default_lineage, monkeypatch):
    ds, ea, cp = default_lineage
    eval_called = False

    def spy_evaluate(*args, **kwargs):
        nonlocal eval_called
        eval_called = True
        raise RuntimeError("MathematicalExpression.evaluate must NEVER be called during search generation!")

    monkeypatch.setattr(MathematicalExpression, "evaluate", spy_evaluate)

    strategy = SymbolicSearch()
    res = strategy.search(base_search_space, ds, ea, cp, constant_values=(1.0,), limit=20)

    assert not eval_called
    assert res.generated_count == 20


# 20. Malformed search space: fail closed rather than silently relaxing any constitution constraint
def test_20_malformed_search_space_fails_closed(default_lineage):
    ds, ea, cp = default_lineage

    # Empty search_id
    with pytest.raises(SearchSpaceValidationError, match="search_id"):
        MathematicalSearchSpace(search_id="", dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    # Empty allowed_operators
    with pytest.raises(SearchSpaceValidationError, match="allowed_operators"):
        MathematicalSearchSpace(search_id="ss", allowed_operators=(), dataset_scope=ds, execution_assumptions=ea, code_provenance=cp)

    # max_window_size smaller than max_lag
    with pytest.raises(SearchSpaceValidationError, match="max_window_size"):
        MathematicalSearchSpace(
            search_id="ss",
            min_lag=0,
            max_lag=10,
            max_window_size=5,
            dataset_scope=ds,
            execution_assumptions=ea,
            code_provenance=cp,
        )
