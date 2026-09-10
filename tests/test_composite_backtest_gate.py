import numpy as np
import pandas as pd
import pytest

from src.evaluation.composite_backtest import (
    compare_composite_to_components,
    run_composite_backtest,
)
from src.evaluation.composite_backtest_gate import (
    CompositeBacktestGate,
    require_valid_composite_backtest,
    validate_composite_backtest,
)
from src.evaluation.strategy_evaluator import (
    evaluate_all_strategies,
)
from src.strategies.registry import DEFAULT_REGISTRY


def make_data(size: int = 160) -> pd.DataFrame:
    close = pd.Series(
        np.linspace(100.0, 180.0, size)
        + np.sin(np.arange(size) / 5.0),
        name="close",
    )

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=size,
                freq="D",
            ),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        }
    )


def build_inputs():
    data = make_data()

    outputs = {
        name: DEFAULT_REGISTRY.run(name, data)
        for name in DEFAULT_REGISTRY.names()
    }

    evaluations = evaluate_all_strategies(data)

    composite = run_composite_backtest(
        data,
        outputs,
        evaluations,
        top_n=3,
    )

    comparison = compare_composite_to_components(
        composite,
        evaluations,
    )

    return data, outputs, evaluations, composite, comparison


def test_actual_composite_backtest_gate_returns_result():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    gate = validate_composite_backtest(
        composite,
        comparison,
    )

    assert isinstance(gate, CompositeBacktestGate)
    assert gate.component_count == 3
    assert gate.composite_name == "ensemble"
    assert gate.composite_total_return == pytest.approx(
        composite.total_return
    )
    assert gate.composite_sharpe == pytest.approx(
        composite.sharpe_ratio
    )


def test_gate_uses_actual_composite_score():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    gate = validate_composite_backtest(
        composite,
        comparison,
    )

    assert gate.validation.composite_score == pytest.approx(
        composite.evaluation.ranking_score
    )


def test_gate_does_not_accept_without_outperformance():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    forced = comparison.copy()

    component_mask = forced["kind"] == "component"
    forced.loc[
        component_mask,
        "ranking_score",
    ] = composite.evaluation.ranking_score + 1.0

    gate = validate_composite_backtest(
        composite,
        forced,
    )

    assert gate.accepted is False

    with pytest.raises(ValueError, match="failed"):
        require_valid_composite_backtest(gate)


def test_gate_can_accept_when_actual_score_beats_components():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    forced = comparison.copy()

    component_mask = forced["kind"] == "component"
    forced.loc[
        component_mask,
        "ranking_score",
    ] = composite.evaluation.ranking_score - 0.01

    gate = validate_composite_backtest(
        composite,
        forced,
    )

    assert gate.accepted is True
    assert require_valid_composite_backtest(gate) is gate


def test_minimum_improvement_is_enforced():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    forced = comparison.copy()

    component_mask = forced["kind"] == "component"
    forced.loc[
        component_mask,
        "ranking_score",
    ] = composite.evaluation.ranking_score - 0.01

    gate = validate_composite_backtest(
        composite,
        forced,
        minimum_improvement=0.02,
    )

    assert gate.accepted is False


def test_invalid_composite_type_is_rejected():
    _, _, _, _, comparison = build_inputs()

    with pytest.raises(TypeError):
        validate_composite_backtest(
            object(),
            comparison,
        )


def test_missing_composite_row_is_rejected():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    comparison = comparison[
        comparison["kind"] != "composite"
    ].copy()

    with pytest.raises(
        ValueError,
        match="composite row",
    ):
        validate_composite_backtest(
            composite,
            comparison,
        )


def test_missing_component_rows_are_rejected():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    comparison = comparison[
        comparison["kind"] != "component"
    ].copy()

    with pytest.raises(
        ValueError,
        match="at least one component",
    ):
        validate_composite_backtest(
            composite,
            comparison,
        )


def test_negative_minimum_improvement_is_rejected():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_composite_backtest(
            composite,
            comparison,
            minimum_improvement=-0.1,
        )


def test_gate_as_dict_contains_actual_metrics():
    (
        _,
        _,
        _,
        composite,
        comparison,
    ) = build_inputs()

    gate = validate_composite_backtest(
        composite,
        comparison,
    )

    payload = gate.as_dict()

    assert payload["composite_name"] == "ensemble"
    assert payload["component_count"] == 3
    assert payload["composite_total_return"] == pytest.approx(
        composite.total_return
    )
    assert "accepted" in payload
    assert "reason" in payload
