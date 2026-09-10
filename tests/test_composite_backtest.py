import numpy as np
import pandas as pd
import pytest

from src.evaluation.composite_backtest import (
    CompositeBacktestResult,
    compare_composite_to_components,
    run_composite_backtest,
)
from src.evaluation.strategy_evaluator import (
    evaluate_all_strategies,
)
from src.strategies.registry import (
    DEFAULT_REGISTRY,
)


def make_data(size: int = 120) -> pd.DataFrame:
    close = pd.Series(
        np.linspace(100.0, 160.0, size)
        + np.sin(np.arange(size) / 4.0),
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

    return data, outputs, evaluations


def test_composite_runs_actual_backtest():
    data, outputs, evaluations = build_inputs()

    result = run_composite_backtest(
        data,
        outputs,
        evaluations,
        top_n=3,
    )

    assert isinstance(result, CompositeBacktestResult)
    assert result.name == "ensemble"
    assert len(result.selected_strategies) == 3
    assert result.observations == len(data)
    assert len(result.signal) == len(data)
    assert len(result.ensemble_score) == len(data)


def test_composite_uses_same_metrics_pipeline():
    data, outputs, evaluations = build_inputs()

    result = run_composite_backtest(
        data,
        outputs,
        evaluations,
        top_n=2,
    )

    evaluation = result.evaluation

    assert evaluation.name == "ensemble"
    assert evaluation.category == "ensemble"
    assert np.isfinite(evaluation.total_return)
    assert np.isfinite(evaluation.max_drawdown)
    assert np.isfinite(evaluation.sharpe_ratio)
    assert np.isfinite(evaluation.calmar_ratio)
    assert np.isfinite(evaluation.win_rate)
    assert np.isfinite(evaluation.profit_factor)
    assert np.isfinite(evaluation.exposure)
    assert np.isfinite(evaluation.ranking_score)


def test_composite_signal_is_binary_long_only():
    data, outputs, evaluations = build_inputs()

    result = run_composite_backtest(
        data,
        outputs,
        evaluations,
        top_n=3,
    )

    assert set(result.signal.dropna().unique()).issubset({0, 1})
    assert result.signal.dtype.kind in {"i", "u"}


def test_composite_does_not_claim_superiority_automatically():
    data, outputs, evaluations = build_inputs()

    result = run_composite_backtest(
        data,
        outputs,
        evaluations,
        top_n=3,
    )

    comparison = compare_composite_to_components(
        result,
        evaluations,
    )

    assert len(comparison) == 4
    assert "composite" in set(comparison["kind"])
    assert set(comparison["kind"]) == {"component", "composite"}


def test_comparison_contains_actual_composite_metrics():
    data, outputs, evaluations = build_inputs()

    result = run_composite_backtest(
        data,
        outputs,
        evaluations,
        top_n=2,
    )

    comparison = compare_composite_to_components(
        result,
        evaluations,
    )

    composite_row = comparison[
        comparison["kind"] == "composite"
    ].iloc[0]

    assert composite_row["name"] == "ensemble"
    assert composite_row["total_return"] == pytest.approx(
        result.total_return
    )
    assert composite_row["sharpe_ratio"] == pytest.approx(
        result.sharpe_ratio
    )


def test_invalid_top_n_is_rejected():
    data, outputs, evaluations = build_inputs()

    with pytest.raises(ValueError):
        run_composite_backtest(
            data,
            outputs,
            evaluations,
            top_n=0,
        )


def test_missing_strategy_output_is_rejected():
    data, outputs, evaluations = build_inputs()

    missing = dict(outputs)
    missing.pop(evaluations.iloc[0]["name"])

    with pytest.raises(ValueError, match="Missing strategy outputs"):
        run_composite_backtest(
            data,
            missing,
            evaluations,
            top_n=3,
        )


def test_empty_data_is_rejected():
    _, outputs, evaluations = build_inputs()

    with pytest.raises(ValueError):
        run_composite_backtest(
            pd.DataFrame(),
            outputs,
            evaluations,
        )


def test_invalid_composite_name_is_rejected():
    data, outputs, evaluations = build_inputs()

    with pytest.raises(ValueError):
        run_composite_backtest(
            data,
            outputs,
            evaluations,
            name="",
        )

    with pytest.raises(ValueError):
        run_composite_backtest(
            data,
            outputs,
            evaluations,
            name="my ensemble",
        )
