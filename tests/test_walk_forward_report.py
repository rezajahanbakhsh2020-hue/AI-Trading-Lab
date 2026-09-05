import pandas as pd
import pytest

from src.evaluation.walk_forward_report import (
    combine_oos_results,
    evaluate_walk_forward,
)


def create_oos_result(
    start_date: str,
    size: int = 3,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        start_date,
        periods=size,
        freq="D",
    )

    strategy_returns = pd.Series(
        [0.01, 0.02, -0.01][:size],
        dtype=float,
    )

    equity = (
        1.0 + strategy_returns
    ).cumprod()

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy_return": strategy_returns,
            "equity": equity,
            "position": [1.0] * size,
        }
    )


def test_combine_oos_results_combines_multiple_windows():
    first = create_oos_result(
        "2026-01-01",
        size=3,
    )

    second = create_oos_result(
        "2026-01-04",
        size=3,
    )

    combined = combine_oos_results(
        [first, second]
    )

    assert len(combined) == 6
    assert list(combined["timestamp"]) == list(
        pd.date_range(
            "2026-01-01",
            periods=6,
            freq="D",
        )
    )


def test_combine_oos_results_sorts_by_timestamp():
    first = create_oos_result(
        "2026-01-04",
        size=3,
    )

    second = create_oos_result(
        "2026-01-01",
        size=3,
    )

    combined = combine_oos_results(
        [first, second]
    )

    assert combined["timestamp"].is_monotonic_increasing


def test_combine_oos_results_empty_list():
    combined = combine_oos_results([])

    assert combined.empty


def test_combine_oos_results_requires_list():
    with pytest.raises(TypeError):
        combine_oos_results(None)


def test_combine_oos_results_requires_dataframes():
    with pytest.raises(TypeError):
        combine_oos_results(
            [
                pd.DataFrame(),
                "invalid",
            ]
        )


def test_evaluate_walk_forward_returns_metrics():
    first = create_oos_result(
        "2026-01-01",
        size=3,
    )

    second = create_oos_result(
        "2026-01-04",
        size=3,
    )

    report = evaluate_walk_forward(
        [first, second]
    )

    assert report["windows"] == 2
    assert report["observations"] == 6

    assert "total_return" in report
    assert "max_drawdown" in report
    assert "sharpe_ratio" in report
    assert "calmar_ratio" in report
    assert "sortino_ratio" in report
    assert "exposure" in report
    assert "win_rate" in report
    assert "profit_factor" in report


def test_evaluate_walk_forward_empty_results():
    report = evaluate_walk_forward([])

    assert report["windows"] == 0
    assert report["observations"] == 0
    assert report["total_return"] == 0.0
    assert report["max_drawdown"] == 0.0


def test_evaluate_walk_forward_requires_required_columns():
    invalid = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=3,
                freq="D",
            ),
        }
    )

    with pytest.raises(ValueError):
        evaluate_walk_forward([invalid])
