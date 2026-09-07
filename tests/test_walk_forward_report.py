import pandas as pd
import pytest

from src.evaluation.walk_forward_report import (
    combine_oos_results,
    evaluate_walk_forward,
)


def create_oos_result(
    returns: list[float],
    start: str = "2026-01-01",
    signal: int = 1,
) -> pd.DataFrame:
    strategy_returns = pd.Series(returns, dtype=float)

    equity = (1.0 + strategy_returns).cumprod()

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=start,
                periods=len(returns),
                freq="h",
            ),
            "strategy_return": strategy_returns,
            "equity": equity,
            "signal": signal,
        }
    )


def test_combine_oos_results_returns_empty_dataframe_for_empty_input():
    result = combine_oos_results([])

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_combine_oos_results_rejects_non_list():
    with pytest.raises(TypeError):
        combine_oos_results(None)  # type: ignore[arg-type]


def test_combine_oos_results_rejects_non_dataframe():
    with pytest.raises(TypeError):
        combine_oos_results([None])  # type: ignore[list-item]


def test_combine_oos_results_requires_required_columns():
    invalid = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01",
                periods=2,
                freq="h",
            ),
            "strategy_return": [0.01, 0.02],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        combine_oos_results([invalid])


def test_combine_oos_results_is_chronological():
    first = create_oos_result(
        [0.01, 0.02],
        start="2026-01-03",
    )
    second = create_oos_result(
        [0.03, 0.04],
        start="2026-01-01",
    )

    combined = combine_oos_results([first, second])

    assert combined["timestamp"].is_monotonic_increasing
    assert len(combined) == 4


def test_evaluate_walk_forward_empty():
    report = evaluate_walk_forward([])

    assert report["windows"] == 0
    assert report["observations"] == 0
    assert report["total_return"] == 0.0
    assert report["max_drawdown"] == 0.0
    assert report["sharpe_ratio"] == 0.0
    assert report["calmar_ratio"] == 0.0
    assert report["sortino_ratio"] == 0.0
    assert report["exposure"] == 0.0
    assert report["win_rate"] == 0.0
    assert report["profit_factor"] == 0.0
    assert report["window_returns"] == []
    assert report["profitable_windows"] == 0
    assert report["losing_windows"] == 0
    assert report["positive_window_rate"] == 0.0


def test_evaluate_walk_forward_reports_window_statistics():
    first = create_oos_result(
        [0.01, 0.02, -0.01],
        start="2026-01-01",
    )
    second = create_oos_result(
        [0.03, -0.01, 0.02],
        start="2026-01-02",
    )

    report = evaluate_walk_forward(
        [first, second]
    )

    assert report["windows"] == 2
    assert report["observations"] == 6

    assert len(report["window_returns"]) == 2
    assert report["profitable_windows"] == 2
    assert report["losing_windows"] == 0
    assert report["positive_window_rate"] == 1.0


def test_evaluate_walk_forward_supports_position_column():
    result = create_oos_result(
        [0.01, -0.005, 0.02],
    )

    result["position"] = [1, 0, 1]
    result = result.drop(columns=["signal"])

    report = evaluate_walk_forward([result])

    assert report["windows"] == 1
    assert report["observations"] == 3
    assert 0.0 <= report["exposure"] <= 1.0


def test_evaluate_walk_forward_requires_exposure_column():
    result = create_oos_result(
        [0.01, -0.005, 0.02],
    )

    result = result.drop(columns=["signal"])

    with pytest.raises(
        ValueError,
        match="Missing required exposure column",
    ):
        evaluate_walk_forward([result])


def test_evaluate_walk_forward_calculates_positive_window_rate():
    winning = create_oos_result(
        [0.02, 0.01],
        start="2026-01-01",
    )

    losing = create_oos_result(
        [-0.02, -0.01],
        start="2026-01-02",
    )

    report = evaluate_walk_forward(
        [winning, losing]
    )

    assert report["profitable_windows"] == 1
    assert report["losing_windows"] == 1
    assert report["positive_window_rate"] == 0.5


def test_evaluate_walk_forward_preserves_window_order():
    first = create_oos_result(
        [0.01, 0.01],
        start="2026-01-01",
    )

    second = create_oos_result(
        [-0.01, -0.01],
        start="2026-01-03",
    )

    report = evaluate_walk_forward(
        [first, second]
    )

    assert report["window_returns"][0] > 0
    assert report["window_returns"][1] < 0
