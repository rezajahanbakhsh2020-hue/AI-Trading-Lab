import pandas as pd
import pytest

from src.features.indicators import add_returns
from src.strategy.baseline import generate_baseline_signal
from src.backtest.engine import run_backtest
from src.evaluation.report import evaluate_backtest


def test_backtest_does_not_use_current_signal_return():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(df)

    expected = [float("nan"), -0.05, 0.0, -0.10]

    assert result["strategy_return"].iloc[0] != result["strategy_return"].iloc[0]
    assert result["strategy_return"].iloc[1:].tolist() == pytest.approx(
        expected[1:]
    )


def test_equity_curve_is_based_on_strategy_returns():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
        "signal": [1, 0, 1, 0],
    })

    result = run_backtest(df)

    expected_equity = [
        1.0,
        0.95,
        0.95,
        0.855,
    ]

    assert result["equity"].tolist() == pytest.approx(expected_equity)


def test_baseline_strategy_has_no_future_return_access():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10],
    })

    result = generate_baseline_signal(df)

    assert result["signal"].tolist() == [0, 1, 0, 1]


def test_complete_backtest_pipeline_has_expected_columns():
    df = pd.DataFrame({
        "close": [100.0, 101.0, 99.0, 102.0, 101.0],
    })

    df = add_returns(df)
    df = generate_baseline_signal(df)
    df = run_backtest(df)

    required_columns = {
        "close",
        "return",
        "signal",
        "strategy_return",
        "equity",
    }

    assert required_columns.issubset(df.columns)


def test_complete_backtest_pipeline_produces_valid_equity():
    df = pd.DataFrame({
        "close": [100.0, 101.0, 99.0, 102.0, 101.0],
    })

    df = add_returns(df)
    df = generate_baseline_signal(df)
    df = run_backtest(df)

    assert df["equity"].iloc[0] == pytest.approx(1.0)
    assert df["equity"].notna().all()
    assert (df["equity"] > 0).all()


def test_evaluation_uses_actual_backtest_output():
    df = pd.DataFrame({
        "close": [100.0, 101.0, 99.0, 102.0, 101.0],
    })

    df = add_returns(df)
    df = generate_baseline_signal(df)
    df = run_backtest(df)

    report = evaluate_backtest(df)

    expected_metrics = {
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "sortino_ratio",
        "exposure",
        "win_rate",
        "profit_factor",
    }

    assert set(report.keys()) == expected_metrics


def test_input_dataframe_is_not_modified_by_backtest():
    df = pd.DataFrame({
        "return": [0.01, -0.02, 0.03],
        "signal": [0, 1, 1],
    })

    original = df.copy()

    run_backtest(df)

    pd.testing.assert_frame_equal(df, original)


def test_strategy_and_backtest_work_together_without_lookahead():
    df = pd.DataFrame({
        "return": [0.10, -0.05, 0.20, -0.10, 0.05],
    })

    strategy_result = generate_baseline_signal(df)
    backtest_result = run_backtest(strategy_result)

    # First period has no previous signal, so strategy_return is NaN.
    assert pd.isna(backtest_result["strategy_return"].iloc[0])

    # The second period has an inactive previous signal.
    assert backtest_result["strategy_return"].iloc[1] == pytest.approx(0.0)

    # The third period executes the signal generated from
    # the first period's return.
    assert backtest_result["strategy_return"].iloc[2] == pytest.approx(0.20)

    # The fourth period has an inactive previous signal.
    assert backtest_result["strategy_return"].iloc[3] == pytest.approx(0.0)

    # The fifth period executes the active previous signal.
    assert backtest_result["strategy_return"].iloc[4] == pytest.approx(0.05)
