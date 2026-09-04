import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.features.indicators import add_returns
from src.strategies.baseline import baseline_signal

def test_baseline_signal():
df = pd.DataFrame(
{
"close": [100, 101, 102, 103, 104, 105],
}
)

result = baseline_signal(
    df,
    fast_window=2,
    slow_window=3,
)

assert "fast_ma" in result.columns
assert "slow_ma" in result.columns
assert "signal" in result.columns

assert result["signal"].iloc[-1] == 1

def test_run_backtest():
df = pd.DataFrame(
{
"signal": [0, 1, 1],
"return": [0.00, 0.10, -0.05],
}
)

result = run_backtest(df)

assert "position" in result.columns
assert "turnover" in result.columns
assert "gross_strategy_return" in result.columns
assert "trading_cost" in result.columns
assert "strategy_return" in result.columns
assert "equity" in result.columns

assert result["position"].tolist() == [0.0, 0.0, 1.0]

assert result["strategy_return"].iloc[1] == 0.0
assert result["strategy_return"].iloc[2] == -0.05

def test_run_backtest_executes_signal_on_next_period():
df = pd.DataFrame(
{
"signal": [1, 0, 1],
"return": [0.10, 0.20, -0.10],
}
)

result = run_backtest(df)

assert result["position"].tolist() == [0.0, 1.0, 0.0]

assert result["strategy_return"].iloc[0] == 0.0
assert result["strategy_return"].iloc[1] == 0.20
assert result["strategy_return"].iloc[2] == 0.0

def test_run_backtest_calculates_turnover():
df = pd.DataFrame(
{
"signal": [0, 1, 1, 0, 1],
"return": [0.0, 0.1, 0.2, -0.1, 0.05],
}
)

result = run_backtest(df)

assert result["turnover"].tolist() == [
    0.0,
    0.0,
    0.0,
    1.0,
    1.0,
]

def test_run_backtest_applies_transaction_cost():
df = pd.DataFrame(
{
"signal": [0, 1, 1],
"return": [0.0, 0.10, 0.20],
}
)

result = run_backtest(
    df,
    transaction_cost=0.01,
)

assert result["trading_cost"].iloc[1] == 0.0
assert result["trading_cost"].iloc[2] == 0.01

assert result["strategy_return"].iloc[2] == pytest.approx(
    0.19
)

def test_run_backtest_applies_slippage():
df = pd.DataFrame(
{
"signal": [0, 1, 1],
"return": [0.0, 0.10, 0.20],
}
)

result = run_backtest(
    df,
    slippage=0.005,
)

assert result["trading_cost"].iloc[1] == 0.0
assert result["trading_cost"].iloc[2] == 0.005

assert result["strategy_return"].iloc[2] == pytest.approx(
    0.195
)

def test_run_backtest_combines_transaction_cost_and_slippage():
df = pd.DataFrame(
{
"signal": [0, 1, 1],
"return": [0.0, 0.10, 0.20],
}
)

result = run_backtest(
    df,
    transaction_cost=0.01,
    slippage=0.005,
)

assert result["trading_cost"].iloc[2] == pytest.approx(
    0.015
)

assert result["strategy_return"].iloc[2] == pytest.approx(
    0.185
)

def test_run_backtest_does_not_modify_input():
df = pd.DataFrame(
{
"signal": [0, 1, 1],
"return": [0.0, 0.10, -0.05],
}
)

original = df.copy()

run_backtest(df)

pd.testing.assert_frame_equal(
    df,
    original,
)

def test_run_backtest_rejects_missing_signal_column():
df = pd.DataFrame(
{
"return": [0.10, -0.05],
}
)

with pytest.raises(ValueError):
    run_backtest(df)

def test_run_backtest_rejects_missing_return_column():
df = pd.DataFrame(
{
"signal": [0, 1],
}
)

with pytest.raises(ValueError):
    run_backtest(df)

def test_run_backtest_rejects_negative_transaction_cost():
df = pd.DataFrame(
{
"signal": [0, 1],
"return": [0.0, 0.10],
}
)

with pytest.raises(ValueError):
    run_backtest(
        df,
        transaction_cost=-0.01,
    )

def test_run_backtest_rejects_negative_slippage():
df = pd.DataFrame(
{
"signal": [0, 1],
"return": [0.0, 0.10],
}
)

with pytest.raises(ValueError):
    run_backtest(
        df,
        slippage=-0.01,
    )

def test_run_backtest_equity_is_calculated():
df = pd.DataFrame(
{
"signal": [0, 1, 1],
"return": [0.0, 0.10, 0.20],
}
)

result = run_backtest(df)

assert result["equity"].iloc[0] == pytest.approx(1.0)
assert result["equity"].iloc[1] == pytest.approx(1.0)
assert result["equity"].iloc[2] == pytest.approx(1.20)

def test_full_backtest_pipeline():
df = pd.DataFrame(
{
"close": [100.0, 102.0, 104.0, 106.0, 108.0],
}
)

df = add_returns(df)

df = baseline_signal(
    df,
    fast_window=2,
    slow_window=3,
)

result = run_backtest(df)

assert "return" in result.columns
assert "signal" in result.columns
assert "position" in result.columns
assert "turnover" in result.columns
assert "gross_strategy_return" in result.columns
assert "trading_cost" in result.columns
assert "strategy_return" in result.columns
assert "equity" in result.columns

assert result["equity"].iloc[-1] > 0
