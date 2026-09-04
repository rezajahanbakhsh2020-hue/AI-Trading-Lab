import pandas as pd

def run_backtest(
df: pd.DataFrame,
signal_column: str = "signal",
return_column: str = "return",
transaction_cost: float = 0.0,
slippage: float = 0.0,
) -> pd.DataFrame:
"""
Run a simple long-only backtest.

The signal generated at period t is executed during period t+1
to avoid look-ahead bias.

Transaction costs and slippage are charged when the signal
changes and that change becomes effective on the following
period.

The first strategy return is NaN because there is no previous
signal available for the first period.

Parameters
----------
df:
    Input DataFrame containing signal and return columns.

signal_column:
    Name of the trading signal column.

return_column:
    Name of the market return column.

transaction_cost:
    Proportional transaction cost applied to signal changes.

slippage:
    Proportional slippage applied to signal changes.

Returns
-------
pd.DataFrame
    DataFrame containing strategy returns, position,
    turnover, trading costs, gross returns, and equity.
"""

result = df.copy()

if signal_column not in result.columns:
    raise ValueError(f"Column '{signal_column}' not found.")

if return_column not in result.columns:
    raise ValueError(f"Column '{return_column}' not found.")

if transaction_cost < 0:
    raise ValueError("transaction_cost must be non-negative.")

if slippage < 0:
    raise ValueError("slippage must be non-negative.")

# Execute the signal on the following period.
# The first period therefore has no active position.
result["position"] = result[signal_column].shift(1).fillna(0.0)

# A signal change at period t becomes a position change
# at period t+1. Therefore turnover is the absolute signal
# change shifted by one period.
result["turnover"] = (
    result[signal_column]
    .diff()
    .abs()
    .shift(1)
    .fillna(0.0)
)

# Gross return before transaction costs and slippage.
# Keep the first return as NaN because there is no previous
# signal available at the first period.
result["gross_strategy_return"] = (
    result["position"] * result[return_column]
)

result.loc[result.index[0], "gross_strategy_return"] = (
    float("nan") if not result.empty else None
)

total_cost_rate = transaction_cost + slippage

result["trading_cost"] = (
    result["turnover"] * total_cost_rate
)

# Net strategy return after transaction costs and slippage.
result["strategy_return"] = (
    result["gross_strategy_return"]
    - result["trading_cost"]
)

# The first period has no strategy return, but equity starts
# from the initial capital of 1.0.
result["equity"] = (
    1.0 + result["strategy_return"].fillna(0.0)
).cumprod()

return result
