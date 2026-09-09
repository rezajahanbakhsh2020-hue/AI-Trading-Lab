"""Transaction-cost sensitivity analysis for strategy evaluation.

This module evaluates how a strategy's performance changes across a range
of transaction-cost assumptions without duplicating the backtest engine.

The caller provides a backtest runner callable. The runner receives a
transaction_cost value and must return a mapping containing performance
metrics such as total_return, sharpe_ratio, and max_drawdown.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from math import isfinite
from typing import Any

import pandas as pd


BacktestRunner = Callable[[float], Mapping[str, Any]]


def _validate_costs(transaction_costs: Iterable[float]) -> list[float]:
    """Validate and normalize transaction-cost scenarios."""
    costs = list(transaction_costs)

    if not costs:
        raise ValueError("transaction_costs must not be empty")

    normalized: list[float] = []

    for cost in costs:
        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            raise TypeError("transaction costs must be numeric")

        value = float(cost)

        if not isfinite(value):
            raise ValueError("transaction costs must be finite")

        if value < 0.0:
            raise ValueError("transaction costs must be non-negative")

        normalized.append(value)

    return normalized


def run_transaction_cost_sensitivity(
    backtest_runner: BacktestRunner,
    transaction_costs: Iterable[float],
) -> pd.DataFrame:
    """Run a strategy across multiple transaction-cost assumptions.

    Parameters
    ----------
    backtest_runner:
        Callable accepting one transaction-cost value and returning a
        mapping of performance metrics.

    transaction_costs:
        Iterable of non-negative transaction-cost assumptions.

    Returns
    -------
    pandas.DataFrame
        One row per transaction-cost scenario.

    Raises
    ------
    TypeError
        If the runner is not callable or a returned result is not a mapping.

    ValueError
        If transaction-cost scenarios are invalid or the runner returns
        no result.
    """
    if not callable(backtest_runner):
        raise TypeError("backtest_runner must be callable")

    costs = _validate_costs(transaction_costs)

    rows: list[dict[str, Any]] = []

    for cost in costs:
        result = backtest_runner(cost)

        if not isinstance(result, Mapping):
            raise TypeError(
                "backtest_runner must return a mapping of performance metrics"
            )

        row: dict[str, Any] = {"transaction_cost": cost}

        for key, value in result.items():
            row[str(key)] = value

        rows.append(row)

    return pd.DataFrame(rows)


def assess_transaction_cost_robustness(
    sensitivity_results: pd.DataFrame,
    *,
    return_column: str = "total_return",
    sharpe_column: str = "sharpe_ratio",
    minimum_positive_return_rate: float = 1.0,
) -> dict[str, Any]:
    """Assess whether performance remains acceptable under cost stress.

    A strategy is considered robust when:

    1. all tested scenarios have finite total returns;
    2. the proportion of scenarios with positive total return is at least
       ``minimum_positive_return_rate``;
    3. if Sharpe data is available, all Sharpe values are finite.

    The function does not claim profitability or guarantee future performance.
    It only summarizes behavior under the supplied historical scenarios.
    """
    if not isinstance(sensitivity_results, pd.DataFrame):
        raise TypeError("sensitivity_results must be a pandas DataFrame")

    if sensitivity_results.empty:
        raise ValueError("sensitivity_results must not be empty")

    if return_column not in sensitivity_results.columns:
        raise ValueError(
            f"missing required return column: {return_column}"
        )

    if not 0.0 <= minimum_positive_return_rate <= 1.0:
        raise ValueError(
            "minimum_positive_return_rate must be between 0 and 1"
        )

    returns = pd.to_numeric(
        sensitivity_results[return_column],
        errors="coerce",
    )

    if returns.isna().any():
        raise ValueError("total returns must contain only numeric finite values")

    if not returns.map(isfinite).all():
        raise ValueError("total returns must contain only finite values")

    positive_rate = float((returns > 0.0).mean())
    worst_return = float(returns.min())
    best_return = float(returns.max())
    average_return = float(returns.mean())

    result: dict[str, Any] = {
        "scenarios": int(len(returns)),
        "positive_return_rate": positive_rate,
        "worst_total_return": worst_return,
        "best_total_return": best_return,
        "average_total_return": average_return,
        "robust": positive_rate >= minimum_positive_return_rate,
    }

    if sharpe_column in sensitivity_results.columns:
        sharpe = pd.to_numeric(
            sensitivity_results[sharpe_column],
            errors="coerce",
        )

        if sharpe.isna().any() or not sharpe.map(isfinite).all():
            raise ValueError("Sharpe values must be finite numeric values")

        result["worst_sharpe_ratio"] = float(sharpe.min())
        result["best_sharpe_ratio"] = float(sharpe.max())

    return result


def build_transaction_cost_sensitivity_report(
    backtest_runner: BacktestRunner,
    transaction_costs: Iterable[float],
    *,
    minimum_positive_return_rate: float = 1.0,
) -> dict[str, Any]:
    """Run sensitivity analysis and return results plus robustness assessment."""
    results = run_transaction_cost_sensitivity(
        backtest_runner,
        transaction_costs,
    )

    assessment = assess_transaction_cost_robustness(
        results,
        minimum_positive_return_rate=minimum_positive_return_rate,
    )

    return {
        "results": results,
        "assessment": assessment,
    }
