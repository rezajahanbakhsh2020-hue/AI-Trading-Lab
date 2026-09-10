from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.backtest.engine import run_backtest
from src.evaluation.metrics import (
    calmar_ratio,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    win_rate,
)
from src.features.indicators import add_returns
from src.strategies.registry import (
    DEFAULT_REGISTRY,
    StrategyRegistry,
)


@dataclass(frozen=True)
class StrategyEvaluation:
    name: str
    category: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    exposure: float
    observations: int
    ranking_score: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "exposure": self.exposure,
            "observations": self.observations,
            "ranking_score": self.ranking_score,
        }


def _safe(value: float) -> float:
    value = float(value)

    if pd.isna(value):
        return 0.0

    return value


def _score_evaluation(
    *,
    total: float,
    drawdown: float,
    sharpe: float,
    calmar: float,
    profit_factor_value: float,
) -> float:
    """
    Compact multi-objective ranking score.

    The score favors:
    - positive return
    - stronger risk-adjusted return
    - smaller drawdown
    - positive profit factor

    Values are bounded so an extreme metric cannot dominate
    the complete ranking.
    """

    return_score = max(-1.0, min(1.0, total))

    sharpe_score = max(-1.0, min(1.0, sharpe / 2.0))

    calmar_score = max(-1.0, min(1.0, calmar / 3.0))

    drawdown_score = max(-1.0, min(1.0, 1.0 + drawdown))

    if profit_factor_value <= 0.0:
        profit_score = -1.0
    else:
        profit_score = max(
            -1.0,
            min(
                1.0,
                (profit_factor_value - 1.0)
                / (profit_factor_value + 1.0),
            ),
        )

    score = (
        0.30 * return_score
        + 0.25 * sharpe_score
        + 0.20 * calmar_score
        + 0.15 * drawdown_score
        + 0.10 * profit_score
    )

    return round(float(score), 6)


def evaluate_strategy(
    df: pd.DataFrame,
    name: str,
    *,
    registry: StrategyRegistry = DEFAULT_REGISTRY,
    strategy_kwargs: Mapping | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> StrategyEvaluation:
    """
    Evaluate one registered strategy on the supplied market data.

    The strategy is responsible only for generating its signal.
    The existing backtest engine remains responsible for execution
    and look-ahead-safe position handling.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df must not be empty.")

    if not isinstance(strategy_kwargs, Mapping):
        if strategy_kwargs is not None:
            raise TypeError(
                "strategy_kwargs must be a mapping or None."
            )

    strategy = registry.get(name)

    data = df.copy()

    if "return" not in data.columns:
        data = add_returns(data)

    kwargs = dict(strategy_kwargs or {})

    strategy_result = registry.run(
        strategy.name,
        data,
        **kwargs,
    )

    if "signal" not in strategy_result.columns:
        raise ValueError(
            f"Strategy '{strategy.name}' did not produce "
            "a 'signal' column."
        )

    backtest_result = run_backtest(
        strategy_result,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )

    total = _safe(total_return(backtest_result))
    drawdown = _safe(max_drawdown(backtest_result))
    sharpe = _safe(sharpe_ratio(backtest_result))
    calmar = _safe(calmar_ratio(backtest_result))
    wins = _safe(win_rate(backtest_result))
    factor = _safe(profit_factor(backtest_result))
    market_exposure = _safe(
        exposure(strategy_result, signal_column="signal")
    )

    score = _score_evaluation(
        total=total,
        drawdown=drawdown,
        sharpe=sharpe,
        calmar=calmar,
        profit_factor_value=factor,
    )

    return StrategyEvaluation(
        name=strategy.name,
        category=strategy.category,
        total_return=total,
        max_drawdown=drawdown,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        win_rate=wins,
        profit_factor=factor,
        exposure=market_exposure,
        observations=len(strategy_result),
        ranking_score=score,
    )


def evaluate_all_strategies(
    df: pd.DataFrame,
    *,
    registry: StrategyRegistry = DEFAULT_REGISTRY,
    strategy_kwargs: Mapping[str, Mapping] | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """
    Evaluate every strategy in the registry.

    Results are returned as a ranking table. The evaluator does not
    select a production strategy; later Walk-Forward and Stability
    layers remain the final validation gates.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df must not be empty.")

    if strategy_kwargs is not None:
        if not isinstance(strategy_kwargs, Mapping):
            raise TypeError(
                "strategy_kwargs must be a mapping or None."
            )

    rows: list[dict] = []

    kwargs_by_strategy = strategy_kwargs or {}

    for strategy in registry.all():
        kwargs = kwargs_by_strategy.get(
            strategy.name,
            {},
        )

        evaluation = evaluate_strategy(
            df,
            strategy.name,
            registry=registry,
            strategy_kwargs=kwargs,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        rows.append(evaluation.as_dict())

    columns = [
        "name",
        "category",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "exposure",
        "observations",
        "ranking_score",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "ranking_score",
                "sharpe_ratio",
                "total_return",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def select_top_strategies(
    evaluations: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Select the top N candidates from an evaluation table.

    This is only candidate selection. It must not be interpreted as
    final production approval.
    """

    if not isinstance(evaluations, pd.DataFrame):
        raise TypeError(
            "evaluations must be a pandas DataFrame."
        )

    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    if "ranking_score" not in evaluations.columns:
        raise ValueError(
            "evaluations must contain 'ranking_score'."
        )

    return evaluations.head(top_n).reset_index(drop=True)
