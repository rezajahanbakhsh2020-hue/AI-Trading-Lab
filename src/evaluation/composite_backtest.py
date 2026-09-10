from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.evaluation.metrics import (
    calmar_ratio,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    total_return,
    win_rate,
)
from src.evaluation.strategy_evaluator import (
    StrategyEvaluation,
    evaluate_strategy,
)
from src.strategies.ensemble import build_ensemble_from_evaluations
from src.strategies.registry import StrategyRegistry, StrategySpec


@dataclass(frozen=True)
class CompositeBacktestResult:
    """
    Actual backtest result for a ranking-selected composite strategy.

    The composite is evaluated through the same strategy evaluator and
    backtest engine used by individual strategies. This makes the
    composite directly comparable with its components.
    """

    name: str
    selected_strategies: tuple[str, ...]
    evaluation: StrategyEvaluation
    signal: pd.Series
    ensemble_score: pd.Series

    @property
    def total_return(self) -> float:
        return self.evaluation.total_return

    @property
    def max_drawdown(self) -> float:
        return self.evaluation.max_drawdown

    @property
    def sharpe_ratio(self) -> float:
        return self.evaluation.sharpe_ratio

    @property
    def calmar_ratio(self) -> float:
        return self.evaluation.calmar_ratio

    @property
    def win_rate(self) -> float:
        return self.evaluation.win_rate

    @property
    def profit_factor(self) -> float:
        return self.evaluation.profit_factor

    @property
    def exposure(self) -> float:
        return self.evaluation.exposure

    @property
    def observations(self) -> int:
        return self.evaluation.observations

    @property
    def ranking_score(self) -> float:
        return self.evaluation.ranking_score

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "selected_strategies": list(self.selected_strategies),
            **self.evaluation.as_dict(),
        }


def _validate_inputs(
    df: pd.DataFrame,
    strategy_outputs: Mapping[str, pd.DataFrame],
    evaluations: pd.DataFrame,
    top_n: int,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df must not be empty.")

    if not isinstance(strategy_outputs, Mapping):
        raise TypeError("strategy_outputs must be a mapping.")

    if not strategy_outputs:
        raise ValueError("strategy_outputs must not be empty.")

    if not isinstance(evaluations, pd.DataFrame):
        raise TypeError("evaluations must be a pandas DataFrame.")

    if evaluations.empty:
        raise ValueError("evaluations must not be empty.")

    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")

    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    required = {"name", "ranking_score"}
    missing = required.difference(evaluations.columns)

    if missing:
        raise ValueError(
            "evaluations is missing columns: "
            + ", ".join(sorted(missing))
        )


def _selected_names(
    evaluations: pd.DataFrame,
    strategy_outputs: Mapping[str, pd.DataFrame],
    top_n: int,
) -> tuple[str, ...]:
    selected = evaluations.head(top_n)

    names = tuple(
        str(name).strip().lower()
        for name in selected["name"]
        if str(name).strip()
    )

    if not names:
        raise ValueError("no strategies were selected.")

    missing = [
        name
        for name in names
        if name not in strategy_outputs
    ]

    if missing:
        raise ValueError(
            "Missing strategy outputs: "
            + ", ".join(missing)
        )

    if len(set(names)) != len(names):
        raise ValueError("selected strategy names must be unique.")

    return names


def _build_composite_registry(
    composite_name: str,
    composite: pd.DataFrame,
) -> StrategyRegistry:
    def composite_signal(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if len(data) != len(composite):
            raise ValueError(
                "composite signal length does not match market data."
            )

        if not data.index.equals(composite.index):
            raise ValueError(
                "composite signal index does not match market data."
            )

        result = data.copy()
        result["signal"] = composite["signal"].to_numpy()
        return result

    return StrategyRegistry(
        (
            StrategySpec(
                name=composite_name,
                category="ensemble",
                function=composite_signal,
                description="Ranking-weighted composite strategy.",
                supports_long=True,
                supports_short=False,
            ),
        )
    )


def run_composite_backtest(
    df: pd.DataFrame,
    strategy_outputs: Mapping[str, pd.DataFrame],
    evaluations: pd.DataFrame,
    *,
    top_n: int = 3,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    name: str = "ensemble",
) -> CompositeBacktestResult:
    """
    Build and actually backtest a Top-N ranking-weighted composite.

    The composite is not considered superior merely because it combines
    high-ranked strategies. Its own backtest metrics are calculated
    independently through evaluate_strategy().
    """

    _validate_inputs(
        df,
        strategy_outputs,
        evaluations,
        top_n,
    )

    composite_name = str(name).strip().lower()

    if not composite_name:
        raise ValueError("name must not be empty.")

    if " " in composite_name:
        raise ValueError("name must not contain spaces.")

    names = _selected_names(
        evaluations,
        strategy_outputs,
        top_n,
    )

    selected_evaluations = evaluations[
        evaluations["name"].astype(str).str.strip().str.lower().isin(names)
    ].copy()

    selected_evaluations["_selection_order"] = pd.Categorical(
        selected_evaluations["name"].astype(str).str.strip().str.lower(),
        categories=list(names),
        ordered=True,
    )

    selected_evaluations = (
        selected_evaluations
        .sort_values("_selection_order")
        .drop(columns="_selection_order")
        .reset_index(drop=True)
    )

    selected_outputs = {
        strategy_name: strategy_outputs[strategy_name]
        for strategy_name in names
    }

    composite = build_ensemble_from_evaluations(
        selected_outputs,
        selected_evaluations,
        top_n=len(names),
    )

    registry = _build_composite_registry(
        composite_name,
        composite,
    )

    evaluation = evaluate_strategy(
        df,
        composite_name,
        registry=registry,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )

    return CompositeBacktestResult(
        name=composite_name,
        selected_strategies=names,
        evaluation=evaluation,
        signal=composite["signal"].copy(),
        ensemble_score=composite["ensemble_score"].copy(),
    )


def compare_composite_to_components(
    composite: CompositeBacktestResult,
    evaluations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare actual composite metrics with the already calculated
    component metrics.

    No superiority decision is made here. The result is evidence for
    the later composite validation, Walk-Forward, and stability gates.
    """

    if not isinstance(composite, CompositeBacktestResult):
        raise TypeError(
            "composite must be a CompositeBacktestResult."
        )

    if not isinstance(evaluations, pd.DataFrame):
        raise TypeError("evaluations must be a pandas DataFrame.")

    required = {
        "name",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "exposure",
        "ranking_score",
    }

    missing = required.difference(evaluations.columns)

    if missing:
        raise ValueError(
            "evaluations is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = []

    for _, row in evaluations.iterrows():
        strategy_name = str(row["name"]).strip().lower()

        if strategy_name not in composite.selected_strategies:
            continue

        rows.append(
            {
                "name": strategy_name,
                "kind": "component",
                "total_return": float(row["total_return"]),
                "max_drawdown": float(row["max_drawdown"]),
                "sharpe_ratio": float(row["sharpe_ratio"]),
                "calmar_ratio": float(row["calmar_ratio"]),
                "win_rate": float(row["win_rate"]),
                "profit_factor": float(row["profit_factor"]),
                "exposure": float(row["exposure"]),
                "ranking_score": float(row["ranking_score"]),
            }
        )

    rows.append(
        {
            "name": composite.name,
            "kind": "composite",
            "total_return": composite.total_return,
            "max_drawdown": composite.max_drawdown,
            "sharpe_ratio": composite.sharpe_ratio,
            "calmar_ratio": composite.calmar_ratio,
            "win_rate": composite.win_rate,
            "profit_factor": composite.profit_factor,
            "exposure": composite.exposure,
            "ranking_score": composite.ranking_score,
        }
    )

    return pd.DataFrame(rows)
