from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.evaluation.composite_backtest import (
    CompositeBacktestResult,
    run_composite_backtest,
)
from src.evaluation.strategy_evaluator import (
    StrategyEvaluation,
    evaluate_all_strategies,
)
from src.strategies.registry import StrategyRegistry


@dataclass(frozen=True)
class CompositeWalkForwardFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    selected_strategies: tuple[str, ...]
    evaluation: StrategyEvaluation


@dataclass(frozen=True)
class CompositeWalkForwardResult:
    name: str
    folds: tuple[CompositeWalkForwardFold, ...]
    oos_total_return: float
    mean_fold_return: float
    return_std: float
    positive_fold_rate: float
    mean_sharpe: float
    accepted: bool

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def oos_evaluations(self) -> tuple[StrategyEvaluation, ...]:
        return tuple(
            fold.evaluation
            for fold in self.folds
        )

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "fold_count": self.fold_count,
            "oos_total_return": self.oos_total_return,
            "mean_fold_return": self.mean_fold_return,
            "return_std": self.return_std,
            "positive_fold_rate": self.positive_fold_rate,
            "mean_sharpe": self.mean_sharpe,
            "accepted": self.accepted,
        }


def _validate_inputs(
    df: pd.DataFrame,
    strategy_registry: StrategyRegistry | None,
    train_size: int,
    test_size: int,
    step_size: int | None,
    top_n: int,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df must not be empty.")

    if strategy_registry is not None and not isinstance(
        strategy_registry,
        StrategyRegistry,
    ):
        raise TypeError(
            "strategy_registry must be a StrategyRegistry."
        )

    for name, value in (
        ("train_size", train_size),
        ("test_size", test_size),
        ("top_n", top_n),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if step_size is not None:
        if not isinstance(step_size, int):
            raise TypeError("step_size must be an integer.")

        if step_size <= 0:
            raise ValueError("step_size must be positive.")

    if train_size + test_size > len(df):
        raise ValueError(
            "train_size + test_size exceeds data length."
        )


def _fold_ranges(
    length: int,
    train_size: int,
    test_size: int,
    step_size: int,
):
    start = 0
    fold = 0

    while start + train_size + test_size <= length:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size

        yield (
            fold,
            train_start,
            train_end,
            test_start,
            test_end,
        )

        fold += 1
        start += step_size


def _build_strategy_outputs(
    data: pd.DataFrame,
    registry: StrategyRegistry,
) -> Mapping[str, pd.DataFrame]:
    return {
        name: registry.run(name, data)
        for name in registry.names()
    }


def _oos_total_return(
    evaluations: tuple[StrategyEvaluation, ...],
) -> float:
    if not evaluations:
        return 0.0

    equity = 1.0

    for evaluation in evaluations:
        equity *= 1.0 + evaluation.total_return

    return float(equity - 1.0)


def run_composite_walk_forward(
    df: pd.DataFrame,
    *,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    top_n: int = 3,
    strategy_registry: StrategyRegistry | None = None,
    transaction_cost: float = 0.0,
    slippage: float = 0.0,
    minimum_positive_fold_rate: float = 0.5,
    name: str = "ensemble",
) -> CompositeWalkForwardResult:
    """
    Evaluate the Top-N composite using chronological OOS folds.

    Each fold:
      1. ranks strategies on TRAIN data only;
      2. selects the Top-N strategies;
      3. builds the composite from those strategies;
      4. evaluates that fixed selection on TEST data only.

    No test-fold performance is used to select the strategies.
    """

    _validate_inputs(
        df,
        strategy_registry,
        train_size,
        test_size,
        step_size,
        top_n,
    )

    if not 0.0 <= minimum_positive_fold_rate <= 1.0:
        raise ValueError(
            "minimum_positive_fold_rate must be between 0 and 1."
        )

    composite_name = str(name).strip().lower()

    if not composite_name:
        raise ValueError("name must not be empty.")

    if " " in composite_name:
        raise ValueError("name must not contain spaces.")

    registry = (
        strategy_registry
        if strategy_registry is not None
        else StrategyRegistry.default()
    )

    step = step_size or test_size

    folds: list[CompositeWalkForwardFold] = []

    for (
        fold_number,
        train_start,
        train_end,
        test_start,
        test_end,
    ) in _fold_ranges(
        len(df),
        train_size,
        test_size,
        step,
    ):
        train = df.iloc[
            train_start:train_end
        ].copy()

        test = df.iloc[
            test_start:test_end
        ].copy()

        train_evaluations = evaluate_all_strategies(
            train,
            registry=registry,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        if train_evaluations.empty:
            raise ValueError(
                f"fold {fold_number} produced no strategy evaluations."
            )

        train_outputs = _build_strategy_outputs(
            train,
            registry,
        )

        train_composite = run_composite_backtest(
            train,
            train_outputs,
            train_evaluations,
            top_n=top_n,
            transaction_cost=transaction_cost,
            slippage=slippage,
            name=composite_name,
        )

        selected_names = train_composite.selected_strategies

        test_outputs = {
            strategy_name: registry.run(
                strategy_name,
                test,
            )
            for strategy_name in selected_names
        }

        selected_evaluations = train_evaluations[
            train_evaluations["name"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(selected_names)
        ].copy()

        test_composite = run_composite_backtest(
            test,
            test_outputs,
            selected_evaluations,
            top_n=len(selected_names),
            transaction_cost=transaction_cost,
            slippage=slippage,
            name=composite_name,
        )

        folds.append(
            CompositeWalkForwardFold(
                fold=fold_number,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                selected_strategies=selected_names,
                evaluation=test_composite.evaluation,
            )
        )

    if not folds:
        raise ValueError(
            "walk-forward configuration produced no folds."
        )

    evaluations = tuple(
        fold.evaluation
        for fold in folds
    )

    fold_returns = [
        evaluation.total_return
        for evaluation in evaluations
    ]

    positive_rate = sum(
        value > 0
        for value in fold_returns
    ) / len(fold_returns)

    mean_return = float(
        sum(fold_returns) / len(fold_returns)
    )

    return_std = float(
        pd.Series(fold_returns).std(
            ddof=0
        )
    )

    mean_sharpe = float(
        sum(
            evaluation.sharpe_ratio
            for evaluation in evaluations
        )
        / len(evaluations)
    )

    accepted = (
        positive_rate >= minimum_positive_fold_rate
    )

    return CompositeWalkForwardResult(
        name=composite_name,
        folds=tuple(folds),
        oos_total_return=_oos_total_return(
            evaluations
        ),
        mean_fold_return=mean_return,
        return_std=return_std,
        positive_fold_rate=positive_rate,
        mean_sharpe=mean_sharpe,
        accepted=accepted,
    )
