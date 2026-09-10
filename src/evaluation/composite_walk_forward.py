from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from src.evaluation.strategy_evaluator import (
    StrategyEvaluation,
    evaluate_all_strategies,
)
from src.evaluation.composite_backtest import run_composite_backtest
from src.strategies.registry import DEFAULT_REGISTRY, StrategyRegistry


@dataclass(frozen=True)
class CompositeWalkForwardFold:
    fold: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
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
        return tuple(fold.evaluation for fold in self.folds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fold_count": self.fold_count,
            "oos_total_return": self.oos_total_return,
            "mean_fold_return": self.mean_fold_return,
            "return_std": self.return_std,
            "positive_fold_rate": self.positive_fold_rate,
            "mean_sharpe": self.mean_sharpe,
            "accepted": self.accepted,
            "folds": [
                {
                    **asdict(fold),
                    "evaluation": asdict(fold.evaluation),
                }
                for fold in self.folds
            ],
        }


def _validate_inputs(
    df: pd.DataFrame,
    strategy_registry: StrategyRegistry | None,
    train_size: int,
    test_size: int,
    step_size: int,
    top_n: int,
) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("df must not be empty.")

    required_columns = {"timestamp", "open", "high", "low", "close"}
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"df is missing required columns: {sorted(missing)}"
        )

    if strategy_registry is not None and not isinstance(
        strategy_registry,
        StrategyRegistry,
    ):
        raise TypeError(
            "strategy_registry must be a StrategyRegistry or None."
        )

    for value, label in (
        (train_size, "train_size"),
        (test_size, "test_size"),
        (step_size, "step_size"),
        (top_n, "top_n"),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} must be an integer.")

        if value <= 0:
            raise ValueError(f"{label} must be positive.")


def _fold_ranges(
    length: int,
    train_size: int,
    test_size: int,
    step_size: int,
) -> list[tuple[int, int, int, int]]:
    ranges: list[tuple[int, int, int, int]] = []

    start = 0

    while start + train_size + test_size <= length:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size

        ranges.append(
            (
                train_start,
                train_end,
                test_start,
                test_end,
            )
        )

        start += step_size

    return ranges


def _build_strategy_outputs(
    df: pd.DataFrame,
    registry: StrategyRegistry,
    names: list[str],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    for name in names:
        spec = registry.get(name)

        result = spec.function(df.copy())

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Strategy '{name}' must return a pandas DataFrame."
            )

        if "signal" not in result.columns:
            raise ValueError(
                f"Strategy '{name}' output must contain 'signal'."
            )

        outputs[name] = result

    return outputs


def _oos_total_return(
    evaluations: tuple[StrategyEvaluation, ...],
) -> float:
    if not evaluations:
        return 0.0

    equity = 1.0

    for evaluation in evaluations:
        equity *= 1.0 + float(evaluation.total_return)

    return float(equity - 1.0)


def run_composite_walk_forward(
    df: pd.DataFrame,
    train_size: int = 80,
    test_size: int = 20,
    step_size: int = 20,
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
        else DEFAULT_REGISTRY
    )

    if not registry.names():
        raise ValueError(
            "strategy_registry must contain at least one strategy."
        )

    available_count = len(registry.names())

    if top_n > available_count:
        top_n = available_count

    ranges = _fold_ranges(
        len(df),
        train_size,
        test_size,
        step_size,
    )

    if not ranges:
        raise ValueError(
            "df does not contain enough rows for the requested "
            "train_size and test_size."
        )

    folds: list[CompositeWalkForwardFold] = []

    for fold_number, (
        train_start,
        train_end,
        test_start,
        test_end,
    ) in enumerate(ranges, start=1):
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        train_evaluations = evaluate_all_strategies(
            train_df,
            registry=registry,
            transaction_cost=transaction_cost,
            slippage=slippage,
        )

        selected_evaluations = train_evaluations.head(top_n).copy()

        selected_names = [
            str(value)
            for value in selected_evaluations["name"].tolist()
        ]

        if not selected_names:
            raise ValueError(
                "No strategies were selected for a walk-forward fold."
            )

        test_outputs = _build_strategy_outputs(
            test_df,
            registry,
            selected_names,
        )

        composite_result = run_composite_backtest(
            test_df,
            test_outputs,
            selected_evaluations,
            top_n=len(selected_names),
            transaction_cost=transaction_cost,
            slippage=slippage,
            name=composite_name,
        )

        evaluation = composite_result.evaluation

        train_start_value = train_df["timestamp"].iloc[0]
        train_end_value = train_df["timestamp"].iloc[-1]
        test_start_value = test_df["timestamp"].iloc[0]
        test_end_value = test_df["timestamp"].iloc[-1]

        folds.append(
            CompositeWalkForwardFold(
                fold=fold_number,
                train_start=train_start_value,
                train_end=train_end_value,
                test_start=test_start_value,
                test_end=test_end_value,
                selected_strategies=tuple(selected_names),
                evaluation=evaluation,
            )
        )

    fold_returns = [
        float(fold.evaluation.total_return)
        for fold in folds
    ]

    fold_sharpes = [
        float(fold.evaluation.sharpe_ratio)
        for fold in folds
    ]

    positive_fold_rate = sum(
        value > 0.0 for value in fold_returns
    ) / len(fold_returns)

    mean_fold_return = float(
        sum(fold_returns) / len(fold_returns)
    )

    return_std = float(
        pd.Series(fold_returns, dtype=float).std(ddof=0)
    )

    mean_sharpe = float(
        sum(fold_sharpes) / len(fold_sharpes)
    )

    accepted = (
        positive_fold_rate >= minimum_positive_fold_rate
    )

    return CompositeWalkForwardResult(
        name=composite_name,
        folds=tuple(folds),
        oos_total_return=_oos_total_return(
            tuple(fold.evaluation for fold in folds)
        ),
        mean_fold_return=mean_fold_return,
        return_std=return_std,
        positive_fold_rate=float(positive_fold_rate),
        mean_sharpe=mean_sharpe,
        accepted=accepted,
    )
