from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "strategy_return",
}


def _validate_strategy_frame(
    frame: pd.DataFrame,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "Each strategy result must be a pandas DataFrame."
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


def _validate_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(weights, Mapping):
        raise TypeError(
            "weights must be a mapping of strategy names to weights."
        )

    result: dict[str, float] = {}

    for strategy, weight in weights.items():
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid weight for strategy '{strategy}'."
            ) from exc

        if pd.isna(numeric_weight):
            raise ValueError(
                f"Weight for strategy '{strategy}' is invalid."
            )

        if numeric_weight < 0:
            raise ValueError(
                f"Weight for strategy '{strategy}' cannot be negative."
            )

        result[strategy] = numeric_weight

    total_weight = sum(result.values())

    if total_weight > 1.0 + 1e-9:
        raise ValueError(
            "Portfolio weights cannot sum to more than one."
        )

    return result


def _prepare_strategy_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    _validate_strategy_frame(frame)

    result = frame[
        [
            "timestamp",
            "strategy_return",
        ]
    ].copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    if result["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains invalid values."
        )

    result["strategy_return"] = pd.to_numeric(
        result["strategy_return"],
        errors="coerce",
    )

    if result["strategy_return"].isna().any():
        raise ValueError(
            "strategy_return contains invalid values."
        )

    result = (
        result
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


def calculate_portfolio_returns(
    strategy_results: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """
    Combine strategy return streams into a weighted portfolio return.

    Strategies are aligned on timestamp. Missing observations are
    treated as zero return for that strategy.

    Weights may sum to less than one; the remainder stays in cash.
    """

    if not isinstance(strategy_results, Mapping):
        raise TypeError(
            "strategy_results must be a mapping."
        )

    validated_weights = _validate_weights(weights)

    if not strategy_results:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "portfolio_return",
            ]
        )

    missing_strategies = sorted(
        set(strategy_results).difference(
            validated_weights
        )
    )

    if missing_strategies:
        raise ValueError(
            "Missing weights for strategies: "
            f"{missing_strategies}"
        )

    weighted_frames: list[pd.DataFrame] = []

    for strategy, frame in strategy_results.items():
        prepared = _prepare_strategy_frame(frame)

        prepared = prepared.rename(
            columns={
                "strategy_return": strategy,
            }
        )

        prepared[strategy] = (
            prepared[strategy]
            * validated_weights[strategy]
        )

        weighted_frames.append(prepared)

    result = weighted_frames[0]

    for frame in weighted_frames[1:]:
        result = result.merge(
            frame,
            on="timestamp",
            how="outer",
        )

    strategy_columns = list(strategy_results.keys())

    result[strategy_columns] = result[
        strategy_columns
    ].fillna(0.0)

    result["portfolio_return"] = result[
        strategy_columns
    ].sum(axis=1)

    result = (
        result[
            [
                "timestamp",
                "portfolio_return",
            ]
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return result


def calculate_portfolio_equity(
    portfolio_returns: pd.DataFrame,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate portfolio equity from portfolio returns.
    """

    if not isinstance(
        portfolio_returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "portfolio_returns must be a pandas DataFrame."
        )

    required = {
        "timestamp",
        "portfolio_return",
    }

    missing_columns = sorted(
        required.difference(
            portfolio_returns.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if initial_capital < 0:
        raise ValueError(
            "initial_capital must be non-negative."
        )

    result = portfolio_returns.copy()

    result["portfolio_return"] = pd.to_numeric(
        result["portfolio_return"],
        errors="coerce",
    )

    if result["portfolio_return"].isna().any():
        raise ValueError(
            "portfolio_return contains invalid values."
        )

    result = result.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    result["equity"] = (
        initial_capital
        * (1.0 + result["portfolio_return"]).cumprod()
    )

    return result


def calculate_portfolio_drawdown(
    portfolio_equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate running peak and drawdown from portfolio equity.
    """

    if not isinstance(
        portfolio_equity,
        pd.DataFrame,
    ):
        raise TypeError(
            "portfolio_equity must be a pandas DataFrame."
        )

    if "equity" not in portfolio_equity.columns:
        raise ValueError(
            "Missing required column: ['equity']"
        )

    result = portfolio_equity.copy()

    result["equity"] = pd.to_numeric(
        result["equity"],
        errors="coerce",
    )

    if result["equity"].isna().any():
        raise ValueError(
            "equity contains invalid values."
        )

    result["equity_peak"] = (
        result["equity"].cummax()
    )

    result["drawdown"] = (
        result["equity"]
        / result["equity_peak"]
        - 1.0
    )

    return result


def calculate_max_drawdown(
    portfolio_equity: pd.DataFrame,
) -> float:
    """
    Return the maximum portfolio drawdown.
    """

    result = calculate_portfolio_drawdown(
        portfolio_equity
    )

    if result.empty:
        return 0.0

    return float(
        result["drawdown"].min()
    )


def build_portfolio_performance(
    strategy_results: Mapping[str, pd.DataFrame],
    weights: Mapping[str, float],
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """
    Build the complete portfolio performance series.
    """

    returns = calculate_portfolio_returns(
        strategy_results,
        weights,
    )

    equity = calculate_portfolio_equity(
        returns,
        initial_capital=initial_capital,
    )

    return calculate_portfolio_drawdown(
        equity
    )
