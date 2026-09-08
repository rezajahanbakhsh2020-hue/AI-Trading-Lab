from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "portfolio_weight",
}


def _validate_report(report: pd.DataFrame) -> None:
    if not isinstance(report, pd.DataFrame):
        raise TypeError(
            "report must be a pandas DataFrame."
        )

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(report.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


def _prepare_report(
    report: pd.DataFrame,
) -> pd.DataFrame:
    _validate_report(report)

    result = report[
        [
            "strategy",
            "portfolio_weight",
        ]
    ].copy()

    result["portfolio_weight"] = pd.to_numeric(
        result["portfolio_weight"],
        errors="coerce",
    )

    if result["portfolio_weight"].isna().any():
        raise ValueError(
            "portfolio_weight contains invalid values."
        )

    if (result["portfolio_weight"] < 0).any():
        raise ValueError(
            "portfolio_weight cannot be negative."
        )

    if result["portfolio_weight"].sum() > 1.0 + 1e-9:
        raise ValueError(
            "portfolio_weight cannot sum to more than one."
        )

    return result


def _validate_limits(
    max_strategy_weight: float,
    min_strategy_weight: float,
    max_strategies: int | None,
) -> None:
    if not 0 <= min_strategy_weight <= 1:
        raise ValueError(
            "min_strategy_weight must be between zero and one."
        )

    if not 0 <= max_strategy_weight <= 1:
        raise ValueError(
            "max_strategy_weight must be between zero and one."
        )

    if min_strategy_weight > max_strategy_weight:
        raise ValueError(
            "min_strategy_weight cannot exceed max_strategy_weight."
        )

    if max_strategies is not None and max_strategies < 1:
        raise ValueError(
            "max_strategies must be at least one."
        )


def check_weight_constraints(
    report: pd.DataFrame,
    max_strategy_weight: float = 0.60,
    min_strategy_weight: float = 0.0,
    max_strategies: int | None = None,
) -> dict[str, object]:
    """
    Check individual strategy weight and strategy-count constraints.
    """

    _validate_limits(
        max_strategy_weight,
        min_strategy_weight,
        max_strategies,
    )

    result = _prepare_report(report)

    active = result[
        result["portfolio_weight"] > 1e-9
    ]

    max_weight = (
        0.0
        if result.empty
        else float(result["portfolio_weight"].max())
    )

    min_active_weight = (
        0.0
        if active.empty
        else float(active["portfolio_weight"].min())
    )

    strategy_count = len(active)

    max_weight_ok = (
        max_weight
        <= max_strategy_weight + 1e-9
    )

    min_weight_ok = (
        active.empty
        or min_active_weight
        >= min_strategy_weight - 1e-9
    )

    strategy_count_ok = (
        max_strategies is None
        or strategy_count <= max_strategies
    )

    return {
        "passed": bool(
            max_weight_ok
            and min_weight_ok
            and strategy_count_ok
        ),
        "max_weight": max_weight,
        "min_active_weight": min_active_weight,
        "strategy_count": strategy_count,
        "max_weight_ok": max_weight_ok,
        "min_weight_ok": min_weight_ok,
        "strategy_count_ok": strategy_count_ok,
    }


def apply_weight_constraints(
    report: pd.DataFrame,
    max_strategy_weight: float = 0.60,
    min_strategy_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Remove strategies outside the allowed individual weight range.

    Weights are not renormalized automatically.
    """

    _validate_limits(
        max_strategy_weight,
        min_strategy_weight,
        None,
    )

    result = _prepare_report(report)

    if result.empty:
        return result

    mask = (
        (result["portfolio_weight"] <= max_strategy_weight + 1e-9)
        & (
            (result["portfolio_weight"] >= min_strategy_weight - 1e-9)
            | (result["portfolio_weight"] <= 1e-9)
        )
    )

    return result.loc[
        mask
    ].reset_index(drop=True)


def enforce_max_strategy_count(
    report: pd.DataFrame,
    max_strategies: int,
) -> pd.DataFrame:
    """
    Keep at most max_strategies by portfolio weight.

    The selected rows retain their original weights.
    """

    if max_strategies < 1:
        raise ValueError(
            "max_strategies must be at least one."
        )

    result = _prepare_report(report)

    if result.empty:
        return result

    return (
        result
        .sort_values(
            "portfolio_weight",
            ascending=False,
        )
        .head(max_strategies)
        .reset_index(drop=True)
    )


def cap_strategy_weights(
    report: pd.DataFrame,
    max_strategy_weight: float = 0.60,
) -> pd.DataFrame:
    """
    Cap every strategy weight at the specified maximum.
    """

    if not 0 <= max_strategy_weight <= 1:
        raise ValueError(
            "max_strategy_weight must be between zero and one."
        )

    result = _prepare_report(report)

    result["portfolio_weight"] = (
        result["portfolio_weight"]
        .clip(upper=max_strategy_weight)
    )

    return result


def normalize_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    """
    Normalize positive strategy weights so their total equals one.

    Empty or all-zero weights return an empty allocation.
    """

    if not isinstance(weights, Mapping):
        raise TypeError(
            "weights must be a mapping."
        )

    numeric_weights: dict[str, float] = {}

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

        numeric_weights[strategy] = numeric_weight

    total = sum(numeric_weights.values())

    if total <= 0:
        return {
            strategy: 0.0
            for strategy in numeric_weights
        }

    return {
        strategy: weight / total
        for strategy, weight in numeric_weights.items()
    }


def validate_portfolio_constraints(
    report: pd.DataFrame,
    max_strategy_weight: float = 0.60,
    min_strategy_weight: float = 0.0,
    max_strategies: int | None = None,
) -> bool:
    """
    Return True when all portfolio constraints are satisfied.
    """

    result = check_weight_constraints(
        report,
        max_strategy_weight=max_strategy_weight,
        min_strategy_weight=min_strategy_weight,
        max_strategies=max_strategies,
    )

    return bool(result["passed"])
