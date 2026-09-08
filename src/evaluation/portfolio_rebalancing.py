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


def _prepare_weights(
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


def _validate_target_weights(
    target_weights: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(target_weights, Mapping):
        raise TypeError(
            "target_weights must be a mapping."
        )

    result: dict[str, float] = {}

    for strategy, weight in target_weights.items():
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid target weight for strategy '{strategy}'."
            ) from exc

        if pd.isna(numeric_weight):
            raise ValueError(
                f"Target weight for strategy '{strategy}' is invalid."
            )

        if numeric_weight < 0:
            raise ValueError(
                f"Target weight for strategy '{strategy}' cannot be negative."
            )

        result[strategy] = numeric_weight

    if sum(result.values()) > 1.0 + 1e-9:
        raise ValueError(
            "Target weights cannot sum to more than one."
        )

    return result


def calculate_rebalancing(
    report: pd.DataFrame,
    target_weights: Mapping[str, float],
) -> pd.DataFrame:
    """
    Compare current portfolio weights with target weights.

    Strategies present only in target_weights are treated as
    currently having zero allocation.
    Strategies present only in the current report are treated as
    having a target allocation of zero.
    """

    current = _prepare_weights(report)
    targets = _validate_target_weights(
        target_weights
    )

    current_map = dict(
        zip(
            current["strategy"],
            current["portfolio_weight"],
        )
    )

    strategies = list(
        dict.fromkeys(
            list(current_map)
            + list(targets)
        )
    )

    result = pd.DataFrame(
        {
            "strategy": strategies,
            "current_weight": [
                float(current_map.get(strategy, 0.0))
                for strategy in strategies
            ],
            "target_weight": [
                float(targets.get(strategy, 0.0))
                for strategy in strategies
            ],
        }
    )

    result["weight_change"] = (
        result["target_weight"]
        - result["current_weight"]
    )

    result["absolute_weight_change"] = (
        result["weight_change"].abs()
    )

    result["rebalance_required"] = (
        result["absolute_weight_change"] > 1e-9
    )

    return result


def calculate_turnover(
    rebalancing: pd.DataFrame,
) -> float:
    """
    Calculate one-way portfolio turnover.

    Turnover is half of the sum of absolute weight changes.
    """

    required = {
        "current_weight",
        "target_weight",
    }

    if not isinstance(
        rebalancing,
        pd.DataFrame,
    ):
        raise TypeError(
            "rebalancing must be a pandas DataFrame."
        )

    missing_columns = sorted(
        required.difference(
            rebalancing.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    changes = (
        pd.to_numeric(
            rebalancing["target_weight"],
            errors="coerce",
        )
        - pd.to_numeric(
            rebalancing["current_weight"],
            errors="coerce",
        )
    )

    if changes.isna().any():
        raise ValueError(
            "Rebalancing weights contain invalid values."
        )

    return float(
        changes.abs().sum() / 2.0
    )


def calculate_rebalance_capital(
    rebalancing: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Convert target/current weight changes into capital changes.
    """

    if capital < 0:
        raise ValueError(
            "capital must be non-negative."
        )

    required = {
        "strategy",
        "weight_change",
    }

    if not isinstance(
        rebalancing,
        pd.DataFrame,
    ):
        raise TypeError(
            "rebalancing must be a pandas DataFrame."
        )

    missing_columns = sorted(
        required.difference(
            rebalancing.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    result = rebalancing.copy()

    result["capital_change"] = (
        pd.to_numeric(
            result["weight_change"],
            errors="coerce",
        )
        * capital
    )

    if result["capital_change"].isna().any():
        raise ValueError(
            "weight_change contains invalid values."
        )

    return result


def build_rebalancing_plan(
    report: pd.DataFrame,
    target_weights: Mapping[str, float],
    capital: float | None = None,
) -> pd.DataFrame:
    """
    Build a complete portfolio rebalancing plan.

    If capital is provided, capital_change is included.
    """

    result = calculate_rebalancing(
        report,
        target_weights,
    )

    result["turnover"] = calculate_turnover(
        result
    )

    if capital is not None:
        result = calculate_rebalance_capital(
            result,
            capital,
        )

    return result


def should_rebalance(
    report: pd.DataFrame,
    target_weights: Mapping[str, float],
    threshold: float = 0.05,
) -> bool:
    """
    Return True when portfolio turnover exceeds the threshold.
    """

    if threshold < 0:
        raise ValueError(
            "threshold must be non-negative."
        )

    rebalancing = calculate_rebalancing(
        report,
        target_weights,
    )

    turnover = calculate_turnover(
        rebalancing
    )

    return bool(
        turnover > threshold
    )
