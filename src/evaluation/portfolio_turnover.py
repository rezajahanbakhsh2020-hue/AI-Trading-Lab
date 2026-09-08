from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "portfolio_weight",
}


def _validate_report(report: pd.DataFrame) -> None:
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame.")

    missing = sorted(
        REQUIRED_COLUMNS.difference(report.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def _prepare_report(report: pd.DataFrame) -> pd.DataFrame:
    _validate_report(report)

    result = report[
        ["strategy", "portfolio_weight"]
    ].copy()

    result["strategy"] = (
        result["strategy"]
        .astype("string")
        .str.strip()
    )

    result["portfolio_weight"] = pd.to_numeric(
        result["portfolio_weight"],
        errors="coerce",
    )

    if result["strategy"].isna().any():
        raise ValueError("strategy contains missing values.")

    if (result["strategy"] == "").any():
        raise ValueError("strategy contains empty values.")

    if result["portfolio_weight"].isna().any():
        raise ValueError(
            "portfolio_weight contains invalid values."
        )

    if (result["portfolio_weight"] < 0).any():
        raise ValueError(
            "portfolio_weight cannot be negative."
        )

    if result["strategy"].duplicated().any():
        raise ValueError(
            "strategy names must be unique."
        )

    return result


def calculate_turnover(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> float:
    """
    Calculate one-way portfolio turnover between two allocations.

    Turnover is half of the sum of absolute weight changes.
    """

    previous_result = _prepare_report(previous)
    current_result = _prepare_report(current)

    previous_weights = previous_result.set_index(
        "strategy"
    )["portfolio_weight"]

    current_weights = current_result.set_index(
        "strategy"
    )["portfolio_weight"]

    combined = pd.concat(
        [previous_weights, current_weights],
        axis=1,
        keys=["previous", "current"],
    ).fillna(0.0)

    return float(
        (combined["current"] - combined["previous"])
        .abs()
        .sum()
        / 2.0
    )


def calculate_rebalance_amount(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> float:
    """
    Calculate the total absolute amount that must be traded
    to move from the previous allocation to the current one.
    """

    previous_result = _prepare_report(previous)
    current_result = _prepare_report(current)

    previous_weights = previous_result.set_index(
        "strategy"
    )["portfolio_weight"]

    current_weights = current_result.set_index(
        "strategy"
    )["portfolio_weight"]

    combined = pd.concat(
        [previous_weights, current_weights],
        axis=1,
        keys=["previous", "current"],
    ).fillna(0.0)

    return float(
        (
            combined["current"]
            - combined["previous"]
        )
        .abs()
        .sum()
    )


def find_increased_positions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    tolerance: float = 1e-9,
) -> list[str]:
    """
    Return strategies whose portfolio weight increased.
    """

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    previous_result = _prepare_report(previous)
    current_result = _prepare_report(current)

    previous_weights = previous_result.set_index(
        "strategy"
    )["portfolio_weight"]

    current_weights = current_result.set_index(
        "strategy"
    )["portfolio_weight"]

    combined = pd.concat(
        [previous_weights, current_weights],
        axis=1,
        keys=["previous", "current"],
    ).fillna(0.0)

    increased = combined.loc[
        combined["current"]
        > combined["previous"] + tolerance
    ]

    return increased.index.tolist()


def find_decreased_positions(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    tolerance: float = 1e-9,
) -> list[str]:
    """
    Return strategies whose portfolio weight decreased.
    """

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    previous_result = _prepare_report(previous)
    current_result = _prepare_report(current)

    previous_weights = previous_result.set_index(
        "strategy"
    )["portfolio_weight"]

    current_weights = current_result.set_index(
        "strategy"
    )["portfolio_weight"]

    combined = pd.concat(
        [previous_weights, current_weights],
        axis=1,
        keys=["previous", "current"],
    ).fillna(0.0)

    decreased = combined.loc[
        combined["current"]
        < combined["previous"] - tolerance
    ]

    return decreased.index.tolist()


def validate_turnover_limit(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    maximum_turnover: float = 0.50,
) -> bool:
    """
    Validate that turnover does not exceed the configured limit.
    """

    if not 0 <= maximum_turnover <= 1:
        raise ValueError(
            "maximum_turnover must be between zero and one."
        )

    turnover = calculate_turnover(
        previous,
        current,
    )

    return turnover <= maximum_turnover + 1e-9


def build_turnover_summary(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    maximum_turnover: float = 0.50,
) -> dict[str, object]:
    """
    Build a structured portfolio turnover summary.
    """

    turnover = calculate_turnover(
        previous,
        current,
    )

    rebalance_amount = calculate_rebalance_amount(
        previous,
        current,
    )

    increased = find_increased_positions(
        previous,
        current,
    )

    decreased = find_decreased_positions(
        previous,
        current,
    )

    passed = validate_turnover_limit(
        previous,
        current,
        maximum_turnover=maximum_turnover,
    )

    return {
        "turnover": turnover,
        "rebalance_amount": rebalance_amount,
        "increased_positions": increased,
        "decreased_positions": decreased,
        "increased_count": len(increased),
        "decreased_count": len(decreased),
        "maximum_turnover": maximum_turnover,
        "passed": bool(passed),
    }


def summarize_target_weight_change(
    previous_weights: Mapping[str, float],
    current_weights: Mapping[str, float],
) -> dict[str, object]:
    """
    Calculate turnover directly from two target-weight mappings.
    """

    if not isinstance(previous_weights, Mapping):
        raise TypeError(
            "previous_weights must be a mapping."
        )

    if not isinstance(current_weights, Mapping):
        raise TypeError(
            "current_weights must be a mapping."
        )

    def to_report(
        weights: Mapping[str, float],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []

        for strategy, weight in weights.items():
            if strategy is None:
                raise ValueError(
                    "strategy cannot be None."
                )

            name = str(strategy).strip()

            if not name:
                raise ValueError(
                    "strategy cannot be empty."
                )

            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "target weights contain an invalid value."
                ) from exc

            if pd.isna(numeric_weight):
                raise ValueError(
                    "target weights contain a missing value."
                )

            if numeric_weight < 0:
                raise ValueError(
                    "target weights cannot be negative."
                )

            rows.append(
                {
                    "strategy": name,
                    "portfolio_weight": numeric_weight,
                }
            )

        return pd.DataFrame(
            rows,
            columns=[
                "strategy",
                "portfolio_weight",
            ],
        )

    previous_report = to_report(previous_weights)
    current_report = to_report(current_weights)

    return build_turnover_summary(
        previous_report,
        current_report,
    )
