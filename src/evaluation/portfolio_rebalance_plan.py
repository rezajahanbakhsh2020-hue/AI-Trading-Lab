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

    missing_columns = sorted(
        REQUIRED_COLUMNS.difference(report.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
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

    if result["strategy"].duplicated().any():
        raise ValueError("strategy names must be unique.")

    if result["portfolio_weight"].isna().any():
        raise ValueError(
            "portfolio_weight contains invalid values."
        )

    if (result["portfolio_weight"] < 0).any():
        raise ValueError(
            "portfolio_weight cannot be negative."
        )

    return result


def build_rebalance_plan(
    current: pd.DataFrame,
    target: pd.DataFrame,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Build a strategy-level rebalance plan.

    Positive change means increasing allocation.
    Negative change means decreasing allocation.
    """

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    current_result = _prepare_report(current)
    target_result = _prepare_report(target)

    current_weights = current_result.set_index(
        "strategy"
    )["portfolio_weight"]

    target_weights = target_result.set_index(
        "strategy"
    )["portfolio_weight"]

    combined = pd.concat(
        [current_weights, target_weights],
        axis=1,
        keys=["current_weight", "target_weight"],
    ).fillna(0.0)

    combined["weight_change"] = (
        combined["target_weight"]
        - combined["current_weight"]
    )

    combined["action"] = "hold"

    combined.loc[
        combined["weight_change"] > tolerance,
        "action",
    ] = "increase"

    combined.loc[
        combined["weight_change"] < -tolerance,
        "action",
    ] = "decrease"

    combined["absolute_change"] = (
        combined["weight_change"].abs()
    )

    combined = combined.reset_index()

    return combined[
        [
            "strategy",
            "current_weight",
            "target_weight",
            "weight_change",
            "absolute_change",
            "action",
        ]
    ]


def calculate_buy_weight(
    plan: pd.DataFrame,
) -> float:
    """
    Calculate the total weight that must be increased.
    """

    required = {
        "weight_change",
    }

    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a pandas DataFrame.")

    missing = sorted(
        required.difference(plan.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    changes = pd.to_numeric(
        plan["weight_change"],
        errors="coerce",
    )

    if changes.isna().any():
        raise ValueError(
            "weight_change contains invalid values."
        )

    return float(
        changes[changes > 0].sum()
    )


def calculate_sell_weight(
    plan: pd.DataFrame,
) -> float:
    """
    Calculate the total weight that must be decreased.
    """

    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a pandas DataFrame.")

    if "weight_change" not in plan.columns:
        raise ValueError(
            "Missing required columns: ['weight_change']"
        )

    changes = pd.to_numeric(
        plan["weight_change"],
        errors="coerce",
    )

    if changes.isna().any():
        raise ValueError(
            "weight_change contains invalid values."
        )

    return float(
        -changes[changes < 0].sum()
    )


def calculate_rebalance_turnover(
    plan: pd.DataFrame,
) -> float:
    """
    Calculate one-way turnover represented by a rebalance plan.
    """

    buy_weight = calculate_buy_weight(plan)
    sell_weight = calculate_sell_weight(plan)

    return float(
        (buy_weight + sell_weight) / 2.0
    )


def count_rebalance_actions(
    plan: pd.DataFrame,
) -> dict[str, int]:
    """
    Count increase, decrease, and hold actions.
    """

    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a pandas DataFrame.")

    if "action" not in plan.columns:
        raise ValueError(
            "Missing required columns: ['action']"
        )

    return {
        "increase": int(
            (plan["action"] == "increase").sum()
        ),
        "decrease": int(
            (plan["action"] == "decrease").sum()
        ),
        "hold": int(
            (plan["action"] == "hold").sum()
        ),
    }


def validate_rebalance_plan(
    plan: pd.DataFrame,
    maximum_turnover: float = 0.50,
) -> bool:
    """
    Validate that a rebalance plan stays within the turnover limit.
    """

    if not 0 <= maximum_turnover <= 1:
        raise ValueError(
            "maximum_turnover must be between zero and one."
        )

    turnover = calculate_rebalance_turnover(plan)

    return turnover <= maximum_turnover + 1e-9


def summarize_rebalance_plan(
    plan: pd.DataFrame,
    maximum_turnover: float = 0.50,
) -> dict[str, object]:
    """
    Return a structured summary of a rebalance plan.
    """

    actions = count_rebalance_actions(plan)
    buy_weight = calculate_buy_weight(plan)
    sell_weight = calculate_sell_weight(plan)
    turnover = calculate_rebalance_turnover(plan)

    return {
        "buy_weight": buy_weight,
        "sell_weight": sell_weight,
        "turnover": turnover,
        "increase_count": actions["increase"],
        "decrease_count": actions["decrease"],
        "hold_count": actions["hold"],
        "maximum_turnover": maximum_turnover,
        "passed": validate_rebalance_plan(
            plan,
            maximum_turnover=maximum_turnover,
        ),
    }


def summarize_target_weights(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> dict[str, object]:
    """
    Build a rebalance summary directly from two weight mappings.
    """

    if not isinstance(current_weights, Mapping):
        raise TypeError(
            "current_weights must be a mapping."
        )

    if not isinstance(target_weights, Mapping):
        raise TypeError(
            "target_weights must be a mapping."
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
                    "weight contains an invalid value."
                ) from exc

            if pd.isna(numeric_weight):
                raise ValueError(
                    "weight contains a missing value."
                )

            if numeric_weight < 0:
                raise ValueError(
                    "weight cannot be negative."
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

    current_report = to_report(current_weights)
    target_report = to_report(target_weights)

    plan = build_rebalance_plan(
        current_report,
        target_report,
    )

    return summarize_rebalance_plan(plan)
