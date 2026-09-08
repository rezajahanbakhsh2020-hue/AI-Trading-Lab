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
        raise ValueError(
            "strategy contains missing values."
        )

    if (result["strategy"] == "").any():
        raise ValueError(
            "strategy contains empty values."
        )

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


def calculate_normalized_weights(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return portfolio weights normalized to their total.

    An empty report returns an empty DataFrame.
    A report whose total weight is zero raises ValueError.
    """

    result = _prepare_report(report)

    if result.empty:
        return result

    total = float(
        result["portfolio_weight"].sum()
    )

    if total <= 0:
        raise ValueError(
            "portfolio weight total must be greater than zero."
        )

    result["normalized_weight"] = (
        result["portfolio_weight"] / total
    )

    return result


def calculate_weight_entropy(
    report: pd.DataFrame,
) -> float:
    """
    Calculate normalized Shannon entropy of portfolio weights.

    The result is between zero and one when at least one
    strategy has positive weight. Higher values indicate
    a more evenly distributed portfolio.
    """

    normalized = calculate_normalized_weights(
        report
    )

    if normalized.empty:
        return 0.0

    weights = normalized["normalized_weight"]
    positive_weights = weights[weights > 0]

    if len(positive_weights) <= 1:
        return 0.0

    entropy = float(
        -(
            positive_weights
            * positive_weights.map(
                lambda value: __import__("math").log(value)
            )
        ).sum()
    )

    strategy_count = len(positive_weights)

    if strategy_count <= 1:
        return 0.0

    return float(
        entropy
        / __import__("math").log(strategy_count)
    )


def calculate_equal_weight_gap(
    report: pd.DataFrame,
) -> float:
    """
    Measure the average absolute distance from equal weighting.

    A value of zero means all strategies have equal weight.
    """

    normalized = calculate_normalized_weights(
        report
    )

    if normalized.empty:
        return 0.0

    strategy_count = len(normalized)

    if strategy_count == 0:
        return 0.0

    equal_weight = 1.0 / strategy_count

    return float(
        (
            normalized["normalized_weight"]
            - equal_weight
        )
        .abs()
        .mean()
    )


def calculate_active_strategy_count(
    report: pd.DataFrame,
    minimum_weight: float = 1e-9,
) -> int:
    """
    Count strategies with a meaningful positive weight.
    """

    if minimum_weight < 0:
        raise ValueError(
            "minimum_weight cannot be negative."
        )

    normalized = calculate_normalized_weights(
        report
    )

    if normalized.empty:
        return 0

    return int(
        (
            normalized["normalized_weight"]
            > minimum_weight
        ).sum()
    )


def validate_diversification(
    report: pd.DataFrame,
    minimum_active_strategies: int = 2,
    minimum_entropy: float = 0.0,
) -> bool:
    """
    Validate basic diversification requirements.
    """

    if minimum_active_strategies < 0:
        raise ValueError(
            "minimum_active_strategies cannot be negative."
        )

    if not 0 <= minimum_entropy <= 1:
        raise ValueError(
            "minimum_entropy must be between zero and one."
        )

    normalized = calculate_normalized_weights(
        report
    )

    if normalized.empty:
        return minimum_active_strategies == 0

    active_count = calculate_active_strategy_count(
        report
    )

    entropy = calculate_weight_entropy(
        report
    )

    return bool(
        active_count >= minimum_active_strategies
        and entropy >= minimum_entropy
    )


def build_diversification_summary(
    report: pd.DataFrame,
    minimum_active_strategies: int = 2,
    minimum_entropy: float = 0.0,
) -> dict[str, object]:
    """
    Build a structured diversification summary.
    """

    _validate_report(report)

    normalized = calculate_normalized_weights(
        report
    )

    active_count = calculate_active_strategy_count(
        report
    )

    entropy = calculate_weight_entropy(
        report
    )

    equal_weight_gap = calculate_equal_weight_gap(
        report
    )

    passed = validate_diversification(
        report,
        minimum_active_strategies=minimum_active_strategies,
        minimum_entropy=minimum_entropy,
    )

    return {
        "strategy_count": int(len(normalized)),
        "active_strategy_count": active_count,
        "weight_entropy": entropy,
        "equal_weight_gap": equal_weight_gap,
        "minimum_active_strategies": (
            minimum_active_strategies
        ),
        "minimum_entropy": minimum_entropy,
        "passed": bool(passed),
    }


def summarize_target_weights(
    target_weights: Mapping[str, float],
) -> dict[str, object]:
    """
    Build a diversification summary from a target-weight mapping.
    """

    if not isinstance(target_weights, Mapping):
        raise TypeError(
            "target_weights must be a mapping."
        )

    rows: list[dict[str, object]] = []

    for strategy, weight in target_weights.items():
        if strategy is None:
            raise ValueError(
                "strategy cannot be None."
            )

        strategy_name = str(strategy).strip()

        if not strategy_name:
            raise ValueError(
                "strategy cannot be empty."
            )

        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "target_weights contains an invalid weight."
            ) from exc

        if pd.isna(numeric_weight):
            raise ValueError(
                "target_weights contains a missing weight."
            )

        if numeric_weight < 0:
            raise ValueError(
                "target_weights cannot contain negative weights."
            )

        rows.append(
            {
                "strategy": strategy_name,
                "portfolio_weight": numeric_weight,
            }
        )

    report = pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "portfolio_weight",
        ],
    )

    return build_diversification_summary(
        report
    )
