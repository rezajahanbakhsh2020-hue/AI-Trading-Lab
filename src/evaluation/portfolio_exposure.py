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


def calculate_total_exposure(
    report: pd.DataFrame,
) -> float:
    """
    Calculate the total portfolio exposure.
    """

    result = _prepare_report(report)

    return float(
        result["portfolio_weight"].sum()
    )


def calculate_largest_exposure(
    report: pd.DataFrame,
) -> float:
    """
    Return the largest individual strategy exposure.
    """

    result = _prepare_report(report)

    if result.empty:
        return 0.0

    return float(
        result["portfolio_weight"].max()
    )


def calculate_smallest_active_exposure(
    report: pd.DataFrame,
    minimum_weight: float = 1e-9,
) -> float:
    """
    Return the smallest meaningful positive exposure.
    """

    if minimum_weight < 0:
        raise ValueError(
            "minimum_weight cannot be negative."
        )

    result = _prepare_report(report)

    active = result.loc[
        result["portfolio_weight"] > minimum_weight,
        "portfolio_weight",
    ]

    if active.empty:
        return 0.0

    return float(active.min())


def calculate_exposure_ratio(
    report: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add each strategy's percentage of total portfolio exposure.
    """

    result = _prepare_report(report)

    if result.empty:
        result["exposure_ratio"] = pd.Series(
            dtype=float
        )
        return result

    total = float(
        result["portfolio_weight"].sum()
    )

    if total <= 0:
        result["exposure_ratio"] = 0.0
        return result

    result["exposure_ratio"] = (
        result["portfolio_weight"] / total
    )

    return result


def find_overexposed_strategies(
    report: pd.DataFrame,
    maximum_weight: float = 0.50,
) -> list[str]:
    """
    Return strategies whose individual exposure exceeds
    the configured maximum.
    """

    if not 0 <= maximum_weight <= 1:
        raise ValueError(
            "maximum_weight must be between zero and one."
        )

    result = _prepare_report(report)

    return result.loc[
        result["portfolio_weight"] > maximum_weight,
        "strategy",
    ].tolist()


def calculate_exposure_buckets(
    report: pd.DataFrame,
) -> dict[str, float]:
    """
    Group portfolio exposure into practical weight buckets.
    """

    result = _prepare_report(report)

    if result.empty:
        return {
            "zero": 0.0,
            "small": 0.0,
            "medium": 0.0,
            "large": 0.0,
        }

    weights = result["portfolio_weight"]

    return {
        "zero": float(
            weights[weights == 0].sum()
        ),
        "small": float(
            weights[
                (weights > 0)
                & (weights <= 0.10)
            ].sum()
        ),
        "medium": float(
            weights[
                (weights > 0.10)
                & (weights <= 0.25)
            ].sum()
        ),
        "large": float(
            weights[weights > 0.25].sum()
        ),
    }


def validate_exposure(
    report: pd.DataFrame,
    maximum_total_exposure: float = 1.0,
    maximum_strategy_exposure: float = 0.50,
) -> bool:
    """
    Validate total and individual portfolio exposure.
    """

    if not 0 <= maximum_total_exposure <= 1:
        raise ValueError(
            "maximum_total_exposure must be between zero and one."
        )

    if not 0 <= maximum_strategy_exposure <= 1:
        raise ValueError(
            "maximum_strategy_exposure must be between zero and one."
        )

    total = calculate_total_exposure(report)
    largest = calculate_largest_exposure(report)

    return bool(
        total <= maximum_total_exposure + 1e-9
        and largest <= maximum_strategy_exposure + 1e-9
    )


def build_exposure_summary(
    report: pd.DataFrame,
    maximum_total_exposure: float = 1.0,
    maximum_strategy_exposure: float = 0.50,
) -> dict[str, object]:
    """
    Build a structured portfolio exposure summary.
    """

    _validate_report(report)

    total = calculate_total_exposure(report)
    largest = calculate_largest_exposure(report)
    smallest_active = calculate_smallest_active_exposure(
        report
    )

    overexposed = find_overexposed_strategies(
        report,
        maximum_weight=maximum_strategy_exposure,
    )

    passed = validate_exposure(
        report,
        maximum_total_exposure=maximum_total_exposure,
        maximum_strategy_exposure=maximum_strategy_exposure,
    )

    return {
        "strategy_count": int(len(report)),
        "total_exposure": total,
        "largest_exposure": largest,
        "smallest_active_exposure": smallest_active,
        "overexposed_strategies": overexposed,
        "overexposed_count": int(len(overexposed)),
        "maximum_total_exposure": maximum_total_exposure,
        "maximum_strategy_exposure": maximum_strategy_exposure,
        "passed": bool(passed),
    }


def summarize_target_weights(
    target_weights: Mapping[str, float],
) -> dict[str, object]:
    """
    Build an exposure summary from a target-weight mapping.
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

    return build_exposure_summary(report)
