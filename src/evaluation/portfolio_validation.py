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
    """
    Prepare and validate basic portfolio weight values.

    Duplicate strategy names and total weight are validated by
    dedicated public validation functions.
    """

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

    return result


def validate_strategy_names(
    report: pd.DataFrame,
) -> bool:
    """
    Validate that strategy names are non-empty and unique.
    """

    _validate_report(report)

    strategies = report["strategy"]

    if strategies.isna().any():
        return False

    normalized = strategies.astype(str).str.strip()

    if (normalized == "").any():
        return False

    return not normalized.duplicated().any()


def validate_weight_sum(
    report: pd.DataFrame,
    require_full_allocation: bool = False,
    tolerance: float = 1e-9,
    raise_on_excess: bool = True,
) -> bool:
    """
    Validate the portfolio weight total.

    By default, a total above one raises ValueError.
    Structured validation callers can set raise_on_excess=False
    to receive False instead.
    """

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    result = _prepare_report(report)

    total = float(
        result["portfolio_weight"].sum()
    )

    if total > 1.0 + tolerance:
        if raise_on_excess:
            raise ValueError(
                "portfolio_weight cannot sum to more than one."
            )
        return False

    if require_full_allocation:
        return abs(total - 1.0) <= tolerance

    return True


def validate_weight_bounds(
    report: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> bool:
    """
    Validate that every portfolio weight is within bounds.
    """

    if min_weight < 0:
        raise ValueError(
            "min_weight cannot be negative."
        )

    if max_weight > 1:
        raise ValueError(
            "max_weight cannot exceed one."
        )

    if min_weight > max_weight:
        raise ValueError(
            "min_weight cannot exceed max_weight."
        )

    result = _prepare_report(report)

    if result.empty:
        return True

    weights = result["portfolio_weight"]

    return bool(
        (
            (weights >= min_weight - 1e-9)
            & (weights <= max_weight + 1e-9)
        ).all()
    )


def validate_no_nan_values(
    report: pd.DataFrame,
) -> bool:
    """
    Validate that portfolio weights contain no missing values.
    """

    _validate_report(report)

    weights = pd.to_numeric(
        report["portfolio_weight"],
        errors="coerce",
    )

    return bool(
        weights.notna().all()
    )


def validate_portfolio_report(
    report: pd.DataFrame,
    require_full_allocation: bool = False,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> dict[str, object]:
    """
    Run all portfolio validation checks and return a
    structured validation result.
    """

    _validate_report(report)

    names_valid = validate_strategy_names(
        report
    )

    weights_valid = validate_no_nan_values(
        report
    )

    if weights_valid:
        bounds_valid = validate_weight_bounds(
            report,
            min_weight=min_weight,
            max_weight=max_weight,
        )

        weight_sum_valid = validate_weight_sum(
            report,
            require_full_allocation=require_full_allocation,
            raise_on_excess=False,
        )
    else:
        bounds_valid = False
        weight_sum_valid = False

    passed = (
        names_valid
        and weights_valid
        and bounds_valid
        and weight_sum_valid
    )

    weight_sum = None

    if weights_valid:
        weight_sum = float(
            pd.to_numeric(
                report["portfolio_weight"],
                errors="coerce",
            ).sum()
        )

    return {
        "passed": bool(passed),
        "strategy_names_valid": names_valid,
        "weights_valid": weights_valid,
        "weight_bounds_valid": bounds_valid,
        "weight_sum_valid": weight_sum_valid,
        "strategy_count": int(len(report)),
        "weight_sum": weight_sum,
    }


def assert_valid_portfolio(
    report: pd.DataFrame,
    require_full_allocation: bool = False,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> None:
    """
    Raise ValueError when the portfolio report is invalid.
    """

    result = validate_portfolio_report(
        report,
        require_full_allocation=require_full_allocation,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    if not result["passed"]:
        failed_checks = [
            key
            for key, value in result.items()
            if key.endswith("_valid")
            and value is False
        ]

        raise ValueError(
            "Invalid portfolio report. Failed checks: "
            f"{failed_checks}"
        )


def validate_target_weights(
    target_weights: Mapping[str, float],
    require_full_allocation: bool = False,
) -> bool:
    """
    Validate a target-weight mapping before rebalancing.
    """

    if not isinstance(target_weights, Mapping):
        raise TypeError(
            "target_weights must be a mapping."
        )

    values: list[float] = []

    for strategy, weight in target_weights.items():
        if strategy is None or not str(strategy).strip():
            return False

        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            return False

        if pd.isna(numeric_weight) or numeric_weight < 0:
            return False

        values.append(numeric_weight)

    total = sum(values)

    if total > 1.0 + 1e-9:
        return False

    if require_full_allocation:
        return abs(total - 1.0) <= 1e-9

    return True
