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


def calculate_total_exposure(
    report: pd.DataFrame,
) -> float:
    """
    Return total portfolio exposure.
    """

    result = _prepare_report(report)

    return float(
        result["portfolio_weight"].sum()
    )


def calculate_cash_weight(
    report: pd.DataFrame,
) -> float:
    """
    Return the portfolio weight held as cash.
    """

    exposure = calculate_total_exposure(
        report
    )

    return max(
        0.0,
        1.0 - exposure,
    )


def calculate_max_strategy_weight(
    report: pd.DataFrame,
) -> float:
    """
    Return the largest individual strategy weight.
    """

    result = _prepare_report(report)

    if result.empty:
        return 0.0

    return float(
        result["portfolio_weight"].max()
    )


def calculate_weight_concentration(
    report: pd.DataFrame,
) -> float:
    """
    Calculate Herfindahl-Hirschman concentration
    from portfolio weights.

    A higher value means greater concentration.
    """

    result = _prepare_report(report)

    if result.empty:
        return 0.0

    weights = result["portfolio_weight"]

    return float(
        (weights ** 2).sum()
    )


def calculate_effective_strategy_count(
    report: pd.DataFrame,
) -> float:
    """
    Calculate the effective number of strategies.

    This is the inverse of the weight concentration.
    """

    concentration = calculate_weight_concentration(
        report
    )

    if concentration <= 0:
        return 0.0

    return float(
        1.0 / concentration
    )


def check_portfolio_risk_limits(
    report: pd.DataFrame,
    max_strategy_weight: float = 0.60,
    max_concentration: float = 0.50,
) -> dict[str, object]:
    """
    Check portfolio concentration and exposure limits.
    """

    if not 0 <= max_strategy_weight <= 1:
        raise ValueError(
            "max_strategy_weight must be between zero and one."
        )

    if not 0 <= max_concentration <= 1:
        raise ValueError(
            "max_concentration must be between zero and one."
        )

    result = _prepare_report(report)

    total_exposure = float(
        result["portfolio_weight"].sum()
    )

    max_weight = (
        0.0
        if result.empty
        else float(
            result["portfolio_weight"].max()
        )
    )

    concentration = calculate_weight_concentration(
        result
    )

    max_weight_ok = (
        max_weight
        <= max_strategy_weight + 1e-9
    )

    concentration_ok = (
        concentration
        <= max_concentration + 1e-9
    )

    exposure_ok = (
        total_exposure <= 1.0 + 1e-9
    )

    return {
        "passed": bool(
            max_weight_ok
            and concentration_ok
            and exposure_ok
        ),
        "total_exposure": total_exposure,
        "cash_weight": max(
            0.0,
            1.0 - total_exposure,
        ),
        "max_strategy_weight": max_weight,
        "concentration": concentration,
        "max_weight_ok": max_weight_ok,
        "concentration_ok": concentration_ok,
        "exposure_ok": exposure_ok,
    }


def build_portfolio_risk_report(
    report: pd.DataFrame,
    max_strategy_weight: float = 0.60,
    max_concentration: float = 0.50,
) -> pd.DataFrame:
    """
    Add portfolio-level risk monitoring metrics to
    each strategy row.
    """

    result = _prepare_report(report)

    total_exposure = float(
        result["portfolio_weight"].sum()
    )

    cash_weight = max(
        0.0,
        1.0 - total_exposure,
    )

    concentration = calculate_weight_concentration(
        result
    )

    effective_count = (
        0.0
        if concentration <= 0
        else 1.0 / concentration
    )

    max_weight = (
        0.0
        if result.empty
        else float(
            result["portfolio_weight"].max()
        )
    )

    result["total_exposure"] = total_exposure
    result["cash_weight"] = cash_weight
    result["concentration"] = concentration
    result["effective_strategy_count"] = effective_count
    result["max_strategy_weight"] = max_weight

    result["risk_limits_passed"] = (
        (max_weight <= max_strategy_weight + 1e-9)
        & (concentration <= max_concentration + 1e-9)
    )

    return result
