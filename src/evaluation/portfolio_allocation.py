from __future__ import annotations

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


def calculate_capital_allocation(
    report: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Convert portfolio weights into capital allocations.

    Parameters
    ----------
    report:
        DataFrame containing strategy names and portfolio weights.

    capital:
        Total capital available for allocation.
    """

    _validate_report(report)

    if capital < 0:
        raise ValueError(
            "capital must be non-negative."
        )

    result = report.copy()

    if "portfolio_weight" in result.columns:
        weights = pd.to_numeric(
            result["portfolio_weight"],
            errors="coerce",
        )

        if weights.isna().any():
            raise ValueError(
                "portfolio_weight contains invalid values."
            )

        if (weights < 0).any():
            raise ValueError(
                "portfolio_weight cannot be negative."
            )

        if abs(float(weights.sum()) - 1.0) > 1e-9:
            if float(weights.sum()) != 0.0:
                raise ValueError(
                    "portfolio_weight must sum to one."
                )

        result["allocated_capital"] = (
            weights * capital
        )

    return result


def calculate_exposure(
    report: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Calculate portfolio exposure for each strategy.

    Exposure is represented as allocated capital divided by total capital.
    """

    result = calculate_capital_allocation(
        report,
        capital,
    )

    if capital == 0:
        result["exposure"] = 0.0
    else:
        result["exposure"] = (
            result["allocated_capital"]
            / capital
        )

    return result


def calculate_unallocated_capital(
    report: pd.DataFrame,
    capital: float,
) -> float:
    """
    Return capital not allocated to strategies.
    """

    result = calculate_capital_allocation(
        report,
        capital,
    )

    if result.empty:
        return float(capital)

    allocated = float(
        result["allocated_capital"].sum()
    )

    return max(
        0.0,
        float(capital) - allocated,
    )


def validate_capital_allocation(
    report: pd.DataFrame,
    capital: float,
    tolerance: float = 1e-9,
) -> bool:
    """
    Validate that allocated capital is finite, non-negative,
    and does not exceed total capital.
    """

    if capital < 0:
        raise ValueError(
            "capital must be non-negative."
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    try:
        result = calculate_capital_allocation(
            report,
            capital,
        )
    except (TypeError, ValueError):
        return False

    if result.empty:
        return True

    allocated = result["allocated_capital"]

    if not pd.api.types.is_numeric_dtype(
        allocated
    ):
        return False

    if not allocated.notna().all():
        return False

    if (allocated < -tolerance).any():
        return False

    total_allocated = float(
        allocated.sum()
    )

    return (
        total_allocated
        <= capital + tolerance
    )


def build_capital_allocation(
    report: pd.DataFrame,
    capital: float,
) -> pd.DataFrame:
    """
    Build and validate the final capital allocation table.
    """

    result = calculate_exposure(
        report,
        capital,
    )

    if not validate_capital_allocation(
        result,
        capital,
    ):
        raise ValueError(
            "Capital allocation is invalid."
        )

    return result
