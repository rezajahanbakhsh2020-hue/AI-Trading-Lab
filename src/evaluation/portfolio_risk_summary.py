from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "portfolio_weight",
}


def _validate_input(report: pd.DataFrame) -> None:
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
    _validate_input(report)

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

    if (
        result["portfolio_weight"] < 0
    ).any():
        raise ValueError(
            "portfolio_weight cannot be negative."
        )

    return result


def calculate_weight_summary(
    report: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Calculate basic portfolio-weight statistics.
    """

    result = _prepare_weights(report)

    if result.empty:
        return {
            "strategy_count": 0,
            "weight_sum": 0.0,
            "max_weight": 0.0,
            "min_weight": 0.0,
            "average_weight": 0.0,
        }

    weights = result["portfolio_weight"]

    return {
        "strategy_count": int(len(result)),
        "weight_sum": float(weights.sum()),
        "max_weight": float(weights.max()),
        "min_weight": float(weights.min()),
        "average_weight": float(weights.mean()),
    }


def calculate_concentration(
    report: pd.DataFrame,
) -> float:
    """
    Calculate the Herfindahl-style concentration score.

    A higher value means the portfolio is more concentrated.
    """

    result = _prepare_weights(report)

    if result.empty:
        return 0.0

    weights = result["portfolio_weight"]

    return float(
        (weights**2).sum()
    )


def calculate_effective_strategy_count(
    report: pd.DataFrame,
) -> float:
    """
    Calculate the effective number of strategies.

    This is the inverse of the concentration score.
    """

    concentration = calculate_concentration(
        report
    )

    if concentration <= 0:
        return 0.0

    return float(1.0 / concentration)


def find_dominant_strategies(
    report: pd.DataFrame,
    threshold: float = 0.50,
) -> list[str]:
    """
    Return strategies whose individual weight reaches
    or exceeds the supplied threshold.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between zero and one."
        )

    result = _prepare_weights(report)

    dominant = result.loc[
        result["portfolio_weight"] >= threshold,
        "strategy",
    ]

    return dominant.tolist()


def validate_concentration_limit(
    report: pd.DataFrame,
    max_concentration: float = 0.25,
) -> bool:
    """
    Validate that portfolio concentration does not exceed
    the supplied squared-weight limit.
    """

    if not 0 <= max_concentration <= 1:
        raise ValueError(
            "max_concentration must be between zero and one."
        )

    concentration = calculate_concentration(
        report
    )

    return concentration <= max_concentration + 1e-9


def validate_max_strategy_weight(
    report: pd.DataFrame,
    max_weight: float = 0.50,
) -> bool:
    """
    Validate that no individual strategy exceeds max_weight.
    """

    if not 0 <= max_weight <= 1:
        raise ValueError(
            "max_weight must be between zero and one."
        )

    result = _prepare_weights(report)

    if result.empty:
        return True

    return bool(
        (
            result["portfolio_weight"]
            <= max_weight + 1e-9
        ).all()
    )


def build_risk_summary(
    report: pd.DataFrame,
    max_weight: float = 0.50,
    max_concentration: float = 0.25,
) -> dict[str, object]:
    """
    Build a structured portfolio concentration-risk summary.
    """

    stats = calculate_weight_summary(report)

    concentration = calculate_concentration(
        report
    )

    effective_count = (
        calculate_effective_strategy_count(
            report
        )
    )

    max_weight_valid = validate_max_strategy_weight(
        report,
        max_weight=max_weight,
    )

    concentration_valid = validate_concentration_limit(
        report,
        max_concentration=max_concentration,
    )

    return {
        **stats,
        "concentration": concentration,
        "effective_strategy_count": effective_count,
        "max_strategy_weight_valid": max_weight_valid,
        "concentration_limit_valid": concentration_valid,
        "passed": bool(
            max_weight_valid
            and concentration_valid
        ),
    }


def summarize_target_weights(
    target_weights: Mapping[str, float],
) -> dict[str, object]:
    """
    Build a risk summary directly from a strategy-weight mapping.
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
                "strategy": str(strategy).strip(),
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

    return build_risk_summary(report)
