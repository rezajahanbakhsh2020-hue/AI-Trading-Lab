from __future__ import annotations

import math

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


def calculate_weight_balance_score(
    report: pd.DataFrame,
) -> float:
    """
    Calculate how closely active weights resemble equal weighting.

    Returns a value between zero and one.
    Higher values indicate a more balanced allocation.
    """

    result = _prepare_report(report)

    if result.empty:
        return 0.0

    positive = result.loc[
        result["portfolio_weight"] > 0,
        "portfolio_weight",
    ]

    if positive.empty:
        return 0.0

    total = float(positive.sum())

    if total <= 0:
        return 0.0

    normalized = positive / total
    equal_weight = 1.0 / len(normalized)

    deviation = float(
        (normalized - equal_weight)
        .abs()
        .mean()
    )

    maximum_deviation = (
        2.0
        * equal_weight
        * (1.0 - equal_weight)
    )

    if maximum_deviation <= 0:
        return 1.0

    score = 1.0 - (
        deviation / maximum_deviation
    )

    return float(
        max(0.0, min(1.0, score))
    )


def calculate_allocation_entropy(
    report: pd.DataFrame,
) -> float:
    """
    Calculate Shannon entropy of the active allocation.

    The value is normalized to the range zero to one.
    """

    result = _prepare_report(report)

    positive = result.loc[
        result["portfolio_weight"] > 0,
        "portfolio_weight",
    ]

    if len(positive) <= 1:
        return 0.0

    total = float(positive.sum())

    if total <= 0:
        return 0.0

    probabilities = positive / total

    entropy = float(
        -sum(
            weight * math.log(weight)
            for weight in probabilities
        )
    )

    maximum_entropy = math.log(
        len(probabilities)
    )

    if maximum_entropy <= 0:
        return 0.0

    return float(
        entropy / maximum_entropy
    )


def calculate_zero_weight_ratio(
    report: pd.DataFrame,
) -> float:
    """
    Return the fraction of strategies receiving zero weight.
    """

    result = _prepare_report(report)

    if result.empty:
        return 0.0

    return float(
        (
            result["portfolio_weight"] == 0
        ).mean()
    )


def calculate_active_weight_ratio(
    report: pd.DataFrame,
) -> float:
    """
    Return the fraction of strategies receiving positive weight.
    """

    return float(
        1.0 - calculate_zero_weight_ratio(report)
    )


def calculate_top_weight_share(
    report: pd.DataFrame,
    top_n: int = 1,
) -> float:
    """
    Calculate the share of total allocation held by the
    top N strategies.
    """

    if top_n < 1:
        raise ValueError(
            "top_n must be at least one."
        )

    result = _prepare_report(report)

    if result.empty:
        return 0.0

    total = float(
        result["portfolio_weight"].sum()
    )

    if total <= 0:
        return 0.0

    top_weights = (
        result["portfolio_weight"]
        .sort_values(ascending=False)
        .head(top_n)
    )

    return float(
        top_weights.sum() / total
    )


def validate_allocation_quality(
    report: pd.DataFrame,
    minimum_balance_score: float = 0.50,
    maximum_top_weight_share: float = 0.75,
) -> bool:
    """
    Validate basic allocation-quality constraints.
    """

    if not 0 <= minimum_balance_score <= 1:
        raise ValueError(
            "minimum_balance_score must be between zero and one."
        )

    if not 0 <= maximum_top_weight_share <= 1:
        raise ValueError(
            "maximum_top_weight_share must be between zero and one."
        )

    balance_score = calculate_weight_balance_score(
        report
    )

    top_weight_share = calculate_top_weight_share(
        report
    )

    return bool(
        balance_score >= minimum_balance_score
        and top_weight_share
        <= maximum_top_weight_share + 1e-9
    )


def build_allocation_quality_summary(
    report: pd.DataFrame,
    minimum_balance_score: float = 0.50,
    maximum_top_weight_share: float = 0.75,
) -> dict[str, object]:
    """
    Build a structured allocation-quality summary.
    """

    _validate_report(report)

    balance_score = calculate_weight_balance_score(
        report
    )

    entropy = calculate_allocation_entropy(
        report
    )

    zero_weight_ratio = calculate_zero_weight_ratio(
        report
    )

    active_weight_ratio = calculate_active_weight_ratio(
        report
    )

    top_weight_share = calculate_top_weight_share(
        report
    )

    passed = validate_allocation_quality(
        report,
        minimum_balance_score=minimum_balance_score,
        maximum_top_weight_share=maximum_top_weight_share,
    )

    return {
        "strategy_count": int(len(report)),
        "balance_score": balance_score,
        "allocation_entropy": entropy,
        "zero_weight_ratio": zero_weight_ratio,
        "active_weight_ratio": active_weight_ratio,
        "top_weight_share": top_weight_share,
        "minimum_balance_score": minimum_balance_score,
        "maximum_top_weight_share": (
            maximum_top_weight_share
        ),
        "passed": bool(passed),
    }
