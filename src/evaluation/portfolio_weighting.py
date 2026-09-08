from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "portfolio_score",
    "stability_score",
    "sharpe_ratio",
    "max_drawdown",
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


def calculate_portfolio_weights(
    report: pd.DataFrame,
    max_weight: float = 0.60,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate normalized portfolio weights.

    Weights are based on positive portfolio scores and are capped by
    max_weight. Remaining capital is redistributed until either all
    capital is allocated or every strategy reaches the cap.
    """

    _validate_report(report)

    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(
            "min_weight must be between 0 and 1."
        )

    if not 0.0 < max_weight <= 1.0:
        raise ValueError(
            "max_weight must be greater than 0 and at most 1."
        )

    if min_weight > max_weight:
        raise ValueError(
            "min_weight cannot exceed max_weight."
        )

    result = report.copy()

    result["portfolio_weight"] = 0.0

    if result.empty:
        return result

    scores = pd.to_numeric(
        result["portfolio_score"],
        errors="coerce",
    )

    if scores.isna().any():
        raise ValueError(
            "portfolio_score contains invalid numeric values."
        )

    positive_scores = scores.clip(lower=0.0)

    if positive_scores.sum() <= 0:
        return result

    weights = pd.Series(
        0.0,
        index=result.index,
        dtype="float64",
    )

    active = positive_scores > 0
    remaining = 1.0

    while active.any() and remaining > 1e-12:
        active_scores = positive_scores[active]
        score_sum = float(active_scores.sum())

        if score_sum <= 0:
            break

        proposed = (
            active_scores / score_sum
        ) * remaining

        capped = proposed > max_weight

        if not capped.any():
            weights.loc[active] = (
                weights.loc[active] + proposed
            )
            remaining = 0.0
            break

        capped_indices = proposed[capped].index

        weights.loc[capped_indices] = max_weight

        remaining -= (
            max_weight * len(capped_indices)
        )

        active.loc[capped_indices] = False

        if remaining <= 1e-12:
            break

    if remaining > 1e-12:
        eligible = weights < max_weight - 1e-12

        if eligible.any():
            eligible_scores = positive_scores[eligible]
            score_sum = float(eligible_scores.sum())

            if score_sum > 0:
                addition = (
                    eligible_scores / score_sum
                ) * remaining

                for index in addition.index:
                    capacity = (
                        max_weight
                        - weights.loc[index]
                    )

                    amount = min(
                        float(addition.loc[index]),
                        capacity,
                    )

                    weights.loc[index] += amount

    if min_weight > 0:
        positive_indices = positive_scores[
            positive_scores > 0
        ].index

        if len(positive_indices) > 0:
            for index in positive_indices:
                if weights.loc[index] < min_weight:
                    weights.loc[index] = min_weight

            total = float(weights.sum())

            if total > 1.0:
                excess = total - 1.0

                adjustable = weights > min_weight

                while (
                    excess > 1e-12
                    and adjustable.any()
                ):
                    adjustable_weights = weights[
                        adjustable
                    ]

                    reduction_capacity = (
                        adjustable_weights
                        - min_weight
                    )

                    capacity_sum = float(
                        reduction_capacity.sum()
                    )

                    if capacity_sum <= 0:
                        break

                    reduction = (
                        reduction_capacity
                        / capacity_sum
                    ) * excess

                    reduction = reduction.clip(
                        upper=reduction_capacity
                    )

                    weights.loc[
                        reduction.index
                    ] -= reduction

                    excess = max(
                        0.0,
                        1.0 - float(weights.sum()),
                    )

                    if excess <= 1e-12:
                        break

                    adjustable = (
                        weights > min_weight
                        + 1e-12
                    )

    total_weight = float(weights.sum())

    if total_weight > 0:
        weights /= total_weight

    result["portfolio_weight"] = weights

    return result


def validate_portfolio_weights(
    report: pd.DataFrame,
    tolerance: float = 1e-9,
) -> bool:
    """
    Validate that portfolio weights are finite, non-negative,
    and sum to one when capital is allocated.
    """

    if not isinstance(report, pd.DataFrame):
        raise TypeError(
            "report must be a pandas DataFrame."
        )

    if "portfolio_weight" not in report.columns:
        raise ValueError(
            "Missing required column: portfolio_weight"
        )

    if tolerance < 0:
        raise ValueError(
            "tolerance must be non-negative."
        )

    if report.empty:
        return True

    weights = pd.to_numeric(
        report["portfolio_weight"],
        errors="coerce",
    )

    if weights.isna().any():
        return False

    if (weights < -tolerance).any():
        return False

    total = float(weights.sum())

    if total == 0:
        return True

    return abs(total - 1.0) <= tolerance


def allocate_portfolio(
    report: pd.DataFrame,
    max_weight: float = 0.60,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate and validate portfolio weights.
    """

    result = calculate_portfolio_weights(
        report,
        max_weight=max_weight,
        min_weight=min_weight,
    )

    if not validate_portfolio_weights(result):
        raise ValueError(
            "Calculated portfolio weights are invalid."
        )

    return result
