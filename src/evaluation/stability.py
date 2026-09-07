from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {
    "strategy",
    "total_return",
    "max_drawdown",
    "sharpe_ratio",
    "positive_window_rate",
}


def calculate_stability_score(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a cross-experiment stability score.

    Higher score means more consistent performance
    across experiments.
    """
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError(
            "comparison must be a pandas DataFrame."
        )

    missing = REQUIRED_COLUMNS - set(
        comparison.columns
    )

    if missing:
        raise ValueError(
            "comparison is missing required columns: "
            f"{sorted(missing)}"
        )

    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "experiments",
                "avg_total_return",
                "avg_sharpe_ratio",
                "avg_positive_window_rate",
                "avg_drawdown",
                "return_consistency",
                "sharpe_consistency",
                "stability_score",
            ]
        )

    rows = []

    for strategy, group in comparison.groupby(
        "strategy"
    ):
        returns = pd.to_numeric(
            group["total_return"],
            errors="coerce",
        ).dropna()

        sharpes = pd.to_numeric(
            group["sharpe_ratio"],
            errors="coerce",
        ).dropna()

        positive_rates = pd.to_numeric(
            group["positive_window_rate"],
            errors="coerce",
        ).dropna()

        drawdowns = pd.to_numeric(
            group["max_drawdown"],
            errors="coerce",
        ).dropna()

        avg_return = (
            float(returns.mean())
            if not returns.empty
            else 0.0
        )

        avg_sharpe = (
            float(sharpes.mean())
            if not sharpes.empty
            else 0.0
        )

        avg_positive_rate = (
            float(positive_rates.mean())
            if not positive_rates.empty
            else 0.0
        )

        avg_drawdown = (
            float(drawdowns.abs().mean())
            if not drawdowns.empty
            else 0.0
        )

        return_consistency = (
            max(
                0.0,
                1.0 - float(returns.std(ddof=0)),
            )
            if len(returns) > 1
            else 1.0
        )

        sharpe_consistency = (
            max(
                0.0,
                1.0 - float(sharpes.std(ddof=0)),
            )
            if len(sharpes) > 1
            else 1.0
        )

        stability_score = (
            0.30 * max(0.0, avg_return)
            + 0.20 * max(0.0, avg_sharpe)
            + 0.20 * avg_positive_rate
            + 0.15 * return_consistency
            + 0.10 * sharpe_consistency
            + 0.05 * max(
                0.0,
                1.0 - avg_drawdown,
            )
        )

        rows.append(
            {
                "strategy": strategy,
                "experiments": len(group),
                "avg_total_return": avg_return,
                "avg_sharpe_ratio": avg_sharpe,
                "avg_positive_window_rate": (
                    avg_positive_rate
                ),
                "avg_drawdown": avg_drawdown,
                "return_consistency": (
                    return_consistency
                ),
                "sharpe_consistency": (
                    sharpe_consistency
                ),
                "stability_score": (
                    stability_score
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "stability_score",
                "avg_total_return",
                "avg_sharpe_ratio",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def get_stable_strategy(
    comparison: pd.DataFrame,
) -> str | None:
    """
    Return the strategy with the highest stability score.
    """
    ranking = calculate_stability_score(
        comparison
    )

    if ranking.empty:
        return None

    return str(
        ranking.iloc[0]["strategy"]
    )
