from __future__ import annotations

from math import isfinite

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "strategy_return",
    "equity",
}


def _validate_results(
    results: list[pd.DataFrame],
) -> None:
    if not isinstance(results, list):
        raise TypeError("results must be a list.")

    for result in results:
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "Every walk-forward result must be a pandas DataFrame."
            )

        missing = REQUIRED_COLUMNS.difference(result.columns)
        if missing:
            raise ValueError(
                "Walk-forward result is missing required columns: "
                f"{sorted(missing)}"
            )


def _window_returns(
    results: list[pd.DataFrame],
) -> list[float]:
    returns: list[float] = []

    for result in results:
        if result.empty:
            returns.append(0.0)
            continue

        values = pd.to_numeric(
            result["strategy_return"],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                "strategy_return contains invalid numeric values."
            )

        window_return = float(
            (1.0 + values).prod() - 1.0
        )

        if not isfinite(window_return):
            raise ValueError(
                "Walk-forward window return is not finite."
            )

        returns.append(window_return)

    return returns


def _max_consecutive_losses(
    returns: list[float],
) -> int:
    current = 0
    maximum = 0

    for value in returns:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0

    return maximum


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(sum(values) / len(values))


def calculate_walk_forward_stability(
    results: list[pd.DataFrame],
) -> dict:
    """
    Analyze consistency of out-of-sample walk-forward performance.

    The analysis focuses on the distribution of OOS window returns rather
    than only the aggregate return. This helps identify strategies whose
    total performance depends on a small number of unusually strong windows.
    """

    _validate_results(results)

    if not results:
        return {
            "windows": 0,
            "window_returns": [],
            "mean_window_return": 0.0,
            "median_window_return": 0.0,
            "std_window_return": 0.0,
            "best_window_return": 0.0,
            "worst_window_return": 0.0,
            "positive_window_rate": 0.0,
            "negative_window_rate": 0.0,
            "max_consecutive_losses": 0,
            "return_dispersion": 0.0,
            "stability_score": 0.0,
        }

    returns = _window_returns(results)

    series = pd.Series(returns, dtype="float64")

    mean_return = float(series.mean())
    median_return = float(series.median())
    std_return = float(series.std(ddof=0))

    positive_windows = int((series > 0).sum())
    negative_windows = int((series < 0).sum())

    window_count = len(returns)

    positive_rate = (
        positive_windows / window_count
        if window_count
        else 0.0
    )

    negative_rate = (
        negative_windows / window_count
        if window_count
        else 0.0
    )

    worst_return = float(series.min())
    best_return = float(series.max())

    return_dispersion = (
        std_return / max(abs(mean_return), 1e-12)
        if window_count > 1
        else 0.0
    )

    consistency = 1.0 / (1.0 + return_dispersion)

    worst_window_score = (
        1.0 / (1.0 + abs(worst_return))
    )

    median_score = (
        1.0
        if median_return > 0
        else 0.0
    )

    stability_score = (
        0.40 * positive_rate
        + 0.25 * consistency
        + 0.20 * worst_window_score
        + 0.15 * median_score
    )

    return {
        "windows": window_count,
        "window_returns": returns,
        "mean_window_return": mean_return,
        "median_window_return": median_return,
        "std_window_return": std_return,
        "best_window_return": best_return,
        "worst_window_return": worst_return,
        "positive_window_rate": positive_rate,
        "negative_window_rate": negative_rate,
        "max_consecutive_losses": _max_consecutive_losses(
            returns
        ),
        "return_dispersion": return_dispersion,
        "stability_score": float(stability_score),
    }


def rank_walk_forward_stability(
    strategies: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    """
    Rank multiple strategies by walk-forward OOS stability.

    Parameters
    ----------
    strategies:
        Mapping of strategy name to its list of OOS walk-forward results.
    """

    if not isinstance(strategies, dict):
        raise TypeError(
            "strategies must be a dictionary."
        )

    rows = []

    for strategy, results in strategies.items():
        report = calculate_walk_forward_stability(
            results
        )

        rows.append(
            {
                "strategy": strategy,
                **report,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "strategy",
                "windows",
                "window_returns",
                "mean_window_return",
                "median_window_return",
                "std_window_return",
                "best_window_return",
                "worst_window_return",
                "positive_window_rate",
                "negative_window_rate",
                "max_consecutive_losses",
                "return_dispersion",
                "stability_score",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "stability_score",
                "positive_window_rate",
                "median_window_return",
                "worst_window_return",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


def get_most_stable_walk_forward_strategy(
    strategies: dict[str, list[pd.DataFrame]],
) -> str | None:
    """
    Return the strategy with the strongest OOS stability.
    """

    ranking = rank_walk_forward_stability(
        strategies
    )

    if ranking.empty:
        return None

    return str(ranking.iloc[0]["strategy"])
