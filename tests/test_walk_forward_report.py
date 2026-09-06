import pandas as pd

from src.evaluation.metrics import (
    calmar_ratio,
    exposure,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)


REQUIRED_COLUMNS = {
    "timestamp",
    "strategy_return",
    "equity",
}


def combine_oos_results(
    oos_results: list[pd.DataFrame],
) -> pd.DataFrame:
    """Combine out-of-sample results from multiple walk-forward windows."""

    if not isinstance(oos_results, list):
        raise TypeError("oos_results must be a list")

    for result in oos_results:
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                "every OOS result must be a pandas DataFrame"
            )

    if not oos_results:
        return pd.DataFrame()

    for result in oos_results:
        missing = REQUIRED_COLUMNS - set(result.columns)

        if missing:
            raise ValueError(
                f"OOS result is missing required columns: {sorted(missing)}"
            )

    combined = pd.concat(
        oos_results,
        ignore_index=True,
    )

    if "timestamp" in combined.columns:
        combined = combined.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    return combined


def evaluate_walk_forward(
    oos_results: list[pd.DataFrame],
) -> dict:
    """Evaluate combined out-of-sample walk-forward results."""

    if not isinstance(oos_results, list):
        raise TypeError("oos_results must be a list")

    if not oos_results:
        return {
            "windows": 0,
            "observations": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "exposure": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }

    combined = combine_oos_results(oos_results)

    exposure_column = None

    if "position" in combined.columns:
        exposure_column = "position"
    elif "signal" in combined.columns:
        exposure_column = "signal"
    else:
        raise ValueError(
            "OOS results must contain either 'position' or 'signal'"
        )

    return {
        "windows": len(oos_results),
        "observations": len(combined),
        "total_return": total_return(combined),
        "max_drawdown": max_drawdown(combined),
        "sharpe_ratio": sharpe_ratio(combined),
        "calmar_ratio": calmar_ratio(combined),
        "sortino_ratio": sortino_ratio(combined),
        "exposure": exposure(
            combined,
            signal_column=exposure_column,
        ),
        "win_rate": win_rate(combined),
        "profit_factor": profit_factor(combined),
    }
