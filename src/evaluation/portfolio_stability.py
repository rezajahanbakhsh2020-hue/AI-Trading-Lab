from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {"strategy", "portfolio_weight"}


def _validate_report(report: pd.DataFrame) -> None:
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be a pandas DataFrame")

    missing = REQUIRED_COLUMNS - set(report.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )


def _prepare_report(report: pd.DataFrame) -> pd.DataFrame:
    _validate_report(report)

    prepared = report.copy()
    prepared["portfolio_weight"] = pd.to_numeric(
        prepared["portfolio_weight"], errors="coerce"
    )

    if prepared["portfolio_weight"].isna().any():
        raise ValueError("portfolio_weight contains invalid values")

    return prepared


def calculate_max_weight_change(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> float:
    current = _prepare_report(current)
    previous = _prepare_report(previous)

    merged = current[["strategy", "portfolio_weight"]].merge(
        previous[["strategy", "portfolio_weight"]],
        on="strategy",
        how="outer",
        suffixes=("_current", "_previous"),
    ).fillna(0.0)

    if merged.empty:
        return 0.0

    changes = (
        merged["portfolio_weight_current"]
        - merged["portfolio_weight_previous"]
    ).abs()

    return float(changes.max())


def calculate_mean_weight_change(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> float:
    current = _prepare_report(current)
    previous = _prepare_report(previous)

    merged = current[["strategy", "portfolio_weight"]].merge(
        previous[["strategy", "portfolio_weight"]],
        on="strategy",
        how="outer",
        suffixes=("_current", "_previous"),
    ).fillna(0.0)

    if merged.empty:
        return 0.0

    changes = (
        merged["portfolio_weight_current"]
        - merged["portfolio_weight_previous"]
    ).abs()

    return float(changes.mean())


def count_changed_strategies(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    tolerance: float = 1e-12,
) -> int:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    current = _prepare_report(current)
    previous = _prepare_report(previous)

    merged = current[["strategy", "portfolio_weight"]].merge(
        previous[["strategy", "portfolio_weight"]],
        on="strategy",
        how="outer",
        suffixes=("_current", "_previous"),
    ).fillna(0.0)

    if merged.empty:
        return 0

    changes = (
        merged["portfolio_weight_current"]
        - merged["portfolio_weight_previous"]
    ).abs()

    return int((changes > tolerance).sum())


def find_changed_strategies(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    tolerance: float = 1e-12,
) -> list[str]:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    current = _prepare_report(current)
    previous = _prepare_report(previous)

    merged = current[["strategy", "portfolio_weight"]].merge(
        previous[["strategy", "portfolio_weight"]],
        on="strategy",
        how="outer",
        suffixes=("_current", "_previous"),
    ).fillna(0.0)

    if merged.empty:
        return []

    changes = (
        merged["portfolio_weight_current"]
        - merged["portfolio_weight_previous"]
    ).abs()

    return (
        merged.loc[changes > tolerance, "strategy"]
        .astype(str)
        .tolist()
    )


def calculate_stability_score(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> float:
    max_change = calculate_max_weight_change(current, previous)

    score = 1.0 - max_change
    return float(max(0.0, min(1.0, score)))


def validate_stability(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    max_allowed_change: float,
) -> bool:
    if max_allowed_change < 0:
        raise ValueError("max_allowed_change must be non-negative")

    max_change = calculate_max_weight_change(current, previous)

    return bool(max_change <= max_allowed_change)


def build_stability_summary(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> dict:
    changed = find_changed_strategies(current, previous)

    return {
        "max_weight_change": calculate_max_weight_change(
            current, previous
        ),
        "mean_weight_change": calculate_mean_weight_change(
            current, previous
        ),
        "changed_strategy_count": len(changed),
        "changed_strategies": changed,
        "stability_score": calculate_stability_score(
            current, previous
        ),
    }


def summarize_allocation_stability(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> dict:
    return build_stability_summary(current, previous)
