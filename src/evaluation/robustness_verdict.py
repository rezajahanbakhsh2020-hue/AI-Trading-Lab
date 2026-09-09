"""Robustness verdict utilities for strategy validation.

This module converts transaction-cost sensitivity results into a clear,
deterministic robustness verdict.

It does not rerun backtests. It consumes already-computed sensitivity
results and checks whether the strategy remains acceptable under the
tested stress scenarios.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import pandas as pd


def build_robustness_verdict(
    sensitivity_results: pd.DataFrame,
    *,
    return_column: str = "total_return",
    sharpe_column: str = "sharpe_ratio",
    minimum_positive_return_rate: float = 1.0,
    minimum_worst_return: float = 0.0,
    minimum_worst_sharpe: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic robustness verdict from sensitivity results.

    A strategy passes the robustness check when:

    1. sensitivity_results is non-empty;
    2. all tested total returns are finite;
    3. the positive-return rate meets the requested threshold;
    4. the worst total return meets the requested minimum;
    5. when requested, the worst Sharpe ratio meets its minimum.

    Returns a dictionary containing the individual checks and final verdict.
    """
    if not isinstance(sensitivity_results, pd.DataFrame):
        raise TypeError("sensitivity_results must be a pandas DataFrame")

    if sensitivity_results.empty:
        raise ValueError("sensitivity_results must not be empty")

    if return_column not in sensitivity_results.columns:
        raise ValueError(
            f"missing required return column: {return_column}"
        )

    if not 0.0 <= minimum_positive_return_rate <= 1.0:
        raise ValueError(
            "minimum_positive_return_rate must be between 0 and 1"
        )

    if not isfinite(float(minimum_worst_return)):
        raise ValueError("minimum_worst_return must be finite")

    if minimum_worst_sharpe is not None and not isfinite(
        float(minimum_worst_sharpe)
    ):
        raise ValueError("minimum_worst_sharpe must be finite")

    returns = pd.to_numeric(
        sensitivity_results[return_column],
        errors="coerce",
    )

    if returns.isna().any() or not returns.map(isfinite).all():
        raise ValueError("total returns must contain only finite values")

    positive_return_rate = float((returns > 0.0).mean())
    worst_return = float(returns.min())

    positive_return_check = (
        positive_return_rate >= minimum_positive_return_rate
    )
    worst_return_check = worst_return >= float(minimum_worst_return)

    checks: dict[str, bool] = {
        "positive_return_rate": positive_return_check,
        "worst_total_return": worst_return_check,
    }

    worst_sharpe: float | None = None

    if minimum_worst_sharpe is not None:
        if sharpe_column not in sensitivity_results.columns:
            raise ValueError(
                f"missing required Sharpe column: {sharpe_column}"
            )

        sharpe = pd.to_numeric(
            sensitivity_results[sharpe_column],
            errors="coerce",
        )

        if sharpe.isna().any() or not sharpe.map(isfinite).all():
            raise ValueError("Sharpe values must contain only finite values")

        worst_sharpe = float(sharpe.min())
        checks["worst_sharpe_ratio"] = (
            worst_sharpe >= float(minimum_worst_sharpe)
        )

    robust = all(checks.values())

    verdict: dict[str, Any] = {
        "scenarios": int(len(sensitivity_results)),
        "positive_return_rate": positive_return_rate,
        "worst_total_return": worst_return,
        "minimum_positive_return_rate": float(
            minimum_positive_return_rate
        ),
        "minimum_worst_return": float(minimum_worst_return),
        "checks": checks,
        "robust": robust,
        "verdict": "ROBUST" if robust else "NOT_ROBUST",
    }

    if worst_sharpe is not None:
        verdict["worst_sharpe_ratio"] = worst_sharpe
        verdict["minimum_worst_sharpe"] = float(minimum_worst_sharpe)

    return verdict


def summarize_robustness_verdict(
    verdict: Mapping[str, Any],
) -> str:
    """Return a concise human-readable robustness summary."""
    if not isinstance(verdict, Mapping):
        raise TypeError("verdict must be a mapping")

    if "robust" not in verdict or "verdict" not in verdict:
        raise ValueError("verdict must contain robust and verdict fields")

    status = "PASS" if bool(verdict["robust"]) else "FAIL"
    label = str(verdict["verdict"])

    scenarios = verdict.get("scenarios", 0)
    positive_rate = verdict.get("positive_return_rate", 0.0)
    worst_return = verdict.get("worst_total_return", 0.0)

    return (
        f"{status}: {label}; "
        f"scenarios={scenarios}; "
        f"positive_return_rate={float(positive_rate):.6f}; "
        f"worst_total_return={float(worst_return):.6f}"
    )
