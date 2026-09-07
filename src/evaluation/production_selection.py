from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.stable_strategy_workflow import (
    run_stable_strategy_selection,
)


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def select_production_strategy(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict:
    """
    Select the strategy recommended for production
    based on cross-experiment stability.
    """
    result = run_stable_strategy_selection(
        results_dir=results_dir
    )

    stability_report = result[
        "stability_report"
    ]

    stable_strategy = result[
        "stable_strategy"
    ]

    if stability_report.empty:
        return {
            "strategy": None,
            "stability_score": None,
            "stability_report": stability_report,
        }

    if stable_strategy is None:
        return {
            "strategy": None,
            "stability_score": None,
            "stability_report": stability_report,
        }

    selected = stability_report[
        stability_report["strategy"]
        == stable_strategy
    ]

    if selected.empty:
        return {
            "strategy": None,
            "stability_score": None,
            "stability_report": stability_report,
        }

    score = float(
        selected.iloc[0]["stability_score"]
    )

    return {
        "strategy": stable_strategy,
        "stability_score": score,
        "stability_report": stability_report,
    }


def production_strategy_report(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """
    Return the full stability ranking used for
    production strategy selection.
    """
    result = select_production_strategy(
        results_dir=results_dir
    )

    return result[
        "stability_report"
    ].copy()
