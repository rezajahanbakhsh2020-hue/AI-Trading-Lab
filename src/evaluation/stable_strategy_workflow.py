from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.stable_selection import (
    build_stability_report,
    select_stable_strategy,
)


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def run_stable_strategy_selection(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict:
    """
    Select the most stable strategy from
    previously stored walk-forward experiments.
    """
    stability_report = build_stability_report(
        results_dir
    )

    stable_strategy = select_stable_strategy(
        results_dir
    )

    return {
        "stability_report": stability_report,
        "stable_strategy": stable_strategy,
    }


def get_stable_strategy_summary(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """
    Return the stability ranking as a DataFrame.
    """
    result = run_stable_strategy_selection(
        results_dir
    )

    return result["stability_report"].copy()
