from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.experiment_comparison import (
    compare_experiments,
)
from src.evaluation.stability import (
    calculate_stability_score,
    get_stable_strategy,
)


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def build_stability_report(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    """
    Build a cross-experiment stability report.
    """
    comparison = compare_experiments(
        results_dir
    )

    return calculate_stability_score(
        comparison
    )


def select_stable_strategy(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> str | None:
    """
    Select the most stable strategy across
    all stored walk-forward experiments.
    """
    comparison = compare_experiments(
        results_dir
    )

    return get_stable_strategy(
        comparison
    )
