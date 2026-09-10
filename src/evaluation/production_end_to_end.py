from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.end_to_end_runner import run_end_to_end
from src.evaluation.production_live_bridge import (
    load_production_selection,
)


DEFAULT_DATA_PATH = Path(
    "data/raw/xauusd_daily_2025.csv"
)


def run_production_end_to_end(
    data: pd.DataFrame,
    *,
    results_dir: str | Path = "results/production",
    symbol: str = "XAUUSD",
    interval: str = "1d",
    min_stability_score: float = 0.50,
) -> dict[str, Any]:
    """Connect the saved production selection to the live E2E pipeline."""

    selection = load_production_selection(
        results_dir
    )

    result = run_end_to_end(
        data,
        stable_strategy=selection["stable_strategy"],
        stability_score=selection["stability_score"],
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
    )

    result["production_selection"] = selection

    result["end_to_end_ready"] = (
        result["end_to_end_ready"]
        and result["release_gate"]["release_ready"]
    )

    return result


def load_production_market_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """Load and validate the production market dataset."""

    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Market data file not found: {path}"
        )

    data = pd.read_csv(path)

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    data = data.copy()

    if pd.api.types.is_numeric_dtype(data["timestamp"]):
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            unit="s",
            errors="coerce",
        )
    else:
        data["timestamp"] = pd.to_datetime(
            data["timestamp"],
            errors="coerce",
        )

    for column in (
        "open",
        "high",
        "low",
        "close",
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    if data.empty:
        raise ValueError(
            "No valid market data available."
        )

    return data.reset_index(drop=True)
