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
    candidate_id: str | None = None,
    results_dir: str | Path = "results/production",
    research_dir: Any | None = None,
    symbol: str = "XAUUSD",
    interval: str = "1d",
    min_stability_score: float = 0.50,
    reference_now: Any | None = None,
    store_path: Any | None = None,
) -> dict[str, Any]:
    """Connect promoted candidate lineage to the live E2E pipeline."""
    from src.evaluation.research_store import resolve_promoted_candidate, DEFAULT_RESEARCH_DIR

    r_dir = research_dir if research_dir is not None else DEFAULT_RESEARCH_DIR
    resolved_candidate = resolve_promoted_candidate(
        candidate_id=candidate_id,
        strategy_id="momentum",
        symbol=symbol,
        timeframe=interval,
        base_dir=r_dir,
    )

    if resolved_candidate is None:
        raise ValueError(
            f"No promoted candidate found for candidate_id={candidate_id!r}, symbol={symbol!r}, timeframe={interval!r}"
        )

    result = run_end_to_end(
        data,
        stable_strategy=resolved_candidate.strategy_name,
        stability_score=None,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
        reference_now=reference_now,
        store_path=store_path,
    )

    result["production_selection"] = {
        "stable_strategy": resolved_candidate.strategy_name,
        "stability_score": resolved_candidate.operational_stability_score,
        "candidate_id": resolved_candidate.candidate_id,
        "source": "promoted_candidate_binding",
    }

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
