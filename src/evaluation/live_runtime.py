from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.evaluation.live_production_decision import (
    build_live_production_decision,
)
from src.evaluation.live_trade_display import (
    build_live_trade_display,
)


@dataclass(frozen=True)
class LiveRuntimeResult:
    decision: dict[str, Any]
    display: dict[str, Any]


def build_live_runtime(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
) -> LiveRuntimeResult:
    """Build the complete live decision and display payload.

    The runtime connects the production decision layer to the
    human-readable trade display layer without inventing market data,
    signals, or trade levels.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    decision = build_live_production_decision(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
    )

    display = build_live_trade_display(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
    )

    if display["decision"] != decision["decision"]:
        raise ValueError(
            "Live decision and live display decisions do not match."
        )

    return LiveRuntimeResult(
        decision=decision,
        display=display,
    )
