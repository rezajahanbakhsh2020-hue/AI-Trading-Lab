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
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    min_stability_score: float = 0.50,
) -> LiveRuntimeResult:
    """Build the live decision and the human-readable trade display."""

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    decision = build_live_production_decision(
        data=data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        min_stability_score=min_stability_score,
        symbol=symbol,
        interval=interval,
    )

    display = build_live_trade_display(
        data=data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        min_stability_score=min_stability_score,
        symbol=symbol,
        interval=interval,
    )

    if display["signal"] != decision["signal"]:
        raise ValueError(
            "Live runtime signal and trade display signal do not match."
        )

    if display["trend"] != decision["trend"]:
        raise ValueError(
            "Live runtime trend and trade display trend do not match."
        )

    if display["entry_price"] != decision["entry_price"]:
        raise ValueError(
            "Live runtime entry and trade display entry do not match."
        )

    if display["stop_loss"] != decision["stop_loss"]:
        raise ValueError(
            "Live runtime stop loss and trade display stop loss do not match."
        )

    if display["take_profit"] != decision["take_profit"]:
        raise ValueError(
            "Live runtime take profit and trade display take profit do not match."
        )

    for level in ("tp1", "tp2", "tp3"):
        if display.get(level) != decision.get(level):
            raise ValueError(
                f"Live runtime {level} and trade display {level} do not match."
            )

    return LiveRuntimeResult(
        decision=decision,
        display=display,
    )
