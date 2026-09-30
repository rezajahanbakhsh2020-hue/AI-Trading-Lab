from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.live_release_gate import validate_live_release
from src.evaluation.live_runtime import build_live_runtime
from src.visualization.live_trade_overlay import build_live_trade_overlay


def run_end_to_end(
    data: pd.DataFrame,
    *,
    stable_strategy: str,
    stability_score: float,
    symbol: str = "XAUUSD",
    interval: str = "1d",
    min_stability_score: float = 0.50,
    reference_now: Any | None = None,
    store_path: Any | None = None,
) -> dict[str, Any]:
    """Run the complete live trading proof pipeline."""
    if reference_now is None and isinstance(data, pd.DataFrame) and "timestamp" in data.columns and not data.empty:
        last_ts = pd.to_datetime(data["timestamp"].iloc[-1], utc=True)
        ref_now = last_ts.to_pydatetime() if pd.notna(last_ts) else None
    else:
        ref_now = reference_now

    runtime = build_live_runtime(
        data,
        stable_strategy=stable_strategy,
        stability_score=stability_score,
        symbol=symbol,
        interval=interval,
        min_stability_score=min_stability_score,
        reference_now=ref_now,
        store_path=store_path,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    release_gate = validate_live_release(
        decision=runtime.decision,
        display=runtime.display,
        min_stability_score=min_stability_score,
        required_strategy=stable_strategy,
    )

    return {
        "decision": runtime.decision,
        "display": runtime.display,
        "overlay": overlay,
        "release_gate": release_gate,
        "end_to_end_ready": release_gate["release_ready"],
    }
