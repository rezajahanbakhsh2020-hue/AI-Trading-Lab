from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.live_runtime_history import LiveRuntimeHistory
from src.evaluation.live_runtime_history_summary import (
    build_live_runtime_history_summary,
)


def export_live_runtime_history(
    history: LiveRuntimeHistory,
    output_path: str | Path,
) -> Path:
    if not isinstance(history, LiveRuntimeHistory):
        raise TypeError("history must be a LiveRuntimeHistory.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for snapshot in history.snapshots:
        rows.append(
            {
                "timestamp": snapshot.timestamp,
                "symbol": snapshot.symbol,
                "interval": snapshot.interval,
                "strategy": snapshot.strategy,
                "stability_score": snapshot.stability_score,
                "controller_ready": snapshot.controller_ready,
                "session_ready": snapshot.session_ready,
                "decision": snapshot.decision,
                "trend": snapshot.trend,
                "entry": snapshot.entry,
                "stop_loss": snapshot.stop_loss,
                "take_profit_1": snapshot.take_profit_1,
                "take_profit_2": snapshot.take_profit_2,
                "take_profit_3": snapshot.take_profit_3,
                "failed_gates": ",".join(snapshot.failed_gates),
                "failed_checks": ",".join(snapshot.failed_checks),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def export_live_runtime_history_summary(
    history: LiveRuntimeHistory,
    output_path: str | Path,
) -> Path:
    if not isinstance(history, LiveRuntimeHistory):
        raise TypeError("history must be a LiveRuntimeHistory.")

    summary = build_live_runtime_history_summary(history)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "count": summary.count,
        "latest_timestamp": summary.latest_timestamp,
        "symbol": summary.symbol,
        "interval": summary.interval,
        "latest_strategy": summary.latest_strategy,
        "latest_decision": summary.latest_decision,
        "latest_trend": summary.latest_trend,
        "strategy_count": summary.strategy_count,
        "strategy_changes": summary.strategy_changes,
        "decision_counts": dict(summary.decision_counts),
        "trend_counts": dict(summary.trend_counts),
        "controller_ready_count": summary.controller_ready_count,
        "session_ready_count": summary.session_ready_count,
        "ready_ratio": summary.ready_ratio,
        "health_status": summary.health_status,
        "health_score": summary.health_score,
    }

    pd.DataFrame([row]).to_json(
        path,
        orient="records",
        indent=2,
        force_ascii=False,
    )

    return path
