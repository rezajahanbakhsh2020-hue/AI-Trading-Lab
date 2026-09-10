from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.live_runtime_snapshot import (
    LiveRuntimeSnapshot,
    live_runtime_snapshot_dict,
)


@dataclass(frozen=True)
class LiveRuntimeHistory:
    snapshots: tuple[LiveRuntimeSnapshot, ...]

    @property
    def count(self) -> int:
        return len(self.snapshots)

    @property
    def latest(self) -> LiveRuntimeSnapshot | None:
        if not self.snapshots:
            return None
        return self.snapshots[-1]


def create_live_runtime_history(
    snapshots: list[LiveRuntimeSnapshot] | tuple[
        LiveRuntimeSnapshot, ...
    ] = (),
) -> LiveRuntimeHistory:
    """
    Create an immutable live-runtime history.
    """

    normalized = tuple(snapshots)

    for snapshot in normalized:
        if not isinstance(
            snapshot,
            LiveRuntimeSnapshot,
        ):
            raise TypeError(
                "all snapshots must be "
                "LiveRuntimeSnapshot instances."
            )

    return LiveRuntimeHistory(
        snapshots=normalized
    )


def append_live_runtime_snapshot(
    history: LiveRuntimeHistory,
    snapshot: LiveRuntimeSnapshot,
) -> LiveRuntimeHistory:
    """
    Append one runtime snapshot to the history.
    """

    if not isinstance(
        history,
        LiveRuntimeHistory,
    ):
        raise TypeError(
            "history must be a LiveRuntimeHistory."
        )

    if not isinstance(
        snapshot,
        LiveRuntimeSnapshot,
    ):
        raise TypeError(
            "snapshot must be a LiveRuntimeSnapshot."
        )

    return LiveRuntimeHistory(
        snapshots=history.snapshots + (snapshot,)
    )


def live_runtime_history_records(
    history: LiveRuntimeHistory,
) -> list[dict[str, Any]]:
    """
    Convert runtime history into plain records.
    """

    if not isinstance(
        history,
        LiveRuntimeHistory,
    ):
        raise TypeError(
            "history must be a LiveRuntimeHistory."
        )

    return [
        live_runtime_snapshot_dict(snapshot)
        for snapshot in history.snapshots
    ]


def live_runtime_history_dataframe(
    history: LiveRuntimeHistory,
) -> pd.DataFrame:
    """
    Convert runtime history into a DataFrame.
    """

    records = live_runtime_history_records(
        history
    )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def save_live_runtime_history(
    history: LiveRuntimeHistory,
    path: str | Path,
) -> Path:
    """
    Persist runtime history as CSV.

    Parent directories are created automatically.
    """

    if not isinstance(
        history,
        LiveRuntimeHistory,
    ):
        raise TypeError(
            "history must be a LiveRuntimeHistory."
        )

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = live_runtime_history_dataframe(
        history
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def load_live_runtime_history(
    path: str | Path,
) -> LiveRuntimeHistory:
    """
    Load runtime history from CSV.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"runtime history not found: {input_path}"
        )

    try:
        dataframe = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        return create_live_runtime_history()

    if dataframe.empty:
        return create_live_runtime_history()

    snapshots: list[LiveRuntimeSnapshot] = []

    for record in dataframe.to_dict(
        orient="records"
    ):
        snapshots.append(
            LiveRuntimeSnapshot(
                timestamp=str(
                    record.get("timestamp", "")
                ),
                symbol=str(
                    record.get("symbol", "")
                ),
                interval=str(
                    record.get("interval", "")
                ),
                strategy=(
                    None
                    if pd.isna(
                        record.get("strategy")
                    )
                    else str(
                        record.get("strategy")
                    )
                ),
                stability_score=(
                    None
                    if pd.isna(
                        record.get(
                            "stability_score"
                        )
                    )
                    else float(
                        record.get(
                            "stability_score"
                        )
                    )
                ),
                controller_ready=bool(
                    record.get(
                        "controller_ready",
                        False,
                    )
                ),
                session_ready=bool(
                    record.get(
                        "session_ready",
                        False,
                    )
                ),
                decision=(
                    None
                    if pd.isna(
                        record.get("decision")
                    )
                    else str(
                        record.get("decision")
                    )
                ),
                trend=(
                    None
                    if pd.isna(
                        record.get("trend")
                    )
                    else str(
                        record.get("trend")
                    )
                ),
                entry=_optional_float(
                    record.get("entry")
                ),
                stop_loss=_optional_float(
                    record.get("stop_loss")
                ),
                take_profit_1=_optional_float(
                    record.get("take_profit_1")
                ),
                take_profit_2=_optional_float(
                    record.get("take_profit_2")
                ),
                take_profit_3=_optional_float(
                    record.get("take_profit_3")
                ),
                failed_gates=_parse_sequence(
                    record.get("failed_gates")
                ),
                failed_checks=_parse_sequence(
                    record.get("failed_checks")
                ),
            )
        )

    return create_live_runtime_history(
        snapshots
    )


def _optional_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    if pd.isna(value):
        return None

    return float(value)


def _parse_sequence(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        return ()

    if pd.isna(value):
        return ()

    if isinstance(value, (list, tuple)):
        return tuple(
            str(item)
            for item in value
        )

    text = str(value).strip()

    if not text:
        return ()

    return tuple(
        item.strip()
        for item in text.split(",")
        if item.strip()
    )


def live_runtime_history_summary(
    history: LiveRuntimeHistory,
) -> str:
    """
    Return a concise history status message.
    """

    if not isinstance(
        history,
        LiveRuntimeHistory,
    ):
        raise TypeError(
            "history must be a LiveRuntimeHistory."
        )

    if history.count == 0:
        return "LIVE RUNTIME HISTORY EMPTY"

    latest = history.latest

    assert latest is not None

    status = (
        "READY"
        if latest.controller_ready
        else "BLOCKED"
    )

    return (
        "LIVE RUNTIME HISTORY: "
        f"{history.count} snapshots; "
        f"latest={status}; "
        f"strategy={latest.strategy}"
    )
