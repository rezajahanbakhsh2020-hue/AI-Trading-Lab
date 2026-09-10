from pathlib import Path

import pandas as pd

from src.evaluation.live_runtime_history import create_live_runtime_history
from src.evaluation.live_runtime_history_export import (
    export_live_runtime_history,
    export_live_runtime_history_summary,
)
from src.evaluation.live_runtime_snapshot import LiveRuntimeSnapshot


def _snapshot(timestamp: str) -> LiveRuntimeSnapshot:
    return LiveRuntimeSnapshot(
        timestamp=timestamp,
        symbol="XAUUSD",
        interval="5m",
        strategy="momentum",
        stability_score=0.8,
        controller_ready=True,
        session_ready=True,
        decision="BUY",
        trend="UP",
        entry=3000.0,
        stop_loss=2990.0,
        take_profit_1=3010.0,
        take_profit_2=3020.0,
        take_profit_3=3030.0,
        failed_gates=(),
        failed_checks=(),
    )


def test_export_live_runtime_history(tmp_path: Path) -> None:
    history = create_live_runtime_history(
        [
            _snapshot("2026-01-01T00:00:00+00:00"),
            _snapshot("2026-01-01T00:05:00+00:00"),
        ]
    )

    output = tmp_path / "runtime_history.csv"

    result = export_live_runtime_history(history, output)

    assert result == output
    assert output.exists()

    frame = pd.read_csv(output)

    assert len(frame) == 2
    assert list(frame["symbol"]) == ["XAUUSD", "XAUUSD"]
    assert list(frame["decision"]) == ["BUY", "BUY"]
    assert list(frame["strategy"]) == ["momentum", "momentum"]


def test_export_live_runtime_history_creates_parent_directory(
    tmp_path: Path,
) -> None:
    history = create_live_runtime_history(
        [_snapshot("2026-01-01T00:00:00+00:00")]
    )

    output = tmp_path / "exports" / "nested" / "history.csv"

    result = export_live_runtime_history(history, output)

    assert result == output
    assert output.exists()


def test_export_live_runtime_history_summary(tmp_path: Path) -> None:
    history = create_live_runtime_history(
        [
            _snapshot("2026-01-01T00:00:00+00:00"),
            _snapshot("2026-01-01T00:05:00+00:00"),
        ]
    )

    output = tmp_path / "summary.json"

    result = export_live_runtime_history_summary(history, output)

    assert result == output
    assert output.exists()

    data = pd.read_json(output)

    assert len(data) == 1
    assert int(data.loc[0, "count"]) == 2
    assert data.loc[0, "symbol"] == "XAUUSD"
    assert data.loc[0, "interval"] == "5m"
    assert data.loc[0, "latest_strategy"] == "momentum"
    assert data.loc[0, "latest_decision"] == "BUY"
    assert data.loc[0, "health_status"] == "HEALTHY"


def test_export_rejects_invalid_history(tmp_path: Path) -> None:
    output = tmp_path / "history.csv"

    try:
        export_live_runtime_history(None, output)  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "history must be a LiveRuntimeHistory."
    else:
        raise AssertionError("TypeError was not raised.")


def test_summary_export_rejects_invalid_history(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"

    try:
        export_live_runtime_history_summary(None, output)  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "history must be a LiveRuntimeHistory."
    else:
        raise AssertionError("TypeError was not raised.")
