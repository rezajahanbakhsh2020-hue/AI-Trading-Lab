import pandas as pd
import pytest

from src.evaluation.live_runtime_history import (
    LiveRuntimeHistory,
    append_live_runtime_snapshot,
    create_live_runtime_history,
    live_runtime_history_dataframe,
    live_runtime_history_records,
    live_runtime_history_summary,
    load_live_runtime_history,
    save_live_runtime_history,
)
from src.evaluation.live_runtime_snapshot import (
    LiveRuntimeSnapshot,
)


def _snapshot(
    ready: bool = True,
    timestamp: str = "2025-01-01T00:00:00+00:00",
) -> LiveRuntimeSnapshot:
    return LiveRuntimeSnapshot(
        timestamp=timestamp,
        symbol="XAUUSD",
        interval="5m",
        strategy="momentum",
        stability_score=0.90,
        controller_ready=ready,
        session_ready=ready,
        decision="BUY" if ready else None,
        trend="UP" if ready else None,
        entry=2605.0 if ready else None,
        stop_loss=2590.0 if ready else None,
        take_profit_1=2620.0 if ready else None,
        take_profit_2=2635.0 if ready else None,
        take_profit_3=2650.0 if ready else None,
        failed_gates=(),
        failed_checks=(),
    )


def test_create_history():
    history = create_live_runtime_history()

    assert isinstance(
        history,
        LiveRuntimeHistory,
    )
    assert history.count == 0
    assert history.latest is None


def test_create_history_with_snapshots():
    snapshot = _snapshot()

    history = create_live_runtime_history(
        [snapshot]
    )

    assert history.count == 1
    assert history.latest == snapshot


def test_append_snapshot():
    history = create_live_runtime_history()
    snapshot = _snapshot()

    updated = append_live_runtime_snapshot(
        history,
        snapshot,
    )

    assert history.count == 0
    assert updated.count == 1
    assert updated.latest == snapshot


def test_append_multiple_snapshots():
    history = create_live_runtime_history()

    first = _snapshot(
        timestamp="2025-01-01T00:00:00+00:00"
    )
    second = _snapshot(
        timestamp="2025-01-01T00:05:00+00:00"
    )

    history = append_live_runtime_snapshot(
        history,
        first,
    )
    history = append_live_runtime_snapshot(
        history,
        second,
    )

    assert history.count == 2
    assert history.latest == second


def test_records_conversion():
    snapshot = _snapshot()

    history = create_live_runtime_history(
        [snapshot]
    )

    records = live_runtime_history_records(
        history
    )

    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["symbol"] == "XAUUSD"
    assert records[0]["strategy"] == "momentum"


def test_dataframe_conversion():
    history = create_live_runtime_history(
        [_snapshot()]
    )

    dataframe = live_runtime_history_dataframe(
        history
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )
    assert len(dataframe) == 1
    assert dataframe.iloc[0]["symbol"] == "XAUUSD"


def test_empty_dataframe_conversion():
    history = create_live_runtime_history()

    dataframe = live_runtime_history_dataframe(
        history
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )
    assert dataframe.empty


def test_save_and_load_history(tmp_path):
    path = (
        tmp_path
        / "runtime"
        / "history.csv"
    )

    first = _snapshot(
        timestamp="2025-01-01T00:00:00+00:00"
    )
    second = _snapshot(
        ready=False,
        timestamp="2025-01-01T00:05:00+00:00",
    )

    history = create_live_runtime_history(
        [first, second]
    )

    saved_path = save_live_runtime_history(
        history,
        path,
    )

    assert saved_path == path
    assert path.exists()

    loaded = load_live_runtime_history(
        path
    )

    assert loaded.count == 2
    assert loaded.latest is not None
    assert loaded.latest.timestamp == (
        "2025-01-01T00:05:00+00:00"
    )
    assert loaded.latest.controller_ready is False


def test_load_missing_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_live_runtime_history(
            tmp_path / "missing.csv"
        )


def test_load_empty_history(tmp_path):
    path = tmp_path / "empty.csv"

    pd.DataFrame().to_csv(
        path,
        index=False,
    )

    history = load_live_runtime_history(
        path
    )

    assert history.count == 0


def test_summary_empty():
    history = create_live_runtime_history()

    assert live_runtime_history_summary(
        history
    ) == "LIVE RUNTIME HISTORY EMPTY"


def test_summary_ready():
    history = create_live_runtime_history(
        [_snapshot()]
    )

    message = live_runtime_history_summary(
        history
    )

    assert message.startswith(
        "LIVE RUNTIME HISTORY:"
    )
    assert "1 snapshots" in message
    assert "READY" in message
    assert "momentum" in message


def test_summary_blocked():
    history = create_live_runtime_history(
        [_snapshot(ready=False)]
    )

    message = live_runtime_history_summary(
        history
    )

    assert "BLOCKED" in message


def test_history_rejects_invalid_snapshot():
    with pytest.raises(TypeError):
        create_live_runtime_history(
            [object()]
        )


def test_append_rejects_invalid_history():
    with pytest.raises(TypeError):
        append_live_runtime_snapshot(
            {},
            _snapshot(),
        )


def test_append_rejects_invalid_snapshot():
    history = create_live_runtime_history()

    with pytest.raises(TypeError):
        append_live_runtime_snapshot(
            history,
            object(),
        )


def test_records_reject_invalid_history():
    with pytest.raises(TypeError):
        live_runtime_history_records({})


def test_dataframe_rejects_invalid_history():
    with pytest.raises(TypeError):
        live_runtime_history_dataframe({})


def test_save_rejects_invalid_history(tmp_path):
    with pytest.raises(TypeError):
        save_live_runtime_history(
            {},
            tmp_path / "history.csv",
        )


def test_summary_rejects_invalid_history():
    with pytest.raises(TypeError):
        live_runtime_history_summary({})
