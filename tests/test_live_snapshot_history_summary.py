from __future__ import annotations

from live_snapshot_history import append_live_snapshot
from live_snapshot_history_summary import summarize_live_snapshot_history


def sample_snapshot(
    *,
    signal: int = 0,
    signal_label: str = "NO TRADE",
    trend: str = "UP",
    timestamp: str = "2026-09-09T15:30:00+00:00",
) -> dict:
    return {
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": signal,
        "signal_label": signal_label,
        "trend": trend,
        "strategy": "momentum",
        "momentum": -0.0053,
        "entry_price": 4395.596,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "timestamp": timestamp,
    }


def test_empty_history_returns_zero_summary(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("", encoding="utf-8")

    summary = summarize_live_snapshot_history(path)

    assert summary["records"] == 0
    assert summary["buy_count"] == 0
    assert summary["no_trade_count"] == 0
    assert summary["other_signal_count"] == 0
    assert summary["buy_ratio"] == 0.0
    assert summary["trend_counts"] == {}
    assert summary["symbol_counts"] == {}
    assert summary["interval_counts"] == {}
    assert summary["first_timestamp"] is None
    assert summary["last_timestamp"] is None


def test_summary_counts_buy_and_no_trade(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(sample_snapshot(), path)
    append_live_snapshot(
        sample_snapshot(
            signal=1,
            signal_label="BUY",
            timestamp="2026-09-09T15:35:00+00:00",
        ),
        path,
    )
    append_live_snapshot(
        sample_snapshot(
            timestamp="2026-09-09T15:40:00+00:00",
        ),
        path,
    )

    summary = summarize_live_snapshot_history(path)

    assert summary["records"] == 3
    assert summary["buy_count"] == 1
    assert summary["no_trade_count"] == 2
    assert summary["other_signal_count"] == 0
    assert summary["buy_ratio"] == 1 / 3


def test_summary_counts_trends_symbols_and_intervals(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(
            trend="UP",
            timestamp="2026-09-09T15:30:00+00:00",
        ),
        path,
    )
    append_live_snapshot(
        sample_snapshot(
            signal=1,
            signal_label="BUY",
            trend="UP",
            timestamp="2026-09-09T15:35:00+00:00",
        ),
        path,
    )
    append_live_snapshot(
        sample_snapshot(
            trend="DOWN",
            timestamp="2026-09-09T15:40:00+00:00",
        ),
        path,
    )

    summary = summarize_live_snapshot_history(path)

    assert summary["trend_counts"] == {
        "UP": 2,
        "DOWN": 1,
    }
    assert summary["symbol_counts"] == {
        "XAUUSD": 3,
    }
    assert summary["interval_counts"] == {
        "5m": 3,
    }


def test_summary_preserves_first_and_last_timestamp(tmp_path):
    path = tmp_path / "history.jsonl"

    first = "2026-09-09T15:30:00+00:00"
    last = "2026-09-09T16:00:00+00:00"

    append_live_snapshot(
        sample_snapshot(timestamp=first),
        path,
    )
    append_live_snapshot(
        sample_snapshot(
            signal=1,
            signal_label="BUY",
            timestamp=last,
        ),
        path,
    )

    summary = summarize_live_snapshot_history(path)

    assert summary["first_timestamp"] == first
    assert summary["last_timestamp"] == last


def test_summary_counts_unexpected_signal_values(tmp_path):
    path = tmp_path / "history.jsonl"

    append_live_snapshot(
        sample_snapshot(signal=2, signal_label="OTHER"),
        path,
    )

    summary = summarize_live_snapshot_history(path)

    assert summary["records"] == 1
    assert summary["buy_count"] == 0
    assert summary["no_trade_count"] == 0
    assert summary["other_signal_count"] == 1


def test_summary_ignores_missing_optional_metadata(tmp_path):
    path = tmp_path / "history.jsonl"

    snapshot = sample_snapshot()
    snapshot.pop("trend")
    snapshot.pop("symbol")
    snapshot.pop("interval")
    snapshot.pop("timestamp")

    append_live_snapshot(snapshot, path)

    summary = summarize_live_snapshot_history(path)

    assert summary["records"] == 1
    assert summary["trend_counts"] == {}
    assert summary["symbol_counts"] == {}
    assert summary["interval_counts"] == {}
    assert summary["first_timestamp"] is None
    assert summary["last_timestamp"] is None


def test_summary_raises_for_missing_history(tmp_path):
    path = tmp_path / "missing.jsonl"

    try:
        summarize_live_snapshot_history(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
