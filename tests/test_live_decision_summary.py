from __future__ import annotations

import pytest

from src.evaluation.live_decision_summary import (
    get_latest_live_decision_summary,
    get_live_decision_counts,
    summarize_live_decisions,
)


def _record(
    timestamp: str,
    signal_label: str,
    trend: str,
    strategy: str = "momentum",
) -> dict:
    return {
        "timestamp": timestamp,
        "symbol": "XAUUSD",
        "interval": "5m",
        "signal": 1 if signal_label == "BUY" else 0,
        "signal_label": signal_label,
        "trend": trend,
        "strategy": strategy,
        "entry_price": 4429.802,
        "stop_loss": 4385.50398,
        "take_profit": 4518.39804,
        "risk_reward_ratio": 2.0,
        "stability_score": 0.65,
        "market_state": "open",
        "quote_age_seconds": 0,
        "quote_stale": False,
        "candle_count": 201,
    }


def test_summarize_live_decisions_counts_records():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP"),
        _record("2026-09-10T05:50:00+00:00", "NO TRADE", "FLAT"),
        _record("2026-09-10T05:55:00+00:00", "BUY", "UP"),
    ]

    summary = summarize_live_decisions(history)

    assert summary["record_count"] == 3


def test_summarize_live_decisions_counts_signals():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP"),
        _record("2026-09-10T05:50:00+00:00", "NO TRADE", "FLAT"),
        _record("2026-09-10T05:55:00+00:00", "BUY", "UP"),
    ]

    summary = summarize_live_decisions(history)

    assert summary["signal_counts"] == {
        "BUY": 2,
        "NO TRADE": 1,
    }


def test_summarize_live_decisions_counts_trends():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP"),
        _record("2026-09-10T05:50:00+00:00", "NO TRADE", "FLAT"),
        _record("2026-09-10T05:55:00+00:00", "BUY", "UP"),
    ]

    summary = summarize_live_decisions(history)

    assert summary["trend_counts"] == {
        "UP": 2,
        "FLAT": 1,
    }


def test_summarize_live_decisions_counts_strategies():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP", "momentum"),
        _record(
            "2026-09-10T05:50:00+00:00",
            "NO TRADE",
            "FLAT",
            "mean_reversion",
        ),
        _record("2026-09-10T05:55:00+00:00", "BUY", "UP", "momentum"),
    ]

    summary = summarize_live_decisions(history)

    assert summary["strategy_counts"] == {
        "momentum": 2,
        "mean_reversion": 1,
    }


def test_summarize_live_decisions_returns_latest_record():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP"),
        _record("2026-09-10T05:50:00+00:00", "NO TRADE", "FLAT"),
    ]

    summary = summarize_live_decisions(history)

    assert summary["latest"]["timestamp"] == (
        "2026-09-10T05:50:00+00:00"
    )
    assert summary["latest"]["signal_label"] == "NO TRADE"


def test_summarize_live_decisions_empty_history():
    summary = summarize_live_decisions([])

    assert summary["record_count"] == 0
    assert summary["signal_counts"] == {}
    assert summary["trend_counts"] == {}
    assert summary["strategy_counts"] == {}
    assert summary["latest"] is None


def test_summarize_live_decisions_does_not_recalculate_values():
    record = _record(
        "2026-09-10T05:45:00+00:00",
        "BUY",
        "UP",
    )
    record["entry_price"] = 9999.0

    summary = summarize_live_decisions([record])

    assert summary["latest"]["entry_price"] == 9999.0


def test_summarize_live_decisions_rejects_invalid_input():
    with pytest.raises(TypeError):
        summarize_live_decisions("invalid")


def test_summarize_live_decisions_rejects_invalid_record():
    with pytest.raises(TypeError):
        summarize_live_decisions([[]])


def test_get_live_decision_counts_returns_signal_counts():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP"),
        _record("2026-09-10T05:50:00+00:00", "NO TRADE", "FLAT"),
    ]

    counts = get_live_decision_counts(history)

    assert counts == {
        "BUY": 1,
        "NO TRADE": 1,
    }


def test_get_latest_live_decision_summary_returns_copy():
    history = [
        _record("2026-09-10T05:45:00+00:00", "BUY", "UP")
    ]

    latest = get_latest_live_decision_summary(history)

    assert latest is not None

    latest["signal_label"] = "CHANGED"

    assert history[0]["signal_label"] == "BUY"


def test_get_latest_live_decision_summary_empty_history():
    assert get_latest_live_decision_summary([]) is None
