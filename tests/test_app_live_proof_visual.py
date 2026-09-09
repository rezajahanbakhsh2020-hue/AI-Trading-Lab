import pandas as pd

from app_live_proof_visual import (
    build_quote_age_label,
    build_quote_table,
    build_risk_table,
    build_visual_summary,
)


def sample_snapshot() -> dict:
    return {
        "signal": 1,
        "signal_label": "BUY",
        "trend": "UP",
        "momentum": 0.01,
        "fast_ma": 4402.0,
        "slow_ma": 4395.0,
        "entry_price": 4403.0,
        "stop_loss": 4358.97,
        "take_profit": 4491.06,
        "risk_reward_ratio": 2.0,
        "timestamp": "2026-09-09T15:30:00+00:00",
        "market_state": "OPEN",
        "quote_stale": False,
        "quote_age_seconds": 0,
        "bid": 4402.8,
        "ask": 4403.2,
        "mid": 4403.0,
        "candle_count": 200,
    }


def test_build_visual_summary():
    summary = build_visual_summary(
        sample_snapshot()
    )

    assert summary["signal"] == "BUY"
    assert summary["trend"] == "UP"
    assert summary["market"] == "OPEN"
    assert summary["entry"] == "4403.00"
    assert summary["stop_loss"] == "4358.97"
    assert summary["take_profit"] == "4491.06"


def test_build_quote_table():
    table = build_quote_table(
        sample_snapshot()
    )

    assert isinstance(table, pd.DataFrame)
    assert list(table["Quote"]) == [
        "Bid",
        "Ask",
        "Mid",
    ]
    assert table.iloc[2]["Price"] == 4403.0


def test_build_risk_table():
    table = build_risk_table(
        sample_snapshot()
    )

    assert list(table["Level"]) == [
        "Entry",
        "Stop Loss",
        "Take Profit",
    ]
    assert table.iloc[0]["Price"] == 4403.0


def test_quote_age_label():
    assert (
        build_quote_age_label(
            sample_snapshot()
        )
        == "0s old"
    )


def test_stale_quote_label():
    snapshot = sample_snapshot()
    snapshot["quote_stale"] = True

    assert (
        build_quote_age_label(snapshot)
        == "STALE"
    )


def test_unknown_quote_age_label():
    snapshot = sample_snapshot()
    snapshot["quote_age_seconds"] = None

    assert (
        build_quote_age_label(snapshot)
        == "UNKNOWN"
    )
