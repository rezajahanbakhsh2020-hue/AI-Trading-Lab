import json

import pandas as pd
import pytest

import run_live_proof


def make_market_data(closes):
    closes = [float(value) for value in closes]

    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=len(closes),
                freq="5min",
            ),
            "open": closes,
            "high": [value + 2.0 for value in closes],
            "low": [value - 2.0 for value in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
            "tickVolume": [100.0] * len(closes),
            "isOpen": [False] * len(closes),
        }
    )


def test_run_live_proof_completes_full_pipeline(tmp_path):
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    def fake_fetch_ohlc(*, interval, limit):
        assert interval == "5m"
        assert limit == 6
        return data

    def fake_fetch_quote():
        return {
            "marketState": "OPEN",
            "quoteAgeSeconds": 2,
            "stale": False,
            "bid": 5.99,
            "ask": 6.01,
            "mid": 6.00,
        }

    snapshot_path = tmp_path / "live_proof.json"

    result = run_live_proof.run_live_proof(
        symbol="XAUUSD",
        interval="5m",
        limit=6,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        snapshot_path=snapshot_path,
        fetch_ohlc=fake_fetch_ohlc,
        fetch_quote=fake_fetch_quote,
    )

    snapshot = result["snapshot"]

    assert result["saved_path"] == str(snapshot_path)
    assert snapshot_path.exists()

    assert snapshot["symbol"] == "XAUUSD"
    assert snapshot["interval"] == "5m"
    assert snapshot["candle_count"] == 6

    assert snapshot["signal"] == 1
    assert snapshot["signal_label"] == "BUY"
    assert snapshot["trend"] == "UP"

    assert snapshot["entry_price"] == 6.0
    assert snapshot["stop_loss"] == pytest.approx(5.94)
    assert snapshot["take_profit"] == pytest.approx(6.12)
    assert snapshot["risk_reward_ratio"] == pytest.approx(2.0)

    assert snapshot["market_state"] == "OPEN"
    assert snapshot["quote_age_seconds"] == 2
    assert snapshot["quote_stale"] is False
    assert snapshot["latest_bid"] == 5.99
    assert snapshot["latest_ask"] == 6.01
    assert snapshot["latest_mid"] == 6.0


def test_run_live_proof_persists_no_trade_state(tmp_path):
    data = make_market_data(
        [6, 5, 4, 3, 2, 1]
    )

    def fake_fetch_ohlc(**_kwargs):
        return data

    def fake_fetch_quote():
        return {
            "marketState": "OPEN",
            "quoteAgeSeconds": 1,
            "stale": False,
        }

    snapshot_path = tmp_path / "no_trade.json"

    result = run_live_proof.run_live_proof(
        interval="5m",
        limit=6,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
        snapshot_path=snapshot_path,
        fetch_ohlc=fake_fetch_ohlc,
        fetch_quote=fake_fetch_quote,
    )

    snapshot = result["snapshot"]

    assert snapshot["signal"] == 0
    assert snapshot["signal_label"] == "NO TRADE"
    assert snapshot["trend"] == "DOWN"
    assert snapshot["stop_loss"] is None
    assert snapshot["take_profit"] is None
    assert snapshot["risk_reward_ratio"] is None


def test_run_live_proof_saved_file_is_valid_json(tmp_path):
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    def fake_fetch_ohlc(**_kwargs):
        return data

    def fake_fetch_quote():
        return {
            "marketState": "OPEN",
            "quoteAgeSeconds": 3,
            "stale": False,
        }

    snapshot_path = tmp_path / "proof.json"

    run_live_proof.run_live_proof(
        interval="5m",
        limit=6,
        momentum_window=2,
        fast_window=2,
        slow_window=4,
        snapshot_path=snapshot_path,
        fetch_ohlc=fake_fetch_ohlc,
        fetch_quote=fake_fetch_quote,
    )

    payload = json.loads(
        snapshot_path.read_text(encoding="utf-8")
    )

    assert payload["symbol"] == "XAUUSD"
    assert payload["signal_label"] == "BUY"
    assert payload["market_state"] == "OPEN"
    assert payload["candle_count"] == 6


def test_run_live_proof_rejects_invalid_limit(tmp_path):
    with pytest.raises(
        ValueError,
        match="limit must be greater than zero",
    ):
        run_live_proof.run_live_proof(
            limit=0,
            snapshot_path=tmp_path / "proof.json",
            fetch_ohlc=lambda **_kwargs: pd.DataFrame(),
            fetch_quote=lambda: {},
        )


def test_run_live_proof_rejects_empty_market_data(tmp_path):
    def fake_fetch_ohlc(**_kwargs):
        return pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="returned no candles",
    ):
        run_live_proof.run_live_proof(
            interval="5m",
            limit=10,
            snapshot_path=tmp_path / "proof.json",
            fetch_ohlc=fake_fetch_ohlc,
            fetch_quote=lambda: {},
        )


def test_run_live_proof_rejects_invalid_quote(tmp_path):
    data = make_market_data(
        [1, 2, 3, 4, 5, 6]
    )

    def fake_fetch_ohlc(**_kwargs):
        return data

    with pytest.raises(
        ValueError,
        match="quote fetch must return a dictionary",
    ):
        run_live_proof.run_live_proof(
            interval="5m",
            limit=6,
            snapshot_path=tmp_path / "proof.json",
            fetch_ohlc=fake_fetch_ohlc,
            fetch_quote=lambda: None,
        )
