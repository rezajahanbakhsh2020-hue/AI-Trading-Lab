from __future__ import annotations

import pandas as pd
import pytest

import app_live_trade_runtime


def _sample_live_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.to_datetime(
                [
                    "2026-09-10 10:00:00+00:00",
                    "2026-09-10 10:05:00+00:00",
                    "2026-09-10 10:10:00+00:00",
                ]
            ),
            "open": [4400.0, 4401.0, 4402.0],
            "high": [4402.0, 4403.0, 4404.0],
            "low": [4399.0, 4400.0, 4401.0],
            "close": [4401.0, 4402.0, 4403.0],
        }
    )


def test_load_runtime_data_uses_live_biquote(monkeypatch):
    expected = _sample_live_data()

    calls = {}

    def fake_fetch_xauusd_ohlc(interval, limit):
        calls["interval"] = interval
        calls["limit"] = limit
        return expected.copy()

    monkeypatch.setattr(
        app_live_trade_runtime,
        "fetch_xauusd_ohlc",
        fake_fetch_xauusd_ohlc,
    )

    result = app_live_trade_runtime._load_runtime_data()

    assert calls["interval"] == app_live_trade_runtime.INTERVAL
    assert calls["limit"] == app_live_trade_runtime.LIMIT
    assert len(result) == 3
    assert "timestamp" in result.columns
    assert result["close"].tolist() == [4401.0, 4402.0, 4403.0]


def test_load_runtime_data_rejects_missing_live_columns(monkeypatch):
    invalid = pd.DataFrame(
        {
            "openTime": pd.to_datetime(
                ["2026-09-10 10:00:00+00:00"]
            ),
            "open": [4400.0],
            "high": [4402.0],
            "low": [4399.0],
        }
    )

    monkeypatch.setattr(
        app_live_trade_runtime,
        "fetch_xauusd_ohlc",
        lambda interval, limit: invalid,
    )

    with pytest.raises(ValueError, match="Missing required live columns"):
        app_live_trade_runtime._load_runtime_data()


def test_load_stable_selection(monkeypatch):
    monkeypatch.setattr(
        app_live_trade_runtime,
        "load_production_selection",
        lambda: {
            "stable_strategy": "momentum",
            "stability_score": 0.517268,
            "source_path": "results/production/latest.json",
        },
    )

    result = app_live_trade_runtime._load_stable_selection()

    assert result["stable_strategy"] == "momentum"
    assert result["stability_score"] == pytest.approx(0.517268)
    assert result["source_path"] == "results/production/latest.json"


def test_load_stable_selection_rejects_missing_strategy(monkeypatch):
    monkeypatch.setattr(
        app_live_trade_runtime,
        "load_production_selection",
        lambda: {
            "stability_score": 0.517268,
            "source_path": "results/production/latest.json",
        },
    )

    with pytest.raises(
        ValueError,
        match="stable strategy",
    ):
        app_live_trade_runtime._load_stable_selection()


def test_load_stable_selection_rejects_missing_score(monkeypatch):
    monkeypatch.setattr(
        app_live_trade_runtime,
        "load_production_selection",
        lambda: {
            "stable_strategy": "momentum",
            "source_path": "results/production/latest.json",
        },
    )

    with pytest.raises(
        ValueError,
        match="stability score",
    ):
        app_live_trade_runtime._load_stable_selection()
