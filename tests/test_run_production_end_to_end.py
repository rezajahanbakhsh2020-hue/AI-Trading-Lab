from __future__ import annotations

import json

import pandas as pd
import pytest

import run_production_end_to_end


def _rising_data(rows: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="1D",
    )

    close = pd.Series(
        [2000.0 + index * 2.0 for index in range(rows)],
        dtype=float,
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 1.0,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def test_main_runs_complete_production_end_to_end(
    monkeypatch,
    capsys,
):
    data = _rising_data()

    monkeypatch.setattr(
        run_production_end_to_end,
        "load_production_market_data",
        lambda path: data,
    )

    monkeypatch.setattr(
        run_production_end_to_end,
        "run_production_end_to_end",
        lambda data, results_dir, symbol, interval: {
            "end_to_end_ready": True,
            "decision": {
                "decision": "BUY",
                "trend": "UP",
            },
            "production_selection": {
                "stable_strategy": "momentum",
                "stability_score": 0.517268,
            },
            "display": {
                "entry_price": 2158.0,
                "stop_loss": 2136.42,
                "tp1": 2179.58,
                "tp2": 2201.16,
                "tp3": 2222.74,
            },
            "release_gate": {
                "release_ready": True,
            },
        },
    )

    run_production_end_to_end.main()

    output = capsys.readouterr().out

    assert "PRODUCTION END-TO-END SUCCESS" in output

    payload_text = output.split(
        "PRODUCTION END-TO-END SUCCESS"
    )[0].strip()

    payload = json.loads(payload_text)

    assert payload["end_to_end_ready"] is True
    assert payload["release_ready"] is True
    assert payload["decision"] == "BUY"
    assert payload["stable_strategy"] == "momentum"
    assert payload["stability_score"] == pytest.approx(
        0.517268
    )


def test_main_fails_when_end_to_end_is_not_ready(
    monkeypatch,
):
    data = _rising_data()

    monkeypatch.setattr(
        run_production_end_to_end,
        "load_production_market_data",
        lambda path: data,
    )

    monkeypatch.setattr(
        run_production_end_to_end,
        "run_production_end_to_end",
        lambda data, results_dir, symbol, interval: {
            "end_to_end_ready": False,
            "decision": {
                "decision": "NO TRADE",
                "trend": "DOWN",
            },
            "production_selection": {
                "stable_strategy": "momentum",
                "stability_score": 0.20,
            },
            "display": {
                "entry_price": None,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
            },
            "release_gate": {
                "release_ready": False,
            },
        },
    )

    with pytest.raises(
        RuntimeError,
        match="PRODUCTION END-TO-END FAILED",
    ):
        run_production_end_to_end.main()


def test_main_uses_production_configuration(
    monkeypatch,
):
    calls = {}

    monkeypatch.setattr(
        run_production_end_to_end,
        "load_production_market_data",
        lambda path: _rising_data(),
    )

    def fake_runner(
        data,
        results_dir,
        symbol,
        interval,
    ):
        calls["results_dir"] = results_dir
        calls["symbol"] = symbol
        calls["interval"] = interval

        return {
            "end_to_end_ready": True,
            "decision": {
                "decision": "NO TRADE",
                "trend": "DOWN",
            },
            "production_selection": {
                "stable_strategy": "momentum",
                "stability_score": 0.517268,
            },
            "display": {
                "entry_price": None,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
            },
            "release_gate": {
                "release_ready": True,
            },
        }

    monkeypatch.setattr(
        run_production_end_to_end,
        "run_production_end_to_end",
        fake_runner,
    )

    run_production_end_to_end.main()

    assert calls["results_dir"] == "results/production"
    assert calls["symbol"] == "XAUUSD"
    assert calls["interval"] == "1d"
