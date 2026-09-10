from __future__ import annotations

import json

import pandas as pd
import pytest

import run_final_release


def _market_data(rows: int = 80) -> pd.DataFrame:
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


def _valid_result() -> dict:
    return {
        "end_to_end_ready": True,
        "decision": {
            "decision": "BUY",
        },
        "display": {
            "decision": "BUY",
        },
        "overlay": {
            "decision": "BUY",
            "levels": {
                "stop_loss": 1980.0,
                "entry": 2000.0,
                "tp1": 2020.0,
                "tp2": 2040.0,
                "tp3": 2060.0,
            },
        },
        "release_gate": {
            "release_ready": True,
        },
        "production_selection": {
            "stable_strategy": "momentum",
            "stability_score": 0.517268,
        },
    }


def test_main_reports_final_release_ready(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        run_final_release,
        "load_production_market_data",
        lambda path: _market_data(),
    )

    monkeypatch.setattr(
        run_final_release,
        "run_production_end_to_end",
        lambda data, results_dir, symbol, interval: (
            _valid_result()
        ),
    )

    run_final_release.main()

    output = capsys.readouterr().out

    assert "FINAL RELEASE READY" in output

    payload_text = output.split(
        "FINAL RELEASE READY"
    )[0].strip()

    payload = json.loads(payload_text)

    assert payload["final_release_ready"] is True
    assert payload["status"] == "RELEASE READY"
    assert payload["strategy"] == "momentum"
    assert payload["decision"] == "BUY"
    assert payload["stability_score"] == pytest.approx(
        0.517268
    )


def test_main_blocks_failed_release(
    monkeypatch,
):
    monkeypatch.setattr(
        run_final_release,
        "load_production_market_data",
        lambda path: _market_data(),
    )

    failed = _valid_result()
    failed["end_to_end_ready"] = False

    monkeypatch.setattr(
        run_final_release,
        "run_production_end_to_end",
        lambda data, results_dir, symbol, interval: (
            failed
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="FINAL RELEASE BLOCKED",
    ):
        run_final_release.main()


def test_main_uses_production_configuration(
    monkeypatch,
):
    calls = {}

    monkeypatch.setattr(
        run_final_release,
        "load_production_market_data",
        lambda path: _market_data(),
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
        return _valid_result()

    monkeypatch.setattr(
        run_final_release,
        "run_production_end_to_end",
        fake_runner,
    )

    run_final_release.main()

    assert calls["results_dir"] == "results/production"
    assert calls["symbol"] == "XAUUSD"
    assert calls["interval"] == "1d"
