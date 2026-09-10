from __future__ import annotations

import pandas as pd

import run_live_end_to_end_proof as proof


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.to_datetime(
                [
                    "2026-09-10 10:00:00+00:00",
                    "2026-09-10 10:05:00+00:00",
                ]
            ),
            "open": [4400.0, 4401.0],
            "high": [4402.0, 4403.0],
            "low": [4399.0, 4400.0],
            "close": [4401.0, 4402.0],
        }
    )


def test_live_end_to_end_proof_connects_real_data_to_runtime(
    monkeypatch,
    tmp_path,
):
    data = _sample_data()

    monkeypatch.setattr(
        proof,
        "fetch_xauusd_ohlc",
        lambda interval, limit: data.copy(),
    )

    monkeypatch.setattr(
        proof,
        "fetch_xauusd_quote",
        lambda: {
            "symbol": "XAUUSD",
            "mid": 4402.0,
            "marketState": "OPEN",
            "stale": False,
            "quoteAgeSeconds": 0.0,
        },
    )

    monkeypatch.setattr(
        proof,
        "load_production_selection",
        lambda: {
            "stable_strategy": "momentum",
            "stability_score": 0.517268,
            "source_path": "results/production/latest.json",
        },
    )

    monkeypatch.setattr(
        proof,
        "build_live_runtime",
        lambda *args, **kwargs: type(
            "Runtime",
            (),
            {
                "decision": {
                    "decision": "BUY",
                    "signal": 1,
                    "signal_label": "BUY",
                    "trend": "UP",
                },
                "display": {
                    "decision": "BUY",
                },
            },
        )(),
    )

    monkeypatch.setattr(
        proof,
        "build_live_trade_overlay",
        lambda data, display: {
            "decision": "BUY",
            "levels": {
                "entry": 4402.0,
                "stop_loss": 4358.0,
                "tp1": 4446.0,
                "tp2": 4490.0,
                "tp3": 4534.0,
            },
            "stable_strategy": "momentum",
            "symbol": "XAUUSD",
            "interval": "5m",
        },
    )

    class FakeFigure:
        def write_html(
            self,
            path,
            include_plotlyjs,
            full_html,
        ):
            assert include_plotlyjs is True
            assert full_html is True
            path_obj = tmp_path / "proof.html"
            path_obj.write_text(
                "<html>LIVE END-TO-END PROOF</html>",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        proof,
        "_build_chart",
        lambda data, overlay: FakeFigure(),
    )

    monkeypatch.setattr(
        proof,
        "OUTPUT_DIR",
        tmp_path,
    )

    monkeypatch.setattr(
        proof,
        "OUTPUT_JSON",
        tmp_path / "proof.json",
    )

    monkeypatch.setattr(
        proof,
        "OUTPUT_HTML",
        tmp_path / "proof.html",
    )

    result = proof.run_live_end_to_end_proof()

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"
    assert result["stable_strategy"] == "momentum"
    assert result["decision"] == "BUY"
    assert result["trend"] == "UP"
    assert result["entry"] == 4402.0
    assert result["stop_loss"] == 4358.0
    assert result["tp1"] == 4446.0
    assert result["tp2"] == 4490.0
    assert result["tp3"] == 4534.0
    assert result["quote_stale"] is False
    assert result["end_to_end_passed"] is True
    assert proof.OUTPUT_HTML.is_file()
    assert proof.OUTPUT_JSON.is_file()
