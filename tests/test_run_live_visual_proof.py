from pathlib import Path

import pandas as pd

import run_live_visual_proof


def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "openTime": pd.date_range(
                "2026-01-01",
                periods=60,
                freq="5min",
                tz="UTC",
            ),
            "open": [2500.0 + i for i in range(60)],
            "high": [2501.0 + i for i in range(60)],
            "low": [2499.0 + i for i in range(60)],
            "close": [2500.5 + i for i in range(60)],
        }
    )


def sample_quote() -> dict:
    return {
        "symbol": "XAUUSD",
        "mid": 2559.5,
        "bid": 2559.4,
        "ask": 2559.6,
        "marketState": "open",
        "stale": False,
        "quoteAgeSeconds": 0,
    }


def test_live_visual_proof_creates_real_html(tmp_path, monkeypatch):
    output_path = tmp_path / "live_proof_visual.html"

    monkeypatch.setattr(
        run_live_visual_proof,
        "fetch_xauusd_ohlc",
        lambda interval, limit: sample_data(),
    )
    monkeypatch.setattr(
        run_live_visual_proof,
        "fetch_xauusd_quote",
        lambda: sample_quote(),
    )
    monkeypatch.setattr(
        run_live_visual_proof,
        "OUTPUT_PATH",
        output_path,
    )

    result = run_live_visual_proof.run_live_visual_proof()

    assert result["symbol"] == "XAUUSD"
    assert result["interval"]
    assert result["signal"] in {0, 1}
    assert result["signal_label"] in {"BUY", "NO TRADE"}
    assert result["trend"] in {
        "UP",
        "DOWN",
        "INSUFFICIENT DATA",
    }
    assert result["candle_count"] > 0
    assert result["timestamp"]
    assert result["market_state"]
    assert isinstance(result["quote_stale"], bool)
    assert result["human_text"]

    result_path = Path(result["output_path"])

    assert result_path == output_path
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_live_visual_proof_html_contains_visual_elements(
    tmp_path,
    monkeypatch,
):
    output_path = tmp_path / "live_proof_visual.html"

    monkeypatch.setattr(
        run_live_visual_proof,
        "fetch_xauusd_ohlc",
        lambda interval, limit: sample_data(),
    )
    monkeypatch.setattr(
        run_live_visual_proof,
        "fetch_xauusd_quote",
        lambda: sample_quote(),
    )
    monkeypatch.setattr(
        run_live_visual_proof,
        "OUTPUT_PATH",
        output_path,
    )

    run_live_visual_proof.run_live_visual_proof()

    assert output_path.exists()

    html = output_path.read_text(encoding="utf-8")

    assert "plotly" in html.lower()
    assert "XAU/USD" in html
    assert "Fast MA" in html
    assert "Slow MA" in html
