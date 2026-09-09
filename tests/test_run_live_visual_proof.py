import pandas as pd

import src.run_live_visual_proof as run_live_visual_proof


def sample_data():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2025-01-01",
                periods=10,
                freq="5min",
            ),
            "open": [
                4390.0,
                4391.0,
                4392.0,
                4393.0,
                4394.0,
                4395.0,
                4396.0,
                4397.0,
                4398.0,
                4399.0,
            ],
            "high": [
                4391.0,
                4392.0,
                4393.0,
                4394.0,
                4395.0,
                4396.0,
                4397.0,
                4398.0,
                4399.0,
                4400.0,
            ],
            "low": [
                4389.0,
                4390.0,
                4391.0,
                4392.0,
                4393.0,
                4394.0,
                4395.0,
                4396.0,
                4397.0,
                4398.0,
            ],
            "close": [
                4390.5,
                4391.5,
                4392.5,
                4393.5,
                4394.5,
                4395.5,
                4396.5,
                4397.5,
                4398.5,
                4399.5,
            ],
        }
    )


def sample_quote():
    return {
        "symbol": "XAUUSD",
        "bid": 4399.4,
        "ask": 4399.6,
        "timestamp": "2025-01-01T00:45:00+00:00",
    }


def test_live_visual_proof_returns_expected_result(
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

    result = run_live_visual_proof.run_live_visual_proof()

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == run_live_visual_proof.DEFAULT_INTERVAL
    assert result["signal"] in (0, 1)
    assert result["signal_label"] in {"BUY", "NO TRADE"}
    assert result["trend"] in {"UP", "DOWN", "FLAT"}
    assert result["entry_price"] is not None
    assert result["stop_loss"] is not None
    assert result["take_profit"] is not None
    assert result["candle_count"] == 10
    assert result["output_path"] == str(output_path)
    assert output_path.exists()


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
    assert "Fast MA" in html
    assert "Slow MA" in html
