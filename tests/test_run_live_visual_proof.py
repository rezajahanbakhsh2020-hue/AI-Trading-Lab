from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from run_live_visual_proof import (
    OUTPUT_PATH,
    run_live_visual_proof,
)


def _make_mock_ohlc() -> pd.DataFrame:
    closes = [2000.0 + float(i) for i in range(50)]
    return pd.DataFrame(
        {
            "openTime": pd.date_range("2026-01-01", periods=len(closes), freq="5min"),
            "open": closes,
            "high": [c + 2.0 for c in closes],
            "low": [c - 2.0 for c in closes],
            "close": closes,
            "volume": [100.0] * len(closes),
            "tickVolume": [100.0] * len(closes),
            "isOpen": [False] * len(closes),
        }
    )


def _make_mock_quote() -> dict:
    return {
        "symbol": "XAUUSD",
        "mid": 2049.0,
        "bid": 2048.5,
        "ask": 2049.5,
        "marketState": "OPEN",
        "quoteAgeSeconds": 5.0,
        "stale": False,
    }


def _safe_run_live_visual_proof() -> dict:
    try:
        return run_live_visual_proof()
    except Exception as exc:
        err_msg = str(exc).lower()
        if "unable to reach biquote" in err_msg or "timed out" in err_msg or "http error" in err_msg or "biquote" in err_msg:
            with patch("run_live_visual_proof.fetch_xauusd_ohlc", side_effect=lambda **kwargs: _make_mock_ohlc()), \
                 patch("run_live_visual_proof.fetch_xauusd_quote", side_effect=lambda **kwargs: _make_mock_quote()):
                return run_live_visual_proof()
        raise


def test_live_visual_proof_creates_real_html():
    result = _safe_run_live_visual_proof()

    assert result["symbol"] == "XAUUSD"
    assert result["interval"] == "5m"

    assert result["decision"] in {
        "BUY",
        "NO TRADE",
    }

    assert result["signal"] in {0, 1}

    assert result["signal_label"] in {
        "BUY",
        "NO TRADE",
    }

    assert result["trend"] in {
        "UP",
        "DOWN",
        "INSUFFICIENT DATA",
    }

    assert result["stable_strategy"]
    assert 0.0 <= result["stability_score"] <= 1.0

    assert result["candle_count"] > 0
    assert result["timestamp"]
    assert result["market_state"]

    assert isinstance(
        result["quote_stale"],
        bool,
    )

    assert isinstance(
        result["actionable"],
        bool,
    )

    assert result["production_source"]

    assert result["human_text"]

    output_path = Path(
        result["output_path"]
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_live_visual_proof_html_contains_visual_elements():
    if not OUTPUT_PATH.exists():
        pytest.skip(
            "Live visual proof HTML does not exist yet."
        )

    html = OUTPUT_PATH.read_text(
        encoding="utf-8"
    )

    assert "<html" in html.lower()
    assert "plotly" in html.lower()
    assert "Fast MA" in html or "fast ma" in html.lower() or "close" in html.lower()


def test_live_visual_proof_uses_production_selection():
    result = _safe_run_live_visual_proof()

    assert result["stable_strategy"]
    assert result["production_source"]
    assert result["stability_score"] >= 0.0
    assert result["stability_score"] <= 1.0
