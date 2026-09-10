from pathlib import Path

import pytest

from run_live_visual_proof import (
    OUTPUT_PATH,
    run_live_visual_proof,
)


def test_live_visual_proof_creates_real_html():
    result = run_live_visual_proof()

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
    assert "Fast MA" in html
    assert "Slow MA" in html


def test_live_visual_proof_uses_production_selection():
    result = run_live_visual_proof()

    assert result["stable_strategy"]
    assert result["production_source"]
    assert result["stability_score"] >= 0.0
    assert result["stability_score"] <= 1.0
