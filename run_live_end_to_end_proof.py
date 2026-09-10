from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from app_live_trade_display import _build_chart
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.production_live_bridge import load_production_selection
from src.visualization.live_trade_overlay import build_live_trade_overlay


OUTPUT_DIR = Path("results/live")
OUTPUT_JSON = OUTPUT_DIR / "live_end_to_end_proof.json"
OUTPUT_HTML = OUTPUT_DIR / "live_end_to_end_proof.html"

SYMBOL = "XAUUSD"
INTERVAL = DEFAULT_INTERVAL
LIMIT = DEFAULT_LIMIT


def run_live_end_to_end_proof() -> dict[str, Any]:
    """Run the complete real-data live trading proof."""

    data = fetch_xauusd_ohlc(
        interval=INTERVAL,
        limit=LIMIT,
    )

    quote = fetch_xauusd_quote()

    selection = load_production_selection()

    stable_strategy = selection.get("stable_strategy")
    stability_score = selection.get("stability_score")

    if not stable_strategy:
        raise ValueError(
            "Production selection does not contain a stable strategy."
        )

    if stability_score is None:
        raise ValueError(
            "Production selection does not contain a stability score."
        )

    runtime = build_live_runtime(
        data,
        stable_strategy=str(stable_strategy),
        stability_score=float(stability_score),
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    overlay = build_live_trade_overlay(
        data,
        runtime.display,
    )

    figure = _build_chart(
        data,
        overlay,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        str(OUTPUT_HTML),
        include_plotlyjs=True,
        full_html=True,
    )

    if not OUTPUT_HTML.is_file():
        raise RuntimeError(
            "Live end-to-end HTML was not created."
        )

    if OUTPUT_HTML.stat().st_size <= 0:
        raise RuntimeError(
            "Live end-to-end HTML is empty."
        )

    result = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "candle_count": len(data),
        "stable_strategy": str(stable_strategy),
        "stability_score": float(stability_score),
        "decision": runtime.decision["decision"],
        "signal": runtime.decision["signal"],
        "signal_label": runtime.decision["signal_label"],
        "trend": runtime.decision["trend"],
        "entry": overlay["levels"].get("entry"),
        "stop_loss": overlay["levels"].get("stop_loss"),
        "tp1": overlay["levels"].get("tp1"),
        "tp2": overlay["levels"].get("tp2"),
        "tp3": overlay["levels"].get("tp3"),
        "live_mid": quote.get("mid"),
        "market_state": quote.get("marketState"),
        "quote_stale": quote.get("stale"),
        "quote_age_seconds": quote.get(
            "quoteAgeSeconds"
        ),
        "production_source": selection.get(
            "source_path"
        ),
        "html_path": str(OUTPUT_HTML),
        "end_to_end_passed": True,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            result,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return result


if __name__ == "__main__":
    result = run_live_end_to_end_proof()

    print("=== AI-TRADING-LAB LIVE END-TO-END PROOF ===")

    for key, value in result.items():
        print(f"{key}: {value}")
