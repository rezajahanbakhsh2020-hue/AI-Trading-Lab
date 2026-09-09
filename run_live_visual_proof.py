from __future__ import annotations

from pathlib import Path

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from app_live_proof_history import build_live_history_record
from live_snapshot import save_live_snapshot
from src.evaluation.live_explanation import build_live_explanation
from src.visualization.live_proof_chart import (
    build_live_proof_chart,
)

OUTPUT_PATH = Path(
    "results/live/live_proof_visual.html"
)


def run_live_visual_proof() -> dict:
    """Run a real XAU/USD visual proof and save an HTML chart."""

    data = fetch_xauusd_ohlc(
        interval=DEFAULT_INTERVAL,
        limit=DEFAULT_LIMIT,
    )

    quote = fetch_xauusd_quote()

    snapshot = build_live_history_record(
        data,
        quote,
        interval=DEFAULT_INTERVAL,
    )

    save_live_snapshot(snapshot)

    explanation = build_live_explanation(
        snapshot
    )

    figure = build_live_proof_chart(
        data,
        snapshot,
    )

    figure.write_html(
        str(OUTPUT_PATH),
        include_plotlyjs=True,
        full_html=True,
    )

    if not OUTPUT_PATH.is_file():
        raise RuntimeError(
            "Visual proof HTML was not created."
        )

    if OUTPUT_PATH.stat().st_size <= 0:
        raise RuntimeError(
            "Visual proof HTML is empty."
        )

    return {
        "symbol": snapshot.get("symbol"),
        "interval": snapshot.get("interval"),
        "signal": snapshot.get("signal"),
        "signal_label": snapshot.get(
            "signal_label"
        ),
        "trend": snapshot.get("trend"),
        "entry_price": snapshot.get(
            "entry_price"
        ),
        "stop_loss": snapshot.get(
            "stop_loss"
        ),
        "take_profit": snapshot.get(
            "take_profit"
        ),
        "timestamp": snapshot.get(
            "timestamp"
        ),
        "candle_count": snapshot.get(
            "candle_count"
        ),
        "market_state": snapshot.get(
            "market_state"
        ),
        "quote_stale": snapshot.get(
            "quote_stale"
        ),
        "human_text": explanation.get(
            "human_text"
        ),
        "output_path": str(OUTPUT_PATH),
    }


if __name__ == "__main__":
    result = run_live_visual_proof()

    print("=== REAL XAU/USD VISUAL PROOF ===")
    for key, value in result.items():
        print(f"{key}: {value}")
