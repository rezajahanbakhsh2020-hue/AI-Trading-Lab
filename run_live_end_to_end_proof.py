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
from src.evaluation.live_execution_runtime import (
    LiveExecutionRuntime,
    ProductionRuntimeConfig,
)
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

    config = ProductionRuntimeConfig(
        symbol=SYMBOL,
        timeframe=INTERVAL,
    )

    runtime = LiveExecutionRuntime(
        symbol=SYMBOL,
        interval=INTERVAL,
        limit=LIMIT,
        production_config=config,
    )

    runtime_res = runtime.run_once(
        publish=False,
        persist=True,
    )

    if runtime_res.get("blocked"):
        reason = runtime_res.get("reason", "UNKNOWN_BLOCKED_REASON")
        detail = runtime_res.get("detail", "No promoted candidate or runtime authorization available.")
        raise RuntimeError(
            f"Live execution runtime blocked: {reason} - {detail}"
        )

    record = runtime_res.get("record") or {}
    publication = runtime_res.get("publication") or {}

    display_dict = {
        "decision": runtime_res.get("decision"),
        "entry_price": publication.get("entry"),
        "stop_loss": publication.get("stop_loss"),
        "tp1": publication.get("tp1"),
        "tp2": publication.get("tp2"),
        "tp3": publication.get("tp3"),
        "stable_strategy": runtime_res.get("strategy"),
        "stability_score": runtime_res.get("stability_score"),
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "trend": record.get("trend"),
        "signal_label": record.get("signal_label"),
    }

    overlay = build_live_trade_overlay(
        data,
        display_dict,
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
        "stable_strategy": str(runtime_res.get("strategy")) if runtime_res.get("strategy") is not None else None,
        "stability_score": float(runtime_res.get("stability_score")) if runtime_res.get("stability_score") is not None else None,
        "decision": runtime_res.get("decision"),
        "signal": record.get("signal"),
        "signal_label": record.get("signal_label"),
        "trend": record.get("trend"),
        "entry": overlay["levels"].get("entry"),
        "stop_loss": overlay["levels"].get("stop_loss"),
        "tp1": overlay["levels"].get("tp1"),
        "tp2": overlay["levels"].get("tp2"),
        "tp3": overlay["levels"].get("tp3"),
        "live_mid": quote.get("mid"),
        "market_state": quote.get("marketState"),
        "quote_stale": quote.get("stale"),
        "quote_age_seconds": quote.get("quoteAgeSeconds"),
        "html_path": str(OUTPUT_HTML),
        "end_to_end_passed": True,
        "candidate_id": runtime_res.get("candidate_id"),
        "evidence_id": runtime_res.get("evidence_id"),
        "decision_id": record.get("decision_id"),
        "runtime_authorization_fingerprint": runtime_res.get("runtime_authorization_fingerprint"),
        "promoted_artifact_fingerprint": runtime_res.get("promoted_artifact_fingerprint"),
        "governance_decision_fingerprint": runtime_res.get("governance_decision_fingerprint"),
        "campaign_selection_decision_fingerprint": runtime_res.get("campaign_selection_decision_fingerprint"),
        "canonical_live_decision_fingerprint": record.get("canonical_live_decision_fingerprint"),
        "context_fingerprint": runtime_res.get("context_fingerprint"),
        "evaluation_fingerprint": runtime_res.get("evaluation_fingerprint"),
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
