from __future__ import annotations

from pathlib import Path

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from src.evaluation.live_explanation import build_live_explanation
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.production_live_bridge import (
    load_production_selection,
)
from live_snapshot import (
    build_live_snapshot,
    save_live_snapshot,
)
from src.visualization.live_proof_chart import build_live_proof_chart


OUTPUT_PATH = Path(
    "results/live/live_proof_visual.html"
)


def _load_live_production_selection() -> dict:
    try:
        return load_production_selection()
    except FileNotFoundError:
        return {
            "stable_strategy": "momentum",
            "stability_score": 0.517268,
            "source_path": "built-in production baseline",
        }


def _build_runtime_snapshot(
    data,
    quote,
    runtime,
) -> dict:
    decision = runtime.decision
    display = runtime.display

    system_output = {
        "signal": decision["signal"],
        "signal_label": decision["signal_label"],
        "trend": decision["trend"],
        "strategy": decision["stable_strategy"],
        "momentum": decision["momentum"],
        "entry_price": decision["entry_price"],
        "stop_loss": decision["stop_loss"],
        "take_profit": decision["take_profit"],
        "risk_reward_ratio": decision["risk_reward_ratio"],
        "stop_loss_pct": decision["stop_loss_pct"],
        "take_profit_pct": decision["take_profit_pct"],
        "momentum_window": decision["momentum_window"],
        "fast_window": decision["fast_window"],
        "slow_window": decision["slow_window"],
        "timestamp": decision["timestamp"],
    }

    snapshot = build_live_snapshot(
        system_output,
        symbol=decision["symbol"],
        interval=decision["interval"],
    )

    snapshot.update(
        {
            "decision": decision["decision"],
            "reason": decision["reason"],
            "stable_strategy": decision["stable_strategy"],
            "stability_score": decision["stability_score"],
            "min_stability_score": decision[
                "min_stability_score"
            ],
            "strategy_supported": decision[
                "strategy_supported"
            ],
            "market_state": str(
                quote.get("marketState", "UNKNOWN")
            ).upper(),
            "quote_stale": bool(
                quote.get("stale", False)
            ),
            "quote_age_seconds": quote.get(
                "quoteAgeSeconds"
            ),
            "bid": quote.get("bid"),
            "ask": quote.get("ask"),
            "mid": quote.get("mid"),
            "candle_count": int(len(data)),
            "actionable": bool(
                decision["decision"] == "BUY"
                and not quote.get("stale", False)
            ),
            "display_decision": display["decision"],
        }
    )

    return snapshot


def run_live_visual_proof() -> dict:
    """Run the complete real XAU/USD visual proof."""

    data = fetch_xauusd_ohlc(
        interval=DEFAULT_INTERVAL,
        limit=DEFAULT_LIMIT,
    )

    quote = fetch_xauusd_quote()

    selection = _load_live_production_selection()

    stable_strategy = selection.get(
        "stable_strategy"
    )
    stability_score = selection.get(
        "stability_score"
    )

    if not stable_strategy:
        raise ValueError(
            "Production selection does not contain "
            "a stable strategy."
        )

    if stability_score is None:
        raise ValueError(
            "Production selection does not contain "
            "a stability score."
        )

    runtime = build_live_runtime(
        data,
        stable_strategy=str(stable_strategy),
        stability_score=float(stability_score),
        symbol="XAUUSD",
        interval=DEFAULT_INTERVAL,
    )

    snapshot = _build_runtime_snapshot(
        data,
        quote,
        runtime,
    )

    save_live_snapshot(snapshot)

    explanation = build_live_explanation(
        snapshot
    )

    figure = build_live_proof_chart(
        data,
        snapshot,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
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
        "decision": snapshot.get("decision"),
        "signal": snapshot.get("signal"),
        "signal_label": snapshot.get(
            "signal_label"
        ),
        "trend": snapshot.get("trend"),
        "strategy": snapshot.get("strategy"),
        "stable_strategy": snapshot.get(
            "stable_strategy"
        ),
        "stability_score": snapshot.get(
            "stability_score"
        ),
        "entry_price": snapshot.get(
            "entry_price"
        ),
        "stop_loss": snapshot.get(
            "stop_loss"
        ),
        "take_profit": snapshot.get(
            "take_profit"
        ),
        "risk_reward_ratio": snapshot.get(
            "risk_reward_ratio"
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
        "actionable": snapshot.get(
            "actionable"
        ),
        "production_source": selection.get(
            "source_path"
        ),
        "human_text": explanation.get(
            "human_text"
        ),
        "output_path": str(OUTPUT_PATH),
    }


if __name__ == "__main__":
    result = run_live_visual_proof()

    print(
        "=== REAL XAU/USD PRODUCTION LIVE VISUAL PROOF ==="
    )

    for key, value in result.items():
        print(f"{key}: {value}")
