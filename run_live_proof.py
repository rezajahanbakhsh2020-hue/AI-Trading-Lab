"""Run an end-to-end proof of the live XAU/USD trading system."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from app_live_signal_trend_risk import (
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    build_signal_trend_risk_snapshot,
)
from live_snapshot import (
    DEFAULT_SNAPSHOT_PATH,
    build_live_snapshot,
    save_live_snapshot,
)
from live_signal import DEFAULT_MOMENTUM_WINDOW
from live_trend import DEFAULT_FAST_WINDOW, DEFAULT_SLOW_WINDOW


DEFAULT_SYMBOL = "XAUUSD"


def run_live_proof(
    *,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    momentum_window: int = DEFAULT_MOMENTUM_WINDOW,
    fast_window: int = DEFAULT_FAST_WINDOW,
    slow_window: int = DEFAULT_SLOW_WINDOW,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
    fetch_ohlc: Callable[..., Any] = fetch_xauusd_ohlc,
    fetch_quote: Callable[..., Any] = fetch_xauusd_quote,
) -> dict[str, Any]:
    """
    Execute the complete live-system proof.

    The function intentionally composes the existing project components
    instead of duplicating signal, trend, risk, or persistence logic.
    """

    if not symbol:
        raise ValueError("symbol must not be empty.")

    if not interval:
        raise ValueError("interval must not be empty.")

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be a positive integer.")

    if limit <= 0:
        raise ValueError("limit must be greater than zero.")

    data = fetch_ohlc(
        interval=interval,
        limit=limit,
    )

    if data is None:
        raise ValueError("OHLC data fetch returned None.")

    if getattr(data, "empty", False):
        raise ValueError("OHLC data fetch returned no candles.")

    quote = fetch_quote()

    if not isinstance(quote, dict):
        raise ValueError("quote fetch must return a dictionary.")

    system_output = build_signal_trend_risk_snapshot(
        data,
        momentum_window=momentum_window,
        fast_window=fast_window,
        slow_window=slow_window,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    snapshot = build_live_snapshot(
        system_output,
        symbol=symbol,
        interval=interval,
    )

    snapshot["market_state"] = str(
        quote.get("marketState", "UNKNOWN")
    )
    snapshot["quote_age_seconds"] = quote.get(
        "quoteAgeSeconds"
    )
    snapshot["quote_stale"] = bool(
        quote.get("stale", False)
    )
    snapshot["latest_bid"] = quote.get("bid")
    snapshot["latest_ask"] = quote.get("ask")
    snapshot["latest_mid"] = quote.get("mid")
    snapshot["candle_count"] = int(len(data))

    saved_path = save_live_snapshot(
        snapshot,
        snapshot_path,
    )

    return {
        "snapshot": snapshot,
        "saved_path": str(saved_path),
    }


def main() -> None:
    result = run_live_proof()

    snapshot = result["snapshot"]

    print("=== XAU/USD LIVE PROOF ===")
    print(f"Symbol: {snapshot['symbol']}")
    print(f"Interval: {snapshot['interval']}")
    print(f"Candles: {snapshot['candle_count']}")
    print(f"Signal: {snapshot['signal_label']}")
    print(f"Trend: {snapshot['trend']}")
    print(f"Entry: {snapshot['entry_price']}")
    print(f"Stop Loss: {snapshot['stop_loss']}")
    print(f"Take Profit: {snapshot['take_profit']}")
    print(f"Risk/Reward: {snapshot['risk_reward_ratio']}")
    print(f"Market: {snapshot['market_state']}")
    print(f"Quote stale: {snapshot['quote_stale']}")
    print(f"Quote age: {snapshot['quote_age_seconds']}")
    print(f"Candle timestamp: {snapshot['timestamp']}")
    print(f"Snapshot saved: {result['saved_path']}")


if __name__ == "__main__":
    main()
