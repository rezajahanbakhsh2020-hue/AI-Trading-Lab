"""Headless live execution engine for Project 1 -> Project 2 trading intelligence pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, Optional

import pandas as pd

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
)
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.production_live_bridge import load_production_selection
from src.integration.project2_publisher import (
    Project2Publisher,
    build_contract_v1_payload,
)

logger = logging.getLogger(__name__)


def load_live_market_data(
    symbol: str = "XAUUSD",
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Fetch live market data for a target symbol and interval."""
    if symbol.upper() == "XAUUSD":
        data = fetch_xauusd_ohlc(interval=interval, limit=limit)
    else:
        # Fallback or extension for multi-symbol market data ingestion
        data = fetch_xauusd_ohlc(interval=interval, limit=limit)

    required = {"openTime", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing required live columns for {symbol}: " + ", ".join(sorted(missing))
        )

    data = data.copy()
    data["timestamp"] = pd.to_datetime(data["openTime"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if data.empty:
        raise ValueError(f"No valid live market data available for {symbol}.")

    return data.reset_index(drop=True)


class LiveExecutionRuntime:
    """Headless runtime orchestrator that drives signal evaluation and optional Project 2 delivery."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LIMIT,
        publisher: Optional[Project2Publisher] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.publisher = publisher or Project2Publisher()

    def run_once(
        self,
        publish: bool = True,
        skip_if_no_trade: bool = False,
    ) -> Dict[str, Any]:
        """Execute one full cycle: Ingestion -> Analysis -> Signal Artifact -> Project 2 Publish."""
        # 1. Fetch market data
        data = load_live_market_data(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.limit,
        )

        # 2. Load stable strategy selection
        selection = load_production_selection()
        stable_strategy = selection.get("stable_strategy")
        stability_score = selection.get("stability_score")

        if not stable_strategy or stability_score is None:
            raise ValueError("Valid production strategy selection not found.")

        # 3. Evaluate live runtime decision & trade levels
        runtime = build_live_runtime(
            data,
            stable_strategy=str(stable_strategy),
            stability_score=float(stability_score),
            symbol=self.symbol,
            interval=self.interval,
        )

        display = runtime.display
        decision = runtime.decision

        # 4. Construct canonical Contract v1.0 payload
        contract_payload = build_contract_v1_payload(
            symbol=self.symbol,
            interval=self.interval,
            decision=display["decision"],
            strategy=display["stable_strategy"],
            stability_score=display["stability_score"],
            signal_label=display["signal_label"],
            trend=display["trend"],
            entry_price=display["entry_price"],
            stop_loss=display["stop_loss"],
            tp1=display.get("tp1"),
            tp2=display.get("tp2"),
            tp3=display.get("tp3"),
            take_profit=display.get("take_profit"),
            timestamp=display.get("timestamp"),
        )

        # 5. Publish if enabled
        publish_result = None
        if publish:
            publish_result = self.publisher.publish(
                contract_payload,
                skip_if_no_trade=skip_if_no_trade,
            )

        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "decision": display["decision"],
            "strategy": display["stable_strategy"],
            "stability_score": display["stability_score"],
            "contract_payload": contract_payload,
            "publish_result": publish_result,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Live Execution Runtime")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Target instrument symbol")
    parser.add_argument("--interval", type=str, default="5m", help="Market data timeframe interval")
    parser.add_argument("--limit", type=int, default=100, help="Number of candles to fetch")
    parser.add_argument("--publish", action="store_true", help="Enable outbound publishing to Project 2")
    parser.add_argument("--skip-no-trade", action="store_true", help="Skip publishing when decision is NO TRADE")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        runtime = LiveExecutionRuntime(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
        )
        result = runtime.run_once(
            publish=args.publish,
            skip_if_no_trade=args.skip_no_trade,
        )

        logger.info("Execution complete for %s %s", result["symbol"], result["interval"])
        logger.info("Decision: %s | Strategy: %s (Stability: %.3f)",
                    result["decision"], result["strategy"], result["stability_score"])

        if result["publish_result"]:
            logger.info("Publish Result: %s", result["publish_result"])

    except Exception as exc:
        logger.error("Headless live execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
