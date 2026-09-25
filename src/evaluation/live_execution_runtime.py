"""Headless live execution engine for Project 1 -> Project 2 trading intelligence pipeline."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
)
from src.evaluation.live_decision_record import build_live_decision_record
from src.evaluation.live_decision_store import append_live_decision_to_store
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.production_live_bridge import load_production_selection
from src.integration.project2_publisher import (
    Project2Publisher,
    build_contract_v1_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("results/live/decision_history.json")
DEFAULT_SNAPSHOT_PATH = Path("results/live/latest_execution.json")


# Provider capability registry mapping canonical instrument symbols to live market data adapters
LIVE_DATA_PROVIDERS: Dict[str, Any] = {
    "XAUUSD": fetch_xauusd_ohlc,
}


def get_live_data_adapter(symbol: str) -> Any:
    """Resolve live market data adapter based on provider capabilities."""
    symbol_clean = str(symbol).strip().upper()
    adapter = LIVE_DATA_PROVIDERS.get(symbol_clean)
    if not adapter:
        supported = ", ".join(sorted(LIVE_DATA_PROVIDERS.keys()))
        raise ValueError(
            f"Unsupported instrument symbol '{symbol}'. "
            f"No live market data adapter is configured for '{symbol_clean}'. "
            f"Supported instruments: {supported}."
        )
    return adapter


def load_live_market_data(
    symbol: str = "XAUUSD",
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Fetch live market data for a target symbol and interval using registered provider adapters."""
    symbol_clean = str(symbol).strip().upper()
    adapter = get_live_data_adapter(symbol_clean)
    data = adapter(interval=interval, limit=limit)

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
    """Headless runtime orchestrator that drives signal evaluation, store persistence, and optional Project 2 delivery."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        interval: str = DEFAULT_INTERVAL,
        limit: int = DEFAULT_LIMIT,
        publisher: Optional[Project2Publisher] = None,
        store_path: Path | str = DEFAULT_STORE_PATH,
        snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.publisher = publisher or Project2Publisher()
        self.store_path = Path(store_path)
        self.snapshot_path = Path(snapshot_path)

    def run_once(
        self,
        publish: bool = True,
        skip_if_no_trade: bool = False,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Execute one full cycle: Ingestion -> Analysis -> Persistence -> Signal Artifact -> Project 2 Publish."""
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

        # 4. Construct decision record & persist to store if enabled
        record = build_live_decision_record(display)

        if persist:
            append_live_decision_to_store(record, self.store_path)

        # Event execution timestamp: use current timestamp for contract event publication freshness,
        # but encode candle timestamp into event identity if needed or pass now_iso for staleness tracking.
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        candle_iso = display.get("timestamp") or now_iso

        # 5. Construct canonical Contract v1.0 payload
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
            risk_reward_ratio=display.get("risk_reward_ratio"),
            timestamp=now_iso,
            candle_timestamp=candle_iso,
        )

        # 6. Publish if enabled
        publish_result = None
        if publish:
            publish_result = self.publisher.publish(
                contract_payload,
                skip_if_no_trade=skip_if_no_trade,
            )

        execution_result = {
            "symbol": self.symbol,
            "interval": self.interval,
            "decision": display["decision"],
            "strategy": display["stable_strategy"],
            "stability_score": display["stability_score"],
            "record": record,
            "contract_payload": contract_payload,
            "publish_result": publish_result,
        }

        # 7. Write latest snapshot state file
        if persist:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.write_text(
                json.dumps(execution_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return execution_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Live Execution Runtime")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Target instrument symbol")
    parser.add_argument("--interval", type=str, default="5m", help="Market data timeframe interval")
    parser.add_argument("--limit", type=int, default=100, help="Number of candles to fetch")
    parser.add_argument("--publish", action="store_true", help="Enable outbound publishing to Project 2")
    parser.add_argument("--skip-no-trade", action="store_true", help="Skip publishing when decision is NO TRADE")
    parser.add_argument("--no-persist", action="store_true", help="Disable history persistence")

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
            persist=not args.no_persist,
        )

        logger.info("Execution complete for %s %s", result["symbol"], result["interval"])
        logger.info("Decision: %s | Strategy: %s (Stability: %.3f)",
                    result["decision"], result["strategy"], result["stability_score"])

        pub_res = result.get("publish_result")
        if pub_res:
            status = pub_res.get("status")
            logger.info("Publish Result: %s", pub_res)
            if status == "REJECTED":
                logger.error("Publication REJECTED by Project 2 gateway: %s", pub_res.get("error"))
                sys.exit(3)
            elif status in ("FAILED", "MISCONFIGURED"):
                reason = pub_res.get("reason") or pub_res.get("error") or ""
                if "Missing" in reason or "configuration" in reason:
                    logger.error("Publication misconfigured: %s", reason)
                    sys.exit(2)
                else:
                    logger.error("Publication transport failed: %s", reason)
                    sys.exit(4)
            elif status == "TIMED_OUT":
                logger.error("Publication timed out: %s", pub_res.get("error"))
                sys.exit(4)

    except Exception as exc:
        logger.error("Headless live execution failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
