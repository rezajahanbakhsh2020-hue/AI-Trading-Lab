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


def validate_market_data_freshness(
    data: pd.DataFrame,
    max_age_seconds: float = 300.0,
    reference_now: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Validate event-time provenance and freshness of live market data."""
    if reference_now is None:
        now_dt = datetime.datetime.now(datetime.timezone.utc)
    else:
        now_dt = reference_now
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)

    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        return {
            "fresh": False,
            "stale": True,
            "reason": "missing_market_data",
            "age_seconds": None,
            "candle_timestamp": None,
        }

    if "timestamp" not in data.columns:
        return {
            "fresh": False,
            "stale": True,
            "reason": "missing_timestamp_column",
            "age_seconds": None,
            "candle_timestamp": None,
        }

    latest_ts = data["timestamp"].iloc[-1]
    if pd.isna(latest_ts):
        return {
            "fresh": False,
            "stale": True,
            "reason": "invalid_candle_timestamp",
            "age_seconds": None,
            "candle_timestamp": None,
        }

    if isinstance(latest_ts, pd.Timestamp):
        latest_dt = latest_ts.to_pydatetime()
    elif isinstance(latest_ts, datetime.datetime):
        latest_dt = latest_ts
    else:
        try:
            latest_dt = datetime.datetime.fromisoformat(str(latest_ts).replace("Z", "+00:00"))
        except Exception:
            return {
                "fresh": False,
                "stale": True,
                "reason": "invalid_candle_timestamp",
                "age_seconds": None,
                "candle_timestamp": str(latest_ts),
            }

    if latest_dt.tzinfo is None:
        latest_dt = latest_dt.replace(tzinfo=datetime.timezone.utc)
    else:
        latest_dt = latest_dt.astimezone(datetime.timezone.utc)

    age_seconds = (now_dt - latest_dt).total_seconds()
    candle_iso = latest_dt.isoformat()

    if age_seconds < 0:
        return {
            "fresh": False,
            "stale": True,
            "reason": "future_candle_timestamp",
            "age_seconds": age_seconds,
            "candle_timestamp": candle_iso,
        }

    if age_seconds > max_age_seconds:
        return {
            "fresh": False,
            "stale": True,
            "reason": "stale_market_data",
            "age_seconds": age_seconds,
            "candle_timestamp": candle_iso,
        }

    return {
        "fresh": True,
        "stale": False,
        "reason": "fresh",
        "age_seconds": age_seconds,
        "candle_timestamp": candle_iso,
    }


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
        max_age_seconds: float = 300.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.publisher = publisher or Project2Publisher(max_age_seconds=int(max_age_seconds))
        self.store_path = Path(store_path)
        self.snapshot_path = Path(snapshot_path)
        self.max_age_seconds = max_age_seconds

    def run_once(
        self,
        publish: bool = True,
        skip_if_no_trade: bool = False,
        persist: bool = True,
        reference_now: Optional[datetime.datetime] = None,
        max_age_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one full cycle: Ingestion -> Analysis -> Persistence -> Signal Artifact -> Project 2 Publish."""
        max_age = max_age_seconds if max_age_seconds is not None else self.max_age_seconds
        ref_now = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

        # 1. Fetch market data
        data = load_live_market_data(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.limit,
        )

        # 2. Evaluate market data freshness safety boundary
        freshness = validate_market_data_freshness(
            data=data,
            max_age_seconds=max_age,
            reference_now=ref_now,
        )

        # 3. Load stable strategy selection
        selection = load_production_selection()
        stable_strategy = selection.get("stable_strategy")
        stability_score = selection.get("stability_score")

        if not stable_strategy or stability_score is None:
            raise ValueError("Valid production strategy selection not found.")

        # 4. Enforce freshness boundary BEFORE invoking authoritative trading decision generator
        if not freshness["fresh"]:
            display = {
                "symbol": self.symbol,
                "interval": self.interval,
                "decision": "NO TRADE",
                "reason": freshness["reason"],
                "stable_strategy": str(stable_strategy),
                "stability_score": float(stability_score),
                "strategy_supported": str(stable_strategy) == "momentum",
                "signal": 0,
                "signal_label": "NO TRADE",
                "trend": "NEUTRAL",
                "momentum": None,
                "entry_price": None,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "take_profit": None,
                "risk_distance": None,
                "risk_reward_ratio": None,
                "risk_reward_tp1": None,
                "risk_reward_tp2": None,
                "risk_reward_tp3": None,
                "stop_loss_pct": None,
                "take_profit_pct": None,
                "tp1_multiplier": None,
                "tp2_multiplier": None,
                "tp3_multiplier": None,
                "momentum_window": None,
                "fast_window": None,
                "slow_window": None,
                "timestamp": freshness["candle_timestamp"],
                "quote_stale": True,
                "quote_age_seconds": freshness["age_seconds"],
            }
        else:
            runtime = build_live_runtime(
                data,
                stable_strategy=str(stable_strategy),
                stability_score=float(stability_score),
                symbol=self.symbol,
                interval=self.interval,
            )
            display = dict(runtime.display)
            display["quote_stale"] = False
            display["quote_age_seconds"] = freshness["age_seconds"]

        # 6. Construct decision record & persist to store if enabled
        record = build_live_decision_record(display)

        if persist:
            append_live_decision_to_store(record, self.store_path)

        now_iso = ref_now.isoformat()
        candle_iso = freshness["candle_timestamp"] or display.get("timestamp") or now_iso

        # Event timestamp passed to contract payload must preserve the market observation timestamp (candle_iso)
        # so that downstream publisher staleness check (Project2Publisher.is_stale) evaluates observation freshness.
        contract_event_ts = candle_iso

        # 7. Construct canonical Contract v1.0 payload
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
            timestamp=contract_event_ts,
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
