"""Headless live execution engine for Project 1 -> Project 2 trading intelligence pipeline."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
)
from src.evaluation.live_decision_record import build_live_decision_record
from src.evaluation.live_decision_store import append_live_decision_to_store
from src.evaluation.live_production_decision import (
    Direction,
    ProductionDecision,
    ProductionIntelligencePublication,
    ProductionSignal,
    PromotedCandidateArtifact,
    calculate_production_risk_levels,
    evaluate_production_decision,
    validate_production_scope,
)
from src.evaluation.live_publication_store import append_publication_record
from src.evaluation.live_runtime import build_live_runtime
from src.evaluation.production_live_bridge import load_production_selection
from src.evaluation.research_store import (
    DEFAULT_RESEARCH_DIR,
    PromotionEligibilityError,
    PromotionIntegrityError,
    PromotionUnavailable,
    resolve_promoted_candidate,
)
from src.integration.project2_publisher import (
    Project2Publisher,
)

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("results/live/decision_history.json")
DEFAULT_SNAPSHOT_PATH = Path("results/live/latest_execution.json")


@dataclass(frozen=True)
class ProductionRuntimeConfig:
    """Explicit production runtime configuration. Never establishes promotion."""

    symbol: str
    timeframe: str
    candidate_id: Optional[str] = None
    strategy_id: Optional[str] = None
    strategy_version: Optional[str] = None
    research_dir: Path = DEFAULT_RESEARCH_DIR

    def __post_init__(self) -> None:
        if not self.symbol or not str(self.symbol).strip():
            raise ValueError("symbol must be a non-empty string.")
        if not self.timeframe or not str(self.timeframe).strip():
            raise ValueError("timeframe must be a non-empty string.")
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "timeframe", str(self.timeframe).strip())
        if self.candidate_id is not None:
            cid = str(self.candidate_id).strip()
            object.__setattr__(self, "candidate_id", cid if cid else None)
        if self.strategy_id is not None:
            sid = str(self.strategy_id).strip()
            object.__setattr__(self, "strategy_id", sid if sid else None)
        if self.strategy_version is not None:
            ver = str(self.strategy_version).strip()
            object.__setattr__(self, "strategy_version", ver if ver else None)
        object.__setattr__(self, "research_dir", Path(self.research_dir))

    @classmethod
    def from_runtime(
        cls,
        *,
        symbol: str,
        timeframe: str,
        selection: Optional[Dict[str, Any]] = None,
        research_dir: Path | str = DEFAULT_RESEARCH_DIR,
    ) -> "ProductionRuntimeConfig":
        selection = selection or {}
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            candidate_id=selection.get("candidate_id"),
            strategy_id=selection.get("strategy_id") or selection.get("stable_strategy") or selection.get("strategy"),
            strategy_version=selection.get("strategy_version"),
            research_dir=Path(research_dir),
        )


@dataclass(frozen=True)
class ProductionBlocked:
    """Fail-closed production result when an authoritative promoted candidate cannot be used."""

    reason: str
    detail: str
    candidate_id: Optional[str] = None
    strategy_id: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": True,
            "reason": self.reason,
            "detail": self.detail,
            "candidate_id": self.candidate_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
        }


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


def resolve_authoritative_promoted_candidate(
    config: ProductionRuntimeConfig,
) -> PromotedCandidateArtifact | ProductionBlocked:
    """Resolve a persisted promoted candidate. Never manufactures promotion or substitutes a default."""
    if config.candidate_id is None and config.strategy_id is None:
        return ProductionBlocked(
            reason="PromotionUnavailable",
            detail="Production configuration is missing candidate_id and strategy_id.",
            symbol=config.symbol,
            timeframe=config.timeframe,
        )
    try:
        promoted = resolve_promoted_candidate(
            candidate_id=config.candidate_id,
            strategy_id=config.strategy_id,
            strategy_version=config.strategy_version,
            symbol=config.symbol,
            timeframe=config.timeframe,
            base_dir=config.research_dir,
        )
    except PromotionIntegrityError as exc:
        return ProductionBlocked(
            reason="PromotionIntegrityError",
            detail=str(exc),
            candidate_id=config.candidate_id,
            strategy_id=config.strategy_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
        )
    except PromotionEligibilityError as exc:
        return ProductionBlocked(
            reason="PromotionEligibilityError",
            detail=str(exc),
            candidate_id=config.candidate_id,
            strategy_id=config.strategy_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
        )
    except PromotionUnavailable as exc:
        return ProductionBlocked(
            reason="PromotionUnavailable",
            detail=str(exc),
            candidate_id=config.candidate_id,
            strategy_id=config.strategy_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
        )

    if promoted is None:
        return ProductionBlocked(
            reason="PromotionUnavailable",
            detail=(
                f"No persisted promoted candidate matches candidate_id="
                f"{config.candidate_id!r} strategy_id={config.strategy_id!r}."
            ),
            candidate_id=config.candidate_id,
            strategy_id=config.strategy_id,
            symbol=config.symbol,
            timeframe=config.timeframe,
        )

    try:
        validate_production_scope(
            promoted,
            current_symbol=config.symbol,
            current_timeframe=config.timeframe,
            current_strategy_id=config.strategy_id,
            current_strategy_version=config.strategy_version,
            current_candidate_id=config.candidate_id,
        )
    except ValueError as exc:
        return ProductionBlocked(
            reason="PromotionEligibilityError",
            detail=str(exc),
            candidate_id=promoted.candidate_id,
            strategy_id=promoted.strategy_name,
            symbol=config.symbol,
            timeframe=config.timeframe,
        )
    return promoted


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
        research_dir: Path | str = DEFAULT_RESEARCH_DIR,
        production_config: Optional[ProductionRuntimeConfig] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.publisher = publisher or Project2Publisher(max_age_seconds=int(max_age_seconds))
        self.store_path = Path(store_path)
        self.snapshot_path = Path(snapshot_path)
        self.max_age_seconds = max_age_seconds
        self.research_dir = Path(research_dir)
        self.production_config = production_config

    def _blocked_result(
        self,
        blocked: ProductionBlocked,
        *,
        persist: bool,
        publish: bool,
        skip_if_no_trade: bool,
        reference_now: datetime.datetime,
    ) -> Dict[str, Any]:
        """Return an explicit production-blocked result without manufacturing lineage."""
        now_iso = reference_now.isoformat()
        execution_result = {
            "blocked": True,
            "reason": blocked.reason,
            "detail": blocked.detail,
            "symbol": self.symbol,
            "interval": self.interval,
            "decision": "NO TRADE",
            "strategy": blocked.strategy_id,
            "stability_score": None,
            "record": None,
            "publication": None,
            "contract_payload": None,
            "publish_result": None,
            "candidate_id": blocked.candidate_id,
            "blocked_state": blocked.as_dict(),
            "timestamp": now_iso,
        }
        if persist:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_path.write_text(
                json.dumps(execution_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if publish:
            execution_result["publish_result"] = {
                "status": "SKIPPED_BLOCKED",
                "published": False,
                "reason": blocked.reason,
                "detail": blocked.detail,
                "skip_if_no_trade": skip_if_no_trade,
            }
        return execution_result

    def run_once(
        self,
        publish: bool = True,
        skip_if_no_trade: bool = False,
        persist: bool = True,
        reference_now: Optional[datetime.datetime] = None,
        max_age_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute one full cycle: resolve persisted promotion -> market data -> decision -> persist -> publish."""
        max_age = max_age_seconds if max_age_seconds is not None else self.max_age_seconds
        ref_now = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

        if self.production_config is not None:
            config = self.production_config
            selection: Dict[str, Any] = {
                "candidate_id": config.candidate_id,
                "strategy_id": config.strategy_id,
                "stable_strategy": config.strategy_id,
                "strategy_version": config.strategy_version,
            }
            try:
                extra = load_production_selection()
            except Exception:
                extra = {}
            if isinstance(extra, dict) and extra.get("stability_score") is not None:
                selection["stability_score"] = extra["stability_score"]
        else:
            selection = load_production_selection()
            config = ProductionRuntimeConfig.from_runtime(
                symbol=self.symbol,
                timeframe=self.interval,
                selection=selection,
                research_dir=self.research_dir,
            )

        resolved = resolve_authoritative_promoted_candidate(config)
        if isinstance(resolved, ProductionBlocked):
            return self._blocked_result(
                resolved,
                persist=persist,
                publish=publish,
                skip_if_no_trade=skip_if_no_trade,
                reference_now=ref_now,
            )

        candidate = resolved
        stable_strategy = candidate.strategy_name
        raw_score = selection.get("stability_score")
        if raw_score is None:
            raw_score = selection.get("confidence")
        stability_score = float(raw_score) if raw_score is not None else None

        # 1. Fetch market data only after authoritative promotion resolution
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

        now_iso = ref_now.isoformat()

        # 4. Enforce freshness boundary BEFORE invoking authoritative trading decision generator
        if not freshness["fresh"]:
            candle_iso = freshness["candle_timestamp"] or now_iso
            decision = ProductionDecision(
                candidate_id=candidate.candidate_id,
                evidence_id=candidate.evidence.evidence_id,
                experiment_fingerprint=candidate.evidence.experiment_fingerprint,
                symbol=self.symbol,
                timeframe=self.interval,
                decision_timestamp=now_iso,
                market_timestamp=candle_iso,
                direction=Direction.NO_TRADE,
                reason=freshness["reason"],
                entry_price=None,
                invalidation_condition=None,
                confidence=stability_score,
                parameters=candidate.parameters,
            )
            display = {
                "symbol": self.symbol,
                "interval": self.interval,
                "decision": "NO TRADE",
                "reason": freshness["reason"],
                "stable_strategy": str(stable_strategy),
                "stability_score": stability_score,
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
                "timestamp": candle_iso,
                "quote_stale": True,
                "quote_age_seconds": freshness["age_seconds"],
            }
        else:
            decision = evaluate_production_decision(
                candidate=candidate,
                data=data,
                reference_now=ref_now,
                max_age_seconds=max_age,
            )
            if stability_score is not None:
                runtime = build_live_runtime(
                    data,
                    stable_strategy=str(stable_strategy),
                    stability_score=float(stability_score),
                    symbol=self.symbol,
                    interval=self.interval,
                )
                display = dict(runtime.display)
            else:
                display = {
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "decision": decision.direction.value,
                    "reason": decision.reason,
                    "stable_strategy": str(stable_strategy),
                    "stability_score": None,
                    "strategy_supported": str(stable_strategy) == "momentum",
                    "signal": 1 if decision.direction == Direction.BUY else 0,
                    "signal_label": decision.direction.value,
                    "trend": "UP" if decision.direction == Direction.BUY else "NEUTRAL",
                    "entry_price": decision.entry_price,
                    "timestamp": decision.market_timestamp,
                }
            display["quote_stale"] = False
            display["quote_age_seconds"] = freshness["age_seconds"]

        # Derive ProductionSignal & ProductionRiskLevels from ProductionDecision
        signal = ProductionSignal.from_decision(decision)
        risk = calculate_production_risk_levels(decision, candidate)

        # Derive canonical ProductionIntelligencePublication
        publication = ProductionIntelligencePublication.from_artifacts(
            decision=decision,
            signal=signal,
            risk=risk,
            candidate=candidate,
            confidence=stability_score,
        )

        # 6. Construct decision record & persist to store if enabled
        record = build_live_decision_record(display)
        record["decision_id"] = decision.decision_id
        record["signal_id"] = signal.signal_id

        if persist:
            append_live_decision_to_store(record, self.store_path)
            pub_store_path = self.store_path.parent / "publication_history.json"
            append_publication_record(publication.as_dict(), pub_store_path)

        contract_payload = publication.to_contract_v1_payload()

        # 7. Publish if enabled
        publish_result = None
        if publish:
            publish_result = self.publisher.publish(
                publication,
                skip_if_no_trade=skip_if_no_trade,
            )

        execution_result = {
            "blocked": False,
            "symbol": self.symbol,
            "interval": self.interval,
            "decision": display["decision"],
            "strategy": display["stable_strategy"],
            "stability_score": display["stability_score"],
            "candidate_id": candidate.candidate_id,
            "evidence_id": candidate.evidence.evidence_id,
            "research_fingerprint": candidate.evidence.experiment_fingerprint,
            "strategy_version": candidate.strategy_version,
            "record": record,
            "publication": publication.as_dict(),
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
