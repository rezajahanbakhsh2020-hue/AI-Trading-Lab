"""Headless live execution engine for Project 1 -> Project 2 trading intelligence pipeline."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app_live import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
)
from src.data.provider import (
    BiQuoteProvider,
    FunctionMarketDataProvider,
    MarketDataProvider,
    UnsupportedInstrumentError,
    resolve_provider_for_symbol,
    resolve_requested_symbol,
)
from src.evaluation.live_decision_record import build_live_decision_record
from src.evaluation.live_market_evaluation import (
    create_live_market_evaluation,
)
from src.evaluation.live_production_decision import (
    ProductionAuthorizationReceipt,
    ProductionIntelligencePublication,
    ProductionRuntimeAuthorizationError,
    PromotedCandidateArtifact,
    authorize_production_runtime,
    validate_production_scope,
)
from src.evaluation.live_runtime import (
    evaluate_authorized_live_runtime,
)
from src.evaluation.live_runtime_context import (
    create_authorized_runtime_context,
)
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
    candidate_id: str | None = None
    strategy_id: str | None = None
    strategy_version: str | None = None
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
        selection: dict[str, Any] | None = None,
        research_dir: Path | str = DEFAULT_RESEARCH_DIR,
    ) -> ProductionRuntimeConfig:
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
    candidate_id: str | None = None
    strategy_id: str | None = None
    symbol: str | None = None
    timeframe: str | None = None

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


# Provider capability registry mapping canonical instrument symbols to live market data adapters/providers
LIVE_DATA_PROVIDERS: dict[str, Any] = {
    "XAUUSD": BiQuoteProvider(supported_symbols=("XAUUSD",)),
}


def _get_registered_providers() -> list[MarketDataProvider]:
    """Derive registered MarketDataProvider instances dynamically from LIVE_DATA_PROVIDERS."""
    providers: list[MarketDataProvider] = []
    for symbol_key, val in LIVE_DATA_PROVIDERS.items():
        if isinstance(val, MarketDataProvider):
            providers.append(val)
        elif callable(val):
            providers.append(
                FunctionMarketDataProvider(symbol=symbol_key, ohlc_fetcher=val)
            )
    return providers


def resolve_live_provider(symbol: str) -> MarketDataProvider:
    """Resolve live market data provider based on provider capability truth without fallback substitution."""
    canonical_symbol = resolve_requested_symbol(symbol)
    providers = _get_registered_providers()
    provider = resolve_provider_for_symbol(
        symbol=canonical_symbol,
        available_providers=providers,
    )
    if provider is None:
        raise UnsupportedInstrumentError(
            f"No market-data provider supports {canonical_symbol}"
        )
    return provider


def get_live_data_adapter(symbol: str) -> Any:
    """Resolve live market data adapter based on provider capabilities."""
    provider = resolve_live_provider(symbol)
    canonical_symbol = resolve_requested_symbol(symbol)

    def adapter(interval: str = DEFAULT_INTERVAL, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
        return provider.get_candles(canonical_symbol, timeframe=interval, limit=limit)

    return adapter


def validate_and_prepare_market_snapshot(
    data: pd.DataFrame,
    symbol: str = "XAUUSD",
) -> pd.DataFrame:
    """Centralized validation and defensive copy helper for live OHLC market snapshots."""
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"Supplied market data for {symbol} must be a non-empty pandas DataFrame.")

    required = {"openTime", "open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"Missing required live columns for {symbol}: " + ", ".join(sorted(missing))
        )

    data_copy = data.copy()
    data_copy["timestamp"] = pd.to_datetime(data_copy["openTime"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")

    data_copy = data_copy.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if data_copy.empty:
        raise ValueError(f"No valid live market data available for {symbol}.")

    return data_copy.reset_index(drop=True)


def load_live_market_data(
    symbol: str = "XAUUSD",
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Fetch live market data for a target symbol and interval using registered provider adapters."""
    canonical_symbol = resolve_requested_symbol(symbol)
    provider = resolve_provider_for_symbol(
        symbol=canonical_symbol,
        available_providers=_get_registered_providers(),
    )
    if provider is None:
        raise UnsupportedInstrumentError(
            f"No market-data provider supports {canonical_symbol}"
        )

    data = provider.get_candles(
        symbol=canonical_symbol,
        timeframe=interval,
        limit=limit,
    )

    return validate_and_prepare_market_snapshot(data, symbol=canonical_symbol)


def resolve_authoritative_promoted_candidate(
    config: ProductionRuntimeConfig,
) -> PromotedCandidateArtifact | ProductionBlocked:
    """Resolve a persisted promoted candidate. Never manufactures promotion or substitutes a default."""
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
    reference_now: datetime.datetime | None = None,
) -> dict[str, Any]:
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
        publisher: Project2Publisher | None = None,
        store_path: Path | str = DEFAULT_STORE_PATH,
        snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
        max_age_seconds: float = 300.0,
        research_dir: Path | str = DEFAULT_RESEARCH_DIR,
        production_config: ProductionRuntimeConfig | None = None,
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
    ) -> dict[str, Any]:
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
            "market_data": None,
        }
        if persist:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snap_dict = {k: v for k, v in execution_result.items() if k != "market_data"}
            self.snapshot_path.write_text(
                json.dumps(snap_dict, ensure_ascii=False, indent=2),
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
        reference_now: datetime.datetime | None = None,
        max_age_seconds: float | None = None,
        market_data: pd.DataFrame | None = None,
        market_data_loader: Callable[[], pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        """Execute one full cycle: resolve persisted promotion -> authorize context -> evaluate market data -> canonical downstream runtime."""
        if market_data is not None and market_data_loader is not None:
            raise ValueError("Cannot supply both market_data and market_data_loader to run_once().")
        max_age = max_age_seconds if max_age_seconds is not None else self.max_age_seconds
        ref_now = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

        if self.production_config is not None:
            config = self.production_config
        else:
            config = ProductionRuntimeConfig(
                symbol=self.symbol,
                timeframe=self.interval,
                research_dir=self.research_dir,
            )

        # 1. Resolve candidate EXACTLY ONCE per cycle
        resolved = resolve_authoritative_promoted_candidate(config)
        if isinstance(resolved, ProductionBlocked):
            return self._blocked_result(
                resolved,
                persist=persist,
                publish=publish,
                skip_if_no_trade=skip_if_no_trade,
                reference_now=ref_now,
            )

        # 2. Authorize runtime & create receipt EXACTLY ONCE per cycle
        try:
            authorization = authorize_production_runtime(
                resolved,
                symbol=self.symbol,
                timeframe=self.interval,
                now=ref_now,
            )
            receipt = ProductionAuthorizationReceipt.from_authorization(
                authorization
            )
            context = create_authorized_runtime_context(
                candidate=resolved,
                authorization=authorization,
                authorization_receipt=receipt,
            )
        except ProductionRuntimeAuthorizationError as exc:
            blocked = ProductionBlocked(
                reason="ProductionRuntimeAuthorizationError",
                detail=str(exc),
                candidate_id=resolved.candidate_id,
                strategy_id=resolved.strategy_name,
                symbol=self.symbol,
                timeframe=self.interval,
            )
            return self._blocked_result(
                blocked,
                persist=persist,
                publish=publish,
                skip_if_no_trade=skip_if_no_trade,
                reference_now=ref_now,
            )

        candidate = context.candidate
        receipt = context.authorization_receipt
        stable_strategy = candidate.strategy_name
        stability_score = candidate.operational_stability_score

        # 3. Acquire market data snapshot STRICTLY AFTER candidate resolution and runtime authorization
        if market_data is not None:
            try:
                data = validate_and_prepare_market_snapshot(market_data, symbol=self.symbol)
            except Exception as exc:
                blocked = ProductionBlocked(
                    reason="MARKET_DATA_VALIDATION_FAILED",
                    detail=str(exc),
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_name,
                    symbol=self.symbol,
                    timeframe=self.interval,
                )
                return self._blocked_result(
                    blocked,
                    persist=persist,
                    publish=publish,
                    skip_if_no_trade=skip_if_no_trade,
                    reference_now=ref_now,
                )
        elif market_data_loader is not None:
            try:
                raw_data = market_data_loader()
            except Exception as exc:
                blocked = ProductionBlocked(
                    reason="MARKET_DATA_ACQUISITION_FAILED",
                    detail=str(exc),
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_name,
                    symbol=self.symbol,
                    timeframe=self.interval,
                )
                return self._blocked_result(
                    blocked,
                    persist=persist,
                    publish=publish,
                    skip_if_no_trade=skip_if_no_trade,
                    reference_now=ref_now,
                )
            try:
                data = validate_and_prepare_market_snapshot(raw_data, symbol=self.symbol)
            except Exception as exc:
                blocked = ProductionBlocked(
                    reason="MARKET_DATA_VALIDATION_FAILED",
                    detail=str(exc),
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_name,
                    symbol=self.symbol,
                    timeframe=self.interval,
                )
                return self._blocked_result(
                    blocked,
                    persist=persist,
                    publish=publish,
                    skip_if_no_trade=skip_if_no_trade,
                    reference_now=ref_now,
                )
        else:
            try:
                data = load_live_market_data(
                    symbol=self.symbol,
                    interval=self.interval,
                    limit=self.limit,
                )
            except UnsupportedInstrumentError as exc:
                blocked = ProductionBlocked(
                    reason="UNSUPPORTED_INSTRUMENT",
                    detail=str(exc),
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_name,
                    symbol=self.symbol,
                    timeframe=self.interval,
                )
                res = self._blocked_result(
                    blocked,
                    persist=persist,
                    publish=publish,
                    skip_if_no_trade=skip_if_no_trade,
                    reference_now=ref_now,
                )
                res["blocked_state"]["code"] = "UNSUPPORTED_INSTRUMENT"
                res["blocked_state"]["error"] = {
                    "code": "UNSUPPORTED_INSTRUMENT",
                    "message": str(exc),
                }
                return res
            except ValueError as exc:
                blocked = ProductionBlocked(
                    reason="MARKET_DATA_VALIDATION_FAILED",
                    detail=str(exc),
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_name,
                    symbol=self.symbol,
                    timeframe=self.interval,
                )
                return self._blocked_result(
                    blocked,
                    persist=persist,
                    publish=publish,
                    skip_if_no_trade=skip_if_no_trade,
                    reference_now=ref_now,
                )
            except Exception as exc:
                blocked = ProductionBlocked(
                    reason="MARKET_DATA_ACQUISITION_FAILED",
                    detail=str(exc),
                    candidate_id=candidate.candidate_id,
                    strategy_id=candidate.strategy_name,
                    symbol=self.symbol,
                    timeframe=self.interval,
                )
                return self._blocked_result(
                    blocked,
                    persist=persist,
                    publish=publish,
                    skip_if_no_trade=skip_if_no_trade,
                    reference_now=ref_now,
                )

        # 4. Create authoritative LiveMarketEvaluation
        evaluation = create_live_market_evaluation(
            data=data,
            context=context,
            reference_now=ref_now,
            max_age_seconds=max_age,
        )

        # 5. Delegate directly to the ONE canonical downstream evaluation boundary
        runtime_res = evaluate_authorized_live_runtime(
            data=data,
            evaluation=evaluation,
            context=context,
            stable_strategy=stable_strategy,
            stability_score=stability_score,
            min_stability_score=0.50,
            store_path=self.store_path,
            publisher=self.publisher,
            publish=publish,
            skip_if_no_trade=skip_if_no_trade,
            persist=persist,
            actor="live_execution_runtime",
        )

        canonical_cld = runtime_res.canonical_decision
        display = runtime_res.display

        publication = ProductionIntelligencePublication.from_artifacts(
            decision=canonical_cld.decision,
            signal=canonical_cld.signal,
            risk=canonical_cld.risk_levels,
            candidate=candidate,
            authorization=receipt,
        )

        record = build_live_decision_record(display)
        record["decision_id"] = canonical_cld.decision.decision_id
        record["signal_id"] = canonical_cld.signal.signal_id
        record["canonical_live_decision_fingerprint"] = canonical_cld.canonical_live_decision_fingerprint
        record["current_lifecycle_state"] = canonical_cld.current_state.value
        record["runtime_authorization_fingerprint"] = receipt.authorization_fingerprint
        record["authorization_policy_version"] = receipt.authorization_policy_version
        record["authorized_at_utc"] = receipt.authorized_at_utc
        record["promoted_artifact_fingerprint"] = receipt.promoted_artifact_fingerprint
        record["governance_decision_fingerprint"] = receipt.governance_decision_fingerprint
        record["campaign_selection_decision_fingerprint"] = receipt.campaign_selection_decision_fingerprint
        record["candidate_id"] = receipt.candidate_id
        record["strategy_name"] = receipt.strategy_name
        record["strategy_version"] = receipt.strategy_version
        record["context_fingerprint"] = context.context_fingerprint
        record["evaluation_fingerprint"] = evaluation.evaluation_fingerprint

        contract_payload = publication.to_contract_v1_payload()

        pub_res = runtime_res.decision.get("publish_result")

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
            "runtime_authorization": receipt.as_dict(),
            "runtime_authorization_fingerprint": receipt.authorization_fingerprint,
            "authorization_policy_version": receipt.authorization_policy_version,
            "authorized_at_utc": receipt.authorized_at_utc,
            "promoted_artifact_fingerprint": receipt.promoted_artifact_fingerprint,
            "governance_decision_fingerprint": receipt.governance_decision_fingerprint,
            "campaign_selection_decision_fingerprint": receipt.campaign_selection_decision_fingerprint,
            "context_fingerprint": context.context_fingerprint,
            "evaluation_fingerprint": evaluation.evaluation_fingerprint,
            "current_lifecycle_state": canonical_cld.current_state.value,
            "delivery_status": pub_res.get("delivery_status") if pub_res else None,
            "delivery_receipt_fingerprint": pub_res.get("delivery_receipt_fingerprint") if pub_res else None,
            "record": record,
            "publication": publication.as_dict(),
            "contract_payload": contract_payload,
            "publish_result": pub_res,
            "market_data": data.copy(),
        }

        if persist:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snap_dict = {k: v for k, v in execution_result.items() if k != "market_data"}
            self.snapshot_path.write_text(
                json.dumps(snap_dict, ensure_ascii=False, indent=2),
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
