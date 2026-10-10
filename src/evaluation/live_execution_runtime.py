"""Headless live execution engine for Project 1 -> Project 2 trading intelligence pipeline."""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

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
from src.evaluation.mtf_intelligence import CanonicalTimeframe
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

DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 200
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
    """Centralized validation and defensive copy helper for live OHLC market snapshots.

    Enforces strict chronological sorting, timestamp parseability, non-negative finite OHLC prices,
    and deduplicates identical timestamps while preserving valid candle structure.
    """
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError(f"Supplied market data for {symbol} must be a non-empty pandas DataFrame.")

    if "timestamp" in data.columns:
        raw_ts = data["timestamp"]
    elif "openTime" in data.columns:
        raw_ts = data["openTime"]
    else:
        raise ValueError(f"Missing required timestamp column ('timestamp' or 'openTime') for {symbol}.")

    required_numeric = {"open", "high", "low", "close"}
    missing_numeric = required_numeric.difference(data.columns)
    if missing_numeric:
        raise ValueError(
            f"Missing required live columns for {symbol}: " + ", ".join(sorted(missing_numeric))
        )

    data_copy = data.copy()
    data_copy["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")

    # Reject unparseable timestamps
    if data_copy["timestamp"].isna().any():
        raise ValueError(f"Market data for {symbol} contains unparseable or naive invalid timestamps.")

    for col in ("open", "high", "low", "close"):
        data_copy[col] = pd.to_numeric(data_copy[col], errors="coerce")
        # Validate finite numeric positive values
        if data_copy[col].isna().any() or not data_copy[col].apply(math.isfinite).all():
            raise ValueError(f"Market data for {symbol} contains non-finite/NaN values in '{col}'.")
        if (data_copy[col] <= 0).any():
            raise ValueError(f"Market data for {symbol} contains non-positive price values in '{col}'.")

    # Full OHLC geometry check: high >= max(open, close) and low <= min(open, close)
    if (data_copy["high"] < data_copy[["open", "close"]].max(axis=1)).any() or (data_copy["low"] > data_copy[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"Market data for {symbol} contains invalid OHLC geometry (high/low bounds violated).")

    # Sort chronologically
    data_copy = data_copy.sort_values("timestamp").reset_index(drop=True)

    # Check for conflicting duplicate timestamps (different OHLC values at same timestamp)
    dups = data_copy[data_copy.duplicated(subset=["timestamp"], keep=False)]
    if not dups.empty:
        # Group by timestamp and verify all numeric values match
        numeric_cols = ["open", "high", "low", "close"]
        for _, group in dups.groupby("timestamp"):
            if not (group[numeric_cols].nunique() == 1).all().all():
                raise ValueError(f"Market data for {symbol} contains conflicting duplicate timestamps with inconsistent OHLC values.")
        data_copy = data_copy.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if data_copy.empty:
        raise ValueError(f"No valid live market data available for {symbol}.")

    return data_copy


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


def verify_timeframe_production_readiness(
    symbol: str = "XAUUSD",
    timeframe: str | CanonicalTimeframe = "5m",
    candidate_id: str | None = None,
    research_dir: Path | str = DEFAULT_RESEARCH_DIR,
    publisher: Project2Publisher | None = None,
    reference_now: datetime.datetime | None = None,
    check_provider_data: bool = False,
) -> dict[str, Any]:
    """Perform deterministic production-readiness evaluation for a single canonical timeframe.

    Evaluates provider interval support, candidate artifact and binding existence/integrity,
    exact symbol/timeframe scope matching, production authorization receipt construction,
    and publication configuration. Never borrows candidates across timeframes or fabricates data.
    """
    canonical_tf = CanonicalTimeframe.from_str(timeframe).value
    r_dir = Path(research_dir)
    ref_now = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

    # 1. Provider interval support check
    provider_interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1H": "1h",
        "4H": "4h",
        "1D": "1d",
    }
    provider_interval = provider_interval_map.get(canonical_tf)
    if not provider_interval:
        return {
            "timeframe": canonical_tf,
            "status": "BLOCKED",
            "reason_code": "UNSUPPORTED_PROVIDER_INTERVAL",
            "detail": f"Timeframe '{canonical_tf}' has no mapped provider interval.",
            "candidate_id": candidate_id,
        }

    if check_provider_data:
        try:
            prov = resolve_live_provider(symbol)
            candles = prov.get_candles(symbol, timeframe=canonical_tf, limit=5)
            if candles is None or candles.empty:
                return {
                    "timeframe": canonical_tf,
                    "status": "BLOCKED",
                    "reason_code": "NO_PROVIDER_CANDLES",
                    "detail": f"Provider returned empty candle data for {symbol} {canonical_tf}.",
                    "candidate_id": candidate_id,
                }
        except Exception as exc:
            return {
                "timeframe": canonical_tf,
                "status": "BLOCKED",
                "reason_code": "PROVIDER_FETCH_FAILED",
                "detail": f"Failed to fetch market candles for {symbol} {canonical_tf}: {exc}",
                "candidate_id": candidate_id,
            }

    # 2. Promoted candidate resolution & binding integrity
    config = ProductionRuntimeConfig(
        symbol=symbol,
        timeframe=canonical_tf,
        candidate_id=candidate_id,
        research_dir=r_dir,
    )
    resolved = resolve_authoritative_promoted_candidate(config)
    if isinstance(resolved, ProductionBlocked):
        return {
            "timeframe": canonical_tf,
            "status": "BLOCKED",
            "reason_code": resolved.reason,
            "detail": resolved.detail,
            "candidate_id": resolved.candidate_id or candidate_id,
        }

    # Explicit scope check
    cand_tf = CanonicalTimeframe.from_str(resolved.timeframe).value
    if cand_tf != canonical_tf:
        return {
            "timeframe": canonical_tf,
            "status": "BLOCKED",
            "reason_code": "TIMEFRAME_IDENTITY_MISMATCH",
            "detail": f"Candidate '{resolved.candidate_id}' timeframe '{cand_tf}' does not match requested '{canonical_tf}'.",
            "candidate_id": resolved.candidate_id,
        }

    # 3. Production authorization receipt verification
    try:
        authorization = authorize_production_runtime(
            resolved,
            symbol=symbol,
            timeframe=canonical_tf,
            now=ref_now,
        )
        receipt = ProductionAuthorizationReceipt.from_authorization(authorization)
    except Exception as exc:
        return {
            "timeframe": canonical_tf,
            "status": "BLOCKED",
            "reason_code": "AUTHORIZATION_FAILED",
            "detail": f"Failed to authorize production runtime for candidate '{resolved.candidate_id}': {exc}",
            "candidate_id": resolved.candidate_id,
        }

    # 4. Publication configuration check
    pub = publisher or Project2Publisher()
    pub_url = pub.publish_url
    pub_enabled = pub.enabled
    if pub_enabled and not pub_url:
        return {
            "timeframe": canonical_tf,
            "status": "BLOCKED",
            "reason_code": "PUBLICATION_MISCONFIGURED",
            "detail": "Project 2 publication is enabled but no publish URL is configured.",
            "candidate_id": resolved.candidate_id,
        }

    return {
        "timeframe": canonical_tf,
        "status": "READY",
        "reason_code": "READY",
        "detail": f"Authoritative candidate '{resolved.candidate_id}' fully authorized for {symbol} {canonical_tf}.",
        "candidate_id": resolved.candidate_id,
        "strategy_name": resolved.strategy_name,
        "strategy_version": resolved.strategy_version,
        "operational_stability_score": resolved.operational_stability_score,
        "authorization_fingerprint": receipt.authorization_fingerprint,
    }


def verify_all_canonical_timeframes_readiness(
    symbol: str = "XAUUSD",
    candidate_ids: dict[str, str] | None = None,
    research_dir: Path | str = DEFAULT_RESEARCH_DIR,
    publisher: Project2Publisher | None = None,
    reference_now: datetime.datetime | None = None,
    check_provider_data: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate production readiness independently across all seven canonical timeframes."""
    cand_map = candidate_ids or {}
    results = {}
    for tf in CanonicalTimeframe.canonical_ladder():
        tf_val = tf.value
        cand_id = cand_map.get(tf_val)
        res = verify_timeframe_production_readiness(
            symbol=symbol,
            timeframe=tf_val,
            candidate_id=cand_id,
            research_dir=research_dir,
            publisher=publisher,
            reference_now=reference_now,
            check_provider_data=check_provider_data,
        )
        results[tf_val] = res
    return results


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
    timeframe: str | CanonicalTimeframe,
    max_age_seconds: float | None = None,
    reference_now: datetime.datetime | None = None,
) -> dict[str, Any]:
    """Validate that the latest provider candle is closed and within provider-lateness allowance.

    Provider timestamps identify candle opens. An explicit, finite, non-negative ``max_age_seconds``
    defines the maximum permitted provider/update lateness AFTER the candle's own duration.
    If ``max_age_seconds`` is missing, non-numeric, or non-finite, freshness evaluation fails closed.
    """
    if reference_now is None:
        now_dt = datetime.datetime.now(datetime.timezone.utc)
    else:
        now_dt = reference_now
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)

    if max_age_seconds is None or isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float)) or not math.isfinite(float(max_age_seconds)) or float(max_age_seconds) < 0:
        return {
            "fresh": False,
            "stale": True,
            "reason": "absent_or_invalid_freshness_policy",
            "age_seconds": None,
            "candle_timestamp": None,
        }

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
        return {
            "fresh": False,
            "stale": True,
            "reason": "naive_candle_timestamp",
            "age_seconds": None,
            "candle_timestamp": str(latest_ts),
        }
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

    canonical_tf = CanonicalTimeframe.from_str(timeframe)
    latest_row = data.iloc[-1]
    if not is_candle_closed(latest_row, canonical_tf, now_dt):
        return {"fresh": False, "stale": True, "reason": "unclosed_market_data", "age_seconds": age_seconds, "timeframe": canonical_tf.value, "candle_timestamp": candle_iso}

    max_permitted_age = get_canonical_timeframe_duration(canonical_tf).total_seconds() + float(max_age_seconds)
    if age_seconds > max_permitted_age:
        return {
            "fresh": False,
            "stale": True,
            "reason": "stale_market_data",
            "age_seconds": age_seconds,
            "max_market_age_seconds": max_permitted_age,
            "timeframe": canonical_tf.value,
            "candle_timestamp": candle_iso,
        }

    return {
        "fresh": True,
        "stale": False,
        "reason": "fresh",
        "age_seconds": age_seconds,
        "timeframe": canonical_tf.value,
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

        # 0. Attempt recovery for any prior pending FAILED_RETRYABLE publication receipts
        if publish and self.publisher and getattr(self.publisher, "enabled", False):
            from src.evaluation.live_publication_store import recover_pending_publication_deliveries
            try:
                recover_pending_publication_deliveries(
                    publisher=self.publisher,
                    publication_path=self.store_path.parent / "publication_history.json",
                    delivery_path=self.store_path.parent / "delivery_history.json",
                    decision_store_path=self.store_path,
                    research_dir=self.research_dir,
                )
            except Exception as exc:
                logger.warning("Pending publication delivery recovery error: %s", exc)

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
        try:
            evaluation = create_live_market_evaluation(
                data=data,
                context=context,
                reference_now=ref_now,
                max_age_seconds=max_age,
            )
        except Exception as exc:
            blocked = ProductionBlocked(
                reason="LIVE_MARKET_EVALUATION_FAILED",
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

        # 5. Delegate directly to the ONE canonical downstream evaluation boundary
        try:
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
        except Exception as exc:
            blocked = ProductionBlocked(
                reason="LIVE_RUNTIME_EVALUATION_FAILED",
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

        canonical_cld = runtime_res.canonical_decision
        display = runtime_res.display
        pub_res = runtime_res.decision.get("publish_result")

        record = None
        publication_dict = None
        contract_payload = None

        try:
            publication = ProductionIntelligencePublication.from_artifacts(
                decision=canonical_cld.decision,
                signal=canonical_cld.signal,
                risk=canonical_cld.risk_levels,
                candidate=candidate,
                authorization=receipt,
            )
            publication_dict = publication.as_dict()
            contract_payload = publication.to_contract_v1_payload()

            rec = build_live_decision_record(display)
            rec["decision_id"] = canonical_cld.decision.decision_id
            rec["signal_id"] = canonical_cld.signal.signal_id
            rec["canonical_live_decision_fingerprint"] = canonical_cld.canonical_live_decision_fingerprint
            rec["current_lifecycle_state"] = canonical_cld.current_state.value
            rec["runtime_authorization_fingerprint"] = receipt.authorization_fingerprint
            rec["authorization_policy_version"] = receipt.authorization_policy_version
            rec["authorized_at_utc"] = receipt.authorized_at_utc
            rec["promoted_artifact_fingerprint"] = receipt.promoted_artifact_fingerprint
            rec["governance_decision_fingerprint"] = receipt.governance_decision_fingerprint
            rec["campaign_selection_decision_fingerprint"] = receipt.campaign_selection_decision_fingerprint
            rec["candidate_id"] = receipt.candidate_id
            rec["strategy_name"] = receipt.strategy_name
            rec["strategy_version"] = receipt.strategy_version
            rec["context_fingerprint"] = context.context_fingerprint
            rec["evaluation_fingerprint"] = evaluation.evaluation_fingerprint
            record = rec
        except Exception as exc:
            if pub_res is None:
                pub_res = {
                    "status": "FAILED",
                    "published": False,
                    "reason": "PUBLICATION_ARTIFACT_CONSTRUCTION_FAILED",
                    "error": str(exc),
                    "detail": str(exc),
                }

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
            "publication": publication_dict,
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


def get_canonical_timeframe_duration(timeframe: str | CanonicalTimeframe) -> datetime.timedelta:
    """Return exact timedelta duration for a canonical timeframe. Fails closed on unsupported timeframes."""
    tf = CanonicalTimeframe.from_str(timeframe)
    if tf == CanonicalTimeframe.ONE_MINUTE:
        return datetime.timedelta(minutes=1)
    elif tf == CanonicalTimeframe.FIVE_MINUTES:
        return datetime.timedelta(minutes=5)
    elif tf == CanonicalTimeframe.FIFTEEN_MINUTES:
        return datetime.timedelta(minutes=15)
    elif tf == CanonicalTimeframe.THIRTY_MINUTES:
        return datetime.timedelta(minutes=30)
    elif tf == CanonicalTimeframe.ONE_HOUR:
        return datetime.timedelta(hours=1)
    elif tf == CanonicalTimeframe.FOUR_HOURS:
        return datetime.timedelta(hours=4)
    elif tf == CanonicalTimeframe.ONE_DAY:
        return datetime.timedelta(days=1)
    else:
        raise ValueError(f"Unsupported timeframe duration: {timeframe}")


def is_candle_closed(
    row: pd.Series,
    timeframe: str | CanonicalTimeframe,
    reference_now: datetime.datetime,
) -> bool:
    """Deterministically ascertain closed state of a candle row.

    1. Uses explicit provider open/closed boolean state if present in columns.
    2. Fallback: candle_open_timestamp + exact canonical duration <= reference_now.
    """
    for open_col in ("isOpen", "is_open"):
        if open_col in row.index and not pd.isna(row[open_col]):
            return not bool(row[open_col])

    for closed_col in ("isClosed", "is_closed"):
        if closed_col in row.index and not pd.isna(row[closed_col]):
            return bool(row[closed_col])

    ts = row.get("timestamp") if "timestamp" in row.index else row.get("openTime")
    if pd.isna(ts) or ts is None:
        return False

    if isinstance(ts, pd.Timestamp):
        open_dt = ts.to_pydatetime()
    elif isinstance(ts, datetime.datetime):
        open_dt = ts
    else:
        try:
            open_dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return False

    if open_dt.tzinfo is None:
        open_dt = open_dt.replace(tzinfo=datetime.timezone.utc)

    ref_dt = reference_now
    if ref_dt.tzinfo is None:
        ref_dt = ref_dt.replace(tzinfo=datetime.timezone.utc)

    duration = get_canonical_timeframe_duration(timeframe)
    return (open_dt + duration) <= ref_dt


def parse_continuous_candidate_ids(
    arg_str: str | None,
    timeframes: Sequence[str],
) -> dict[str, str]:
    """Parse candidate ID mappings for continuous multi-timeframe execution.

    Explicit per-timeframe format: '5m=cand_5m,15m=cand_15m'
    Single candidate ID is accepted ONLY when exactly 1 timeframe is configured.
    Blind copying of a single candidate across multiple continuous timeframes is strictly rejected.
    """
    if not arg_str or not str(arg_str).strip():
        return {}

    cleaned = str(arg_str).strip()
    result: dict[str, str] = {}

    if "=" in cleaned:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        for part in parts:
            if "=" not in part:
                raise ValueError(f"Invalid candidate-id mapping part '{part}'. Expected 'timeframe=candidate_id'.")
            tf, cand = part.split("=", 1)
            tf_canonical = CanonicalTimeframe.from_str(tf.strip()).value
            cand_id = cand.strip()
            if not cand_id:
                raise ValueError(f"Empty candidate_id supplied for timeframe '{tf}'.")
            result[tf_canonical] = cand_id
        return result

    # Plain string without '='
    canonical_tfs = [CanonicalTimeframe.from_str(tf).value for tf in timeframes]
    if len(canonical_tfs) == 1:
        result[canonical_tfs[0]] = cleaned
        return result
    else:
        raise ValueError(
            f"In continuous mode with multiple timeframes {canonical_tfs}, "
            f"--candidate-id must specify explicit per-timeframe mappings "
            f"(e.g. '5m=cand1,15m=cand2') rather than copying a single candidate ID '{cleaned}' across timeframes."
        )


class ContinuousLiveRuntime:
    """Reusable continuous live-runtime orchestration layer.

    Repeatedly polls for market data across configured canonical timeframes,
    detects due closed candles, deduplicates evaluations, isolates recoverable
    failures, and delegates strictly to the existing authoritative LiveExecutionRuntime.run_once()
    primitive for production decisions and Project 2 publication.
    """

    def __init__(
        self,
        symbol: str = "XAUUSD",
        timeframes: Sequence[str] | str | None = None,
        poll_interval: float = 1.0,
        publish: bool = True,
        skip_if_no_trade: bool = False,
        persist: bool = True,
        max_age_seconds: float = 300.0,
        limit: int = DEFAULT_LIMIT,
        research_dir: Path | str = DEFAULT_RESEARCH_DIR,
        store_path: Path | str = DEFAULT_STORE_PATH,
        snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
        publisher: Project2Publisher | None = None,
        market_data_loaders: dict[str, Callable[[], pd.DataFrame]] | Callable[[str], pd.DataFrame] | None = None,
        candidate_ids: dict[str, str] | None = None,
        clock: Callable[[], datetime.datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        if timeframes is None:
            tf_list = [DEFAULT_INTERVAL]
        elif isinstance(timeframes, str):
            tf_list = [timeframes]
        else:
            tf_list = list(timeframes)

        # Enforce repository's existing canonical timeframe vocabulary.
        # Fails closed on unsupported timeframes.
        self.timeframes = tuple(CanonicalTimeframe.from_str(tf).value for tf in tf_list)

        self.poll_interval = max(0.0, poll_interval)
        self.publish = publish
        self.skip_if_no_trade = skip_if_no_trade
        self.persist = persist
        self.max_age_seconds = max_age_seconds
        self.limit = limit
        self.research_dir = Path(research_dir)
        self.store_path = Path(store_path)
        self.snapshot_path = Path(snapshot_path)
        self.publisher = publisher or Project2Publisher(max_age_seconds=int(max_age_seconds))
        self.market_data_loaders = market_data_loaders
        self.candidate_ids = candidate_ids or {}
        self.clock = clock or (lambda: datetime.datetime.now(datetime.timezone.utc))
        self.sleep_fn = sleep_fn or time.sleep

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._evaluated_candles: set[tuple[str, str, str, str | None]] = set()
        self._tick_count = 0
        self._execution_history: list[dict[str, Any]] = []
        self._started_at_utc: str | None = None
        self._last_tick_at_utc: str | None = None
        self._last_successful_evaluation_at_utc: str | None = None
        self._last_successful_publication_at_utc: str | None = None
        self._last_failure_reason: str | None = None

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def get_health_status(
        self,
        staleness_threshold_seconds: float = 300.0,
        reference_now: datetime.datetime | None = None,
    ) -> dict[str, Any]:
        """Compute current worker health diagnostics.

        Health states:
        - WORKER_NOT_STARTED: continuous loop hasn't started or executed ticks
        - WORKER_HEALTHY: tick heartbeat within staleness threshold
        - WORKER_STALE: tick heartbeat older than staleness threshold
        """
        ref_now = reference_now if reference_now is not None else self.clock()
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

        if not self._started_at_utc or not self._last_tick_at_utc:
            return {
                "status": "WORKER_NOT_STARTED",
                "healthy": False,
                "started_at_utc": self._started_at_utc,
                "last_tick_at_utc": self._last_tick_at_utc,
                "last_successful_evaluation_at_utc": self._last_successful_evaluation_at_utc,
                "last_successful_publication_at_utc": self._last_successful_publication_at_utc,
                "last_failure_reason": self._last_failure_reason,
                "seconds_since_last_tick": None,
                "tick_count": self._tick_count,
            }

        last_tick_dt = datetime.datetime.fromisoformat(self._last_tick_at_utc)
        if last_tick_dt.tzinfo is None:
            last_tick_dt = last_tick_dt.replace(tzinfo=datetime.timezone.utc)

        age_seconds = (ref_now - last_tick_dt).total_seconds()
        is_healthy = age_seconds >= 0 and age_seconds <= staleness_threshold_seconds

        status = "WORKER_HEALTHY" if is_healthy else "WORKER_STALE"

        return {
            "status": status,
            "healthy": is_healthy,
            "started_at_utc": self._started_at_utc,
            "last_tick_at_utc": self._last_tick_at_utc,
            "last_successful_evaluation_at_utc": self._last_successful_evaluation_at_utc,
            "last_successful_publication_at_utc": self._last_successful_publication_at_utc,
            "last_failure_reason": self._last_failure_reason,
            "seconds_since_last_tick": max(0.0, age_seconds),
            "tick_count": self._tick_count,
        }

    def stop(self) -> None:
        """Signal the continuous runtime loop to stop deterministically."""
        self._stop_event.set()

    def reset_deduplication(self) -> None:
        """Clear the evaluated candle deduplication cache."""
        self._evaluated_candles.clear()

    def _acquire_market_data(self, timeframe: str) -> pd.DataFrame:
        """Acquire market data snapshot for a given timeframe using injected loaders or default provider."""
        if self.market_data_loaders is not None:
            if callable(self.market_data_loaders):
                return self.market_data_loaders(timeframe)
            elif isinstance(self.market_data_loaders, dict) and timeframe in self.market_data_loaders:
                loader = self.market_data_loaders[timeframe]
                return loader()

        return load_live_market_data(
            symbol=self.symbol,
            interval=timeframe,
            limit=self.limit,
        )

    def _get_latest_closed_candle(
        self,
        df: pd.DataFrame,
        timeframe: str,
        reference_now: datetime.datetime,
    ) -> tuple[int, pd.Series, str] | None:
        """Locate the latest row in df that satisfies the closed-candle boundary.

        Returns (index, row, candle_timestamp_iso) or None if no closed candle exists.
        """
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if is_candle_closed(row, timeframe, reference_now):
                ts = row["timestamp"]
                open_dt = (
                    ts.to_pydatetime()
                    if isinstance(ts, pd.Timestamp)
                    else ts
                )
                if open_dt.tzinfo is None:
                    open_dt = open_dt.replace(tzinfo=datetime.timezone.utc)
                return i, row, open_dt.isoformat()
        return None

    def tick(self, reference_now: datetime.datetime | None = None) -> dict[str, Any]:
        """Execute one continuous orchestration tick across all configured canonical timeframes."""
        if not self._lock.acquire(blocking=False):
            logger.warning("Tick skipped: previous tick evaluation still in progress.")
            return {"skipped": True, "reason": "concurrent_tick_locked", "evaluations": {}}

        try:
            ref_now = reference_now if reference_now is not None else self.clock()
            if ref_now.tzinfo is None:
                ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

            self._tick_count += 1
            tick_evaluations: dict[str, Any] = {}

            for tf in self.timeframes:
                if self._stop_event.is_set():
                    break

                canonical_tf = CanonicalTimeframe.from_str(tf).value

                # 1. Acquire market data snapshot for timeframe
                try:
                    raw_df = self._acquire_market_data(canonical_tf)
                    prepared_df = validate_and_prepare_market_snapshot(raw_df, symbol=self.symbol)
                except Exception as exc:
                    logger.error("Recoverable failure acquiring market data for %s %s: %s", self.symbol, canonical_tf, exc)
                    tick_evaluations[canonical_tf] = {
                        "status": "FAILED_RECOVERABLE",
                        "error": str(exc),
                        "symbol": self.symbol,
                        "timeframe": canonical_tf,
                    }
                    continue

                # 2. Closed-candle gate: verify explicit provider state or duration calculation
                closed_info = self._get_latest_closed_candle(prepared_df, canonical_tf, ref_now)
                if closed_info is None:
                    tick_evaluations[canonical_tf] = {
                        "status": "SKIPPED_NO_CLOSED_CANDLE",
                        "reason": "latest_candle_is_open_or_unavailable",
                        "symbol": self.symbol,
                        "timeframe": canonical_tf,
                    }
                    continue

                closed_idx, closed_row, closed_ts_iso = closed_info

                # Slice market data up to and including the closed candle
                eval_df = prepared_df.iloc[: closed_idx + 1].copy()

                # 3. Candidate resolution for exact timeframe
                config = ProductionRuntimeConfig(
                    symbol=self.symbol,
                    timeframe=canonical_tf,
                    candidate_id=self.candidate_ids.get(canonical_tf),
                    research_dir=self.research_dir,
                )
                resolved = resolve_authoritative_promoted_candidate(config)
                if isinstance(resolved, PromotedCandidateArtifact):
                    # Validate candidate timeframe matches exact requested timeframe
                    candidate_tf = CanonicalTimeframe.from_str(resolved.timeframe).value
                    req_tf = CanonicalTimeframe.from_str(canonical_tf).value
                    if candidate_tf != req_tf:
                        resolved = ProductionBlocked(
                            reason="PromotionEligibilityError",
                            detail=(
                                f"Promoted candidate '{resolved.candidate_id}' timeframe "
                                f"'{candidate_tf}' does not match requested timeframe '{req_tf}'."
                            ),
                            candidate_id=resolved.candidate_id,
                            strategy_id=resolved.strategy_name,
                            symbol=self.symbol,
                            timeframe=req_tf,
                        )

                cand_id = resolved.candidate_id if resolved is not None else None

                # 4. Construct 4-field deduplication key
                dedup_key = (self.symbol, canonical_tf, closed_ts_iso, cand_id)

                # 5. Deduplication check BEFORE evaluation
                if dedup_key in self._evaluated_candles:
                    tick_evaluations[canonical_tf] = {
                        "status": "SKIPPED_DEDUPLICATED",
                        "reason": "candle_already_evaluated",
                        "candle_timestamp": closed_ts_iso,
                        "candidate_id": cand_id,
                        "timeframe": canonical_tf,
                    }
                    continue

                # 6. Delegate strictly to existing authoritative run_once() primitive
                try:
                    runtime = LiveExecutionRuntime(
                        symbol=self.symbol,
                        interval=canonical_tf,
                        limit=self.limit,
                        publisher=self.publisher,
                        store_path=self.store_path,
                        snapshot_path=self.snapshot_path,
                        max_age_seconds=self.max_age_seconds,
                        research_dir=self.research_dir,
                        production_config=config,
                    )

                    result = runtime.run_once(
                        publish=self.publish,
                        skip_if_no_trade=self.skip_if_no_trade,
                        persist=self.persist,
                        reference_now=ref_now,
                        market_data=eval_df,
                    )

                    tick_evaluations[canonical_tf] = result

                    # 7. Add to deduplication set ONLY AFTER successful evaluation execution
                    if result.get("blocked") is not True:
                        self._evaluated_candles.add(dedup_key)
                        self._last_successful_evaluation_at_utc = ref_now.isoformat()
                        pub_res = result.get("publish_result")
                        if pub_res and pub_res.get("published") is True:
                            self._last_successful_publication_at_utc = ref_now.isoformat()
                    else:
                        self._last_failure_reason = result.get("reason") or "EVALUATION_BLOCKED"

                except Exception as exc:
                    logger.error("Recoverable failure evaluating %s %s: %s", self.symbol, canonical_tf, exc)
                    self._last_failure_reason = f"EXCEPTIONAL_FAILURE: {exc}"
                    tick_evaluations[canonical_tf] = {
                        "status": "FAILED_RECOVERABLE",
                        "error": str(exc),
                        "symbol": self.symbol,
                        "timeframe": canonical_tf,
                    }

            if not self._started_at_utc:
                self._started_at_utc = ref_now.isoformat()
            self._last_tick_at_utc = ref_now.isoformat()

            tick_summary = {
                "tick_number": self._tick_count,
                "timestamp": ref_now.isoformat(),
                "evaluations": tick_evaluations,
                "health": self.get_health_status(reference_now=ref_now),
            }
            self._execution_history.append(tick_summary)
            return tick_summary
        finally:
            self._lock.release()

    def run_ticks(self, max_ticks: int, reference_now: datetime.datetime | None = None) -> list[dict[str, Any]]:
        """Run a fixed number of continuous ticks deterministically."""
        results = []
        for _ in range(max_ticks):
            if self._stop_event.is_set():
                break
            res = self.tick(reference_now=reference_now)
            results.append(res)
            if self.poll_interval > 0 and not self._stop_event.is_set():
                self.sleep_fn(self.poll_interval)
        return results

    def verify_startup_readiness(self) -> dict[str, Any]:
        """Perform preflight checks ensuring authoritative promoted candidate artifacts are resolvable before starting loop.

        Fails closed with PromotionUnavailable / RuntimeError if any configured timeframe candidate cannot be resolved.
        """
        if self.publish and self.publisher and getattr(self.publisher, "enabled", False):
            from src.evaluation.live_publication_store import recover_pending_publication_deliveries
            try:
                recover_pending_publication_deliveries(
                    publisher=self.publisher,
                    publication_path=self.store_path.parent / "publication_history.json",
                    delivery_path=self.store_path.parent / "delivery_history.json",
                    decision_store_path=self.store_path,
                    research_dir=self.research_dir,
                )
            except Exception as exc:
                logger.warning("Startup pending delivery recovery warning: %s", exc)

        readiness_results = {}
        for tf in self.timeframes:
            canonical_tf = CanonicalTimeframe.from_str(tf).value
            cand_id = self.candidate_ids.get(canonical_tf)
            readiness = verify_timeframe_production_readiness(
                symbol=self.symbol,
                timeframe=canonical_tf,
                candidate_id=cand_id,
                research_dir=self.research_dir,
                publisher=self.publisher,
                reference_now=self.clock(),
            )
            if readiness.get("status") != "READY":
                msg = (
                    f"Startup readiness preflight failed for {self.symbol} {canonical_tf}: "
                    f"[{readiness.get('reason_code')}] {readiness.get('detail')}"
                )
                logger.error(msg)
                raise PromotionUnavailable(msg)
            readiness_results[canonical_tf] = readiness
        return readiness_results

    def run_continuous(self, max_ticks: int | None = None) -> None:
        """Run continuous market polling loop until explicit stop or max_ticks reached."""
        # Execute startup preflight check
        self.verify_startup_readiness()

        self._stop_event.clear()
        ticks_executed = 0
        logger.info("Starting ContinuousLiveRuntime loop for %s timeframes=%s", self.symbol, self.timeframes)
        try:
            while not self._stop_event.is_set():
                if max_ticks is not None and ticks_executed >= max_ticks:
                    logger.info("Reached max_ticks=%d, stopping continuous loop.", max_ticks)
                    break
                self.tick()
                ticks_executed += 1
                if self.poll_interval > 0 and not self._stop_event.is_set():
                    self.sleep_fn(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("ContinuousLiveRuntime interrupted by user (SIGINT).")
        finally:
            self._stop_event.set()
            logger.info("ContinuousLiveRuntime loop stopped after %d ticks.", ticks_executed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Live Execution Runtime")
    parser.add_argument("--symbol", type=str, default="XAUUSD", help="Target instrument symbol")
    parser.add_argument("--interval", type=str, default="5m", help="Market data timeframe interval")
    parser.add_argument("--candidate-id", type=str, default=None, help="Explicit promoted candidate ID")
    parser.add_argument("--limit", type=int, default=100, help="Number of candles to fetch")
    parser.add_argument("--publish", action="store_true", help="Enable outbound publishing to Project 2")
    parser.add_argument("--skip-no-trade", action="store_true", help="Skip publishing when decision is NO TRADE")
    parser.add_argument("--no-persist", action="store_true", help="Disable history persistence")
    parser.add_argument("--continuous", action="store_true", help="Enable continuous live runtime orchestration")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Polling interval in seconds for continuous mode")
    parser.add_argument("--max-ticks", type=int, default=None, help="Maximum number of continuous loop ticks")
    parser.add_argument("--timeframes", type=str, default=None, help="Comma-separated timeframes for continuous mode (e.g. '5m,15m')")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.continuous:
        if args.timeframes:
            tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]
        else:
            tfs = [args.interval]

        candidate_map = parse_continuous_candidate_ids(args.candidate_id, timeframes=tfs)

        cont_runtime = ContinuousLiveRuntime(
            symbol=args.symbol,
            timeframes=tfs,
            poll_interval=args.poll_interval,
            publish=args.publish,
            skip_if_no_trade=args.skip_no_trade,
            persist=not args.no_persist,
            limit=args.limit,
            research_dir=DEFAULT_RESEARCH_DIR,
            candidate_ids=candidate_map,
        )

        def handle_signal(sig, frame):
            logger.info("Signal %s received, stopping continuous runtime...", sig)
            cont_runtime.stop()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        try:
            cont_runtime.run_continuous(max_ticks=args.max_ticks)
        except Exception as exc:
            logger.error("Continuous live execution failed: %s", exc)
            sys.exit(1)
        return

    try:
        production_config = ProductionRuntimeConfig(
            symbol=args.symbol,
            timeframe=args.interval,
            candidate_id=args.candidate_id,
            research_dir=DEFAULT_RESEARCH_DIR,
        )
        runtime = LiveExecutionRuntime(
            symbol=args.symbol,
            interval=args.interval,
            limit=args.limit,
            production_config=production_config,
        )
        result = runtime.run_once(
            publish=args.publish,
            skip_if_no_trade=args.skip_no_trade,
            persist=not args.no_persist,
        )

        logger.info("Execution complete for %s %s", result["symbol"], result["interval"])
        stability_str = (
            f"{result['stability_score']:.3f}"
            if isinstance(result.get("stability_score"), (int, float))
            else "N/A"
        )
        logger.info(
            "Decision: %s | Strategy: %s (Stability: %s)",
            result["decision"],
            result["strategy"],
            stability_str,
        )

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
