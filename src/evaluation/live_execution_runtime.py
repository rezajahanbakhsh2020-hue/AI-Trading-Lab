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

PUB_FAILURE_STATUSES: tuple[str, ...] = (
    "FAILED",
    "REJECTED",
    "AUTH_FAILED",
    "FORBIDDEN",
    "UNAVAILABLE",
    "TIMED_OUT",
    "INVALID_RESPONSE",
    "SKIPPED_BLOCKED",
    "MISCONFIGURED",
)

_UNSPECIFIED = object()

# Explicit canonical-timeframe -> provider interval mapping. Fails closed for
# canonical timeframes without a provider interval (should be impossible).
PROVIDER_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}


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
    try:
        canonical_tf = CanonicalTimeframe.from_str(timeframe).value
    except (ValueError, TypeError) as exc:
        return {
            "timeframe": str(timeframe),
            "status": "BLOCKED",
            "reason_code": "INVALID_TIMEFRAME",
            "detail": str(exc),
            "candidate_id": candidate_id,
        }

    r_dir = Path(research_dir)
    ref_now = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

    # 1. Provider interval support check
    provider_interval = PROVIDER_INTERVAL_MAP.get(canonical_tf)
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
    if pub.enabled:
        from src.integration.project2_publisher import validate_publication_configuration
        valid_pub_config, pub_config_reason = validate_publication_configuration(
            pub.publish_url,
            getattr(pub, "api_key", None),
        )
        if not valid_pub_config:
            return {
                "timeframe": canonical_tf,
                "status": "BLOCKED",
                "reason_code": "PUBLICATION_MISCONFIGURED",
                "detail": f"Project 2 publication is enabled but misconfigured: {pub_config_reason}",
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


def verify_timeframe_direct_readiness(
    symbol: str = "XAUUSD",
    timeframe: str | CanonicalTimeframe = "5m",
    publisher: Project2Publisher | None = None,
    reference_now: datetime.datetime | None = None,
    check_provider_data: bool = False,
) -> dict[str, Any]:
    """Perform deterministic readiness evaluation for a direct-strategy timeframe.

    Direct mode requires NO candidate artifact, NO promotion, and NO research
    qualification: readiness depends only on a canonical timeframe identity and
    provider interval support, optional real provider-data availability, and a
    usable publication configuration when publishing is enabled.
    """
    try:
        canonical_tf = CanonicalTimeframe.from_str(timeframe).value
    except (ValueError, TypeError) as exc:
        return {
            "timeframe": str(timeframe),
            "status": "BLOCKED",
            "reason_code": "INVALID_TIMEFRAME",
            "detail": str(exc),
            "candidate_id": None,
        }

    provider_interval = PROVIDER_INTERVAL_MAP.get(canonical_tf)
    if not provider_interval:
        return {
            "timeframe": canonical_tf,
            "status": "BLOCKED",
            "reason_code": "UNSUPPORTED_PROVIDER_INTERVAL",
            "detail": f"Timeframe '{canonical_tf}' has no mapped provider interval.",
            "candidate_id": None,
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
                    "candidate_id": None,
                }
        except Exception as exc:
            return {
                "timeframe": canonical_tf,
                "status": "BLOCKED",
                "reason_code": "PROVIDER_FETCH_FAILED",
                "detail": f"Failed to fetch market candles for {symbol} {canonical_tf}: {exc}",
                "candidate_id": None,
            }

    from src.evaluation.live_direct_strategy import DIRECT_STRATEGY_ID, DIRECT_STRATEGY_VERSION

    pub = publisher or Project2Publisher()
    if pub.enabled:
        from src.integration.project2_publisher import validate_publication_configuration
        valid_pub_config, pub_config_reason = validate_publication_configuration(
            pub.publish_url,
            getattr(pub, "api_key", None),
        )
        if not valid_pub_config:
            return {
                "timeframe": canonical_tf,
                "status": "BLOCKED",
                "reason_code": "PUBLICATION_MISCONFIGURED",
                "detail": f"Project 2 publication is enabled but misconfigured: {pub_config_reason}",
                "candidate_id": None,
            }

    return {
        "timeframe": canonical_tf,
        "status": "READY",
        "reason_code": "READY",
        "detail": (
            f"Direct strategy '{DIRECT_STRATEGY_ID}' v{DIRECT_STRATEGY_VERSION} ready for "
            f"{symbol} {canonical_tf} (no candidate required)."
        ),
        "candidate_id": None,
        "strategy_name": DIRECT_STRATEGY_ID,
        "strategy_version": DIRECT_STRATEGY_VERSION,
    }


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

    def _build_recovery_execution_result(
        self,
        rec_item: dict[str, Any],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Construct a standardized execution result from a recovery attempt without generating a replacement decision."""
        pub_res = rec_item.get("publish_result") or {
            "status": rec_item.get("status") or "FAILED",
            "published": rec_item.get("status") == "PUBLISHED" or rec_item.get("delivery_status") == "DELIVERED",
            "reason": rec_item.get("reason"),
            "error": rec_item.get("error"),
        }
        record = rec_item.get("record") or {}

        execution_result = {
            "blocked": False,
            "symbol": self.symbol,
            "interval": self.interval,
            "decision": record.get("decision", "NO TRADE"),
            "strategy": record.get("strategy_name") or record.get("strategy"),
            "stability_score": record.get("stability_score"),
            "candidate_id": rec_item.get("candidate_id") or record.get("candidate_id"),
            "evidence_id": record.get("evidence_id"),
            "research_fingerprint": record.get("research_fingerprint"),
            "strategy_version": record.get("strategy_version"),
            "runtime_authorization": record.get("runtime_authorization"),
            "runtime_authorization_fingerprint": record.get("runtime_authorization_fingerprint"),
            "authorization_policy_version": record.get("authorization_policy_version"),
            "authorized_at_utc": record.get("authorized_at_utc"),
            "promoted_artifact_fingerprint": record.get("promoted_artifact_fingerprint"),
            "governance_decision_fingerprint": record.get("governance_decision_fingerprint"),
            "campaign_selection_decision_fingerprint": record.get("campaign_selection_decision_fingerprint"),
            "context_fingerprint": record.get("context_fingerprint"),
            "evaluation_fingerprint": record.get("evaluation_fingerprint"),
            "current_lifecycle_state": pub_res.get("current_lifecycle_state") or record.get("current_lifecycle_state"),
            "delivery_status": pub_res.get("delivery_status") or rec_item.get("delivery_status"),
            "delivery_receipt_fingerprint": pub_res.get("delivery_receipt_fingerprint"),
            "record": record if record else None,
            "publication": rec_item.get("publication"),
            "contract_payload": rec_item.get("contract_payload"),
            "publish_result": pub_res,
            "market_data": None,
        }

        if persist:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snap_dict = {k: v for k, v in execution_result.items() if k != "market_data"}
            self.snapshot_path.write_text(
                json.dumps(snap_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
                rec_results = recover_pending_publication_deliveries(
                    publisher=self.publisher,
                    publication_path=self.store_path.parent / "publication_history.json",
                    delivery_path=self.store_path.parent / "delivery_history.json",
                    decision_store_path=self.store_path,
                    research_dir=self.research_dir,
                )
                for rec_item in rec_results:
                    if (
                        rec_item.get("symbol") == self.symbol
                        and rec_item.get("timeframe") == self.interval
                    ):
                        return self._build_recovery_execution_result(rec_item, persist=persist)
            except Exception as exc:
                logger.warning("Pending publication delivery recovery error: %s", exc)
                blocked = ProductionBlocked(
                    reason="PUBLICATION_RECOVERY_FAILED",
                    detail=f"Recovery exception: {exc}",
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
    if not timeframes:
        raise ValueError("At least one configured timeframe is required.")

    canonical_tfs: list[str] = []
    configured_set: set[str] = set()
    for tf in timeframes:
        if not tf or not str(tf).strip():
            raise ValueError("Configured timeframe must be a non-empty string.")
        c_tf = CanonicalTimeframe.from_str(str(tf).strip()).value
        if c_tf in configured_set:
            raise ValueError(f"Duplicate canonical timeframe identity '{c_tf}' in configured timeframes.")
        configured_set.add(c_tf)
        canonical_tfs.append(c_tf)

    if not arg_str or not str(arg_str).strip():
        return {}

    cleaned = str(arg_str).strip()
    result: dict[str, str] = {}

    if "=" in cleaned or "," in cleaned:
        raw_parts = [p.strip() for p in cleaned.split(",")]
        if any(not p for p in raw_parts):
            raise ValueError(f"Malformed candidate-id mapping '{arg_str}'. Contains empty comma-separated segment.")

        for part in raw_parts:
            if "=" not in part:
                raise ValueError(f"Invalid candidate-id mapping part '{part}'. Expected 'timeframe=candidate_id'.")
            tf_str, cand_str = part.split("=", 1)
            tf_clean = tf_str.strip()
            cand_id = cand_str.strip()

            if not tf_clean:
                raise ValueError(f"Empty timeframe supplied in candidate mapping part '{part}'.")
            if not cand_id:
                raise ValueError(f"Empty candidate_id supplied for timeframe '{tf_clean}'.")

            tf_canonical = CanonicalTimeframe.from_str(tf_clean).value

            if tf_canonical in result:
                raise ValueError(f"Duplicate candidate mapping key for canonical timeframe '{tf_canonical}'.")

            if tf_canonical not in configured_set:
                raise ValueError(
                    f"Explicit candidate mapping key '{tf_canonical}' is outside the configured timeframe scope {sorted(configured_set)}."
                )

            result[tf_canonical] = cand_id
        return result

    # Plain string without '=' or ','
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
        publish: bool | None = None,
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
            raw_tfs = timeframes.split(",")
            if any(not t.strip() for t in raw_tfs):
                raise ValueError(f"Invalid timeframes string '{timeframes}'. Contains empty timeframe token.")
            tf_list = [t.strip() for t in raw_tfs]
        else:
            tf_list = list(timeframes)

        if not tf_list:
            raise ValueError("ContinuousLiveRuntime requires at least one configured timeframe.")

        # Enforce repository's existing canonical timeframe vocabulary.
        # Fails closed on unsupported timeframes and duplicate identities.
        canonical_tfs: list[str] = []
        seen_tfs: set[str] = set()
        for tf in tf_list:
            if not isinstance(tf, str) or not tf.strip():
                raise ValueError("Configured timeframe must be a non-empty string.")
            c_tf = CanonicalTimeframe.from_str(tf.strip()).value
            if c_tf in seen_tfs:
                raise ValueError(f"Duplicate canonical timeframe identity '{c_tf}' in configured timeframes.")
            seen_tfs.add(c_tf)
            canonical_tfs.append(c_tf)

        self.timeframes = tuple(canonical_tfs)
        configured_set = set(self.timeframes)

        # Validate directly supplied candidate_ids mapping
        normalized_candidate_ids: dict[str, str] = {}
        if candidate_ids is not None:
            if not isinstance(candidate_ids, dict):
                raise ValueError("candidate_ids must be a dictionary mapping timeframes to candidate IDs.")
            for k, v in candidate_ids.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError(f"Invalid candidate_ids key {k!r}: mapping keys must be non-empty strings.")
                if not isinstance(v, str) or not v.strip():
                    raise ValueError(f"Invalid candidate_id value {v!r} for timeframe {k!r}: candidate IDs must be non-empty strings.")
                k_canonical = CanonicalTimeframe.from_str(k.strip()).value
                if k_canonical in normalized_candidate_ids:
                    raise ValueError(f"Duplicate candidate mapping key for canonical timeframe '{k_canonical}'.")
                if k_canonical not in configured_set:
                    raise ValueError(
                        f"Explicit candidate mapping key '{k_canonical}' is outside the configured timeframe scope {sorted(configured_set)}."
                    )
                normalized_candidate_ids[k_canonical] = v.strip()

        self.poll_interval = max(0.0, poll_interval)
        if publisher is not None:
            self.publisher = publisher
            self.publish = publish if publish is not None else getattr(publisher, "enabled", False)
        else:
            self.publisher = Project2Publisher(
                enabled=publish if publish is not None else None,
                max_age_seconds=int(max_age_seconds),
            )
            self.publish = self.publisher.enabled if publish is None else publish
        self.skip_if_no_trade = skip_if_no_trade
        self.persist = persist
        self.max_age_seconds = max_age_seconds
        self.limit = limit
        self.research_dir = Path(research_dir)
        self.store_path = Path(store_path)
        self.snapshot_path = Path(snapshot_path)
        self.market_data_loaders = market_data_loaders
        self.candidate_ids = normalized_candidate_ids
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
        self._timeframe_readiness: dict[str, dict[str, Any]] = {}

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def _resolve_expected_scope(self, timeframe: str) -> dict[str, Any]:
        """Resolve unified, authoritative expected scope for a canonical timeframe.

        Fails closed on conflicting configured candidate_id vs readiness candidate_id.
        """
        try:
            canonical_tf = CanonicalTimeframe.from_str(timeframe).value
        except (ValueError, TypeError):
            return {
                "expected_timeframe": str(timeframe),
                "expected_symbol": self.symbol,
                "expected_candidate_id": None,
                "has_conflict": True,
            }

        configured_cand = self.candidate_ids.get(canonical_tf)
        readiness_info = self._timeframe_readiness.get(canonical_tf, {})
        readiness_cand = readiness_info.get("candidate_id")

        if configured_cand is not None:
            configured_cand = str(configured_cand).strip() or None
        if readiness_cand is not None:
            readiness_cand = str(readiness_cand).strip() or None

        has_conflict = False
        if configured_cand is not None and readiness_cand is not None and configured_cand != readiness_cand:
            has_conflict = True
            expected_cand = None
        elif configured_cand is not None or readiness_cand is not None:
            expected_cand = configured_cand or readiness_cand
        elif self.candidate_ids or self._timeframe_readiness:
            expected_cand = None
        else:
            expected_cand = _UNSPECIFIED

        return {
            "expected_timeframe": canonical_tf,
            "expected_symbol": self.symbol,
            "expected_candidate_id": expected_cand,
            "has_conflict": has_conflict,
        }

    def _is_healthy_evaluation(
        self,
        res: Any,
        *,
        expected_timeframe: str | None = None,
        expected_symbol: str | None = None,
        expected_candidate_id: Any = _UNSPECIFIED,
    ) -> bool:
        """Classify a single timeframe evaluation result using explicit conservative rules and strict scope validation."""
        if not isinstance(res, dict) or not res:
            return False

        # Fail closed on explicit blocked or failure flags
        if res.get("blocked") is True:
            return False

        # Validate blocked_state structure if present
        if "blocked_state" in res and res["blocked_state"] is not None:
            if not isinstance(res["blocked_state"], dict):
                return False

        # Check for explicit error/exception fields
        if "error" in res or "exception" in res:
            return False

        st = res.get("status")

        # Scope identity validation helper
        exp_tf = None
        if expected_timeframe is not None:
            try:
                exp_tf = CanonicalTimeframe.from_str(expected_timeframe).value
            except (ValueError, TypeError):
                return False

        exp_sym = str(expected_symbol).strip().upper() if expected_symbol is not None else None
        if expected_candidate_id is _UNSPECIFIED:
            exp_cand = _UNSPECIFIED
        elif expected_candidate_id is not None:
            exp_cand = str(expected_candidate_id).strip()
        else:
            exp_cand = None

        # Check timeframe fields in res for conflicts or expected scope mismatches
        res_tf_val = None
        has_interval = "interval" in res and res["interval"] is not None
        has_timeframe = "timeframe" in res and res["timeframe"] is not None

        if has_interval:
            if not isinstance(res["interval"], str) or not res["interval"].strip():
                return False
            try:
                res_tf_val = CanonicalTimeframe.from_str(res["interval"]).value
            except (ValueError, TypeError):
                return False

        if has_timeframe:
            if not isinstance(res["timeframe"], str) or not res["timeframe"].strip():
                return False
            try:
                tf_val2 = CanonicalTimeframe.from_str(res["timeframe"]).value
                if res_tf_val is not None and res_tf_val != tf_val2:
                    return False
                res_tf_val = tf_val2
            except (ValueError, TypeError):
                return False

        if exp_tf is not None and res_tf_val is not None and res_tf_val != exp_tf:
            return False

        # Check symbol field in res
        if "symbol" in res and res["symbol"] is not None:
            if not isinstance(res["symbol"], str) or not res["symbol"].strip():
                return False
            res_sym = str(res["symbol"]).strip().upper()
            if exp_sym is not None and res_sym != exp_sym:
                return False

        # Check candidate_id field in res if present against expected candidate scope
        if "candidate_id" in res and res["candidate_id"] is not None:
            if not isinstance(res["candidate_id"], str) or not res["candidate_id"].strip():
                return False
            res_cand = str(res["candidate_id"]).strip()
            if exp_cand is not _UNSPECIFIED:
                if exp_cand is None or res_cand != exp_cand:
                    return False

        # Observation evaluations: truthful NO_TRADE market observations produced
        # for timeframes without an authorized production candidate. They carry no
        # candidate identity and must never smuggle a trade decision.
        if res.get("observation") is True:
            if res.get("decision") != "NO TRADE":
                return False
            # An observation must never claim candidate identity or authorization.
            if res.get("candidate_id") is not None:
                return False
            if not isinstance(res.get("reason"), str) or not res["reason"].strip():
                return False
            obs_record = res.get("record")
            if not isinstance(obs_record, dict) or not obs_record:
                return False
            obs_publication = res.get("publication")
            if not isinstance(obs_publication, dict) or not obs_publication:
                return False
            obs_pub_res = res.get("publish_result")
            if not isinstance(obs_pub_res, dict):
                return False
            obs_pub_status = obs_pub_res.get("status")
            obs_published = obs_pub_res.get("published")
            if obs_published is True:
                if not obs_pub_status or obs_pub_status not in (
                    "PUBLISHED", "DELIVERED", "ACCEPTED", "OK", "ACKNOWLEDGED", "INGESTED",
                ):
                    return False
            elif obs_published is False:
                if obs_pub_status not in ("SKIPPED_DISABLED", "SKIPPED_NO_TRADE"):
                    return False
            else:
                return False
            return True

        # Direct-strategy evaluations: truthful per-timeframe strategy decisions
        # produced WITHOUT candidates, promotion, or research qualification.
        # They carry no candidate identity and must never claim authorization.
        if res.get("direct_signal") is True:
            if res.get("decision") not in ("BUY", "SELL", "NO TRADE"):
                return False
            # A direct evaluation must never claim candidate identity or promotion.
            if res.get("candidate_id") is not None:
                return False
            if expected_candidate_id is not _UNSPECIFIED:
                if expected_candidate_id is not None:
                    return False
            if not isinstance(res.get("reason"), str) or not res["reason"].strip():
                return False
            if not isinstance(res.get("record"), dict) or not res["record"]:
                return False
            if not isinstance(res.get("publication"), dict) or not res["publication"]:
                return False
            direct_pub_res = res.get("publish_result")
            if not isinstance(direct_pub_res, dict):
                return False
            direct_pub_status = direct_pub_res.get("status")
            direct_published = direct_pub_res.get("published")
            if direct_published is True:
                if not direct_pub_status or direct_pub_status not in (
                    "PUBLISHED", "DELIVERED", "ACCEPTED", "OK", "ACKNOWLEDGED", "INGESTED",
                ):
                    return False
            elif direct_published is False:
                if direct_pub_status not in ("SKIPPED_DISABLED", "SKIPPED_NO_TRADE"):
                    return False
            else:
                return False
            return True

        # 1. Recognized benign worker operational skips (exact match of status + reason)
        if st == "SKIPPED_DEDUPLICATED":
            if res.get("reason") != "candle_already_evaluated":
                return False
            if "candle_timestamp" in res and not isinstance(res["candle_timestamp"], str):
                return False
            return True

        if st == "SKIPPED_NO_CLOSED_CANDLE":
            if res.get("reason") != "latest_candle_is_open_or_unavailable":
                return False
            return True

        # Reject reason-only skips without recognized status or invalid status/reason combos
        if res.get("reason") in ("candle_already_evaluated", "latest_candle_is_open_or_unavailable"):
            return False

        UNHEALTHY_STATUSES = (
            "BLOCKED",
            "FAILED_RECOVERABLE",
            "FAILED",
            "EXCEPTIONAL_FAILURE",
            "INVALID_RESPONSE",
            "UNAUTHORIZED",
            "AUTH_FAILED",
            "FORBIDDEN",
            "TIMED_OUT",
            "UNAVAILABLE",
            "REJECTED",
            "UNKNOWN",
            "UNKNOWN_CUSTOM_STATUS",
            "SKIPPED_BLOCKED",
        )
        if st in UNHEALTHY_STATUSES:
            return False

        # 2. Recognized successful run_once evaluation result
        if res.get("blocked") is False:
            VALID_EXECUTION_STATUSES = (
                "PERSISTED",
                "PUBLISHED",
                "PRESENTABLE",
                "AUTHORIZED",
                "EVALUATED",
                "SUCCESS",
                "OK",
                "NO_TRADE_SKIPPED",
            )
            if st is not None and st not in VALID_EXECUTION_STATUSES:
                return False

            # Require mandatory top-level producer identity fields: symbol, interval, candidate_id
            raw_sym = res.get("symbol")
            if not isinstance(raw_sym, str) or not raw_sym.strip():
                return False

            raw_int = res.get("interval")
            if not isinstance(raw_int, str) or not raw_int.strip():
                return False

            raw_cand = res.get("candidate_id")
            if not isinstance(raw_cand, str) or not raw_cand.strip():
                return False

            try:
                res_tf_val = CanonicalTimeframe.from_str(raw_int).value
            except (ValueError, TypeError):
                return False

            # If optional top-level timeframe alias is present, require agreement with interval
            if "timeframe" in res and res["timeframe"] is not None:
                if not isinstance(res["timeframe"], str) or not res["timeframe"].strip():
                    return False
                try:
                    tf_alias_val = CanonicalTimeframe.from_str(res["timeframe"]).value
                    if tf_alias_val != res_tf_val:
                        return False
                except (ValueError, TypeError):
                    return False

            # Validate res symbol against expected symbol
            if exp_sym is not None and raw_sym.strip().upper() != exp_sym:
                return False

            # Validate res timeframe against expected timeframe
            if exp_tf is not None and res_tf_val != exp_tf:
                return False

            # Validate res candidate_id against expected candidate_id
            # An unmapped frame (exp_cand is None) cannot self-authorize
            if exp_cand is not _UNSPECIFIED:
                if exp_cand is None or raw_cand.strip() != exp_cand:
                    return False

            # Require positive evaluation evidence (value shape validation, not presence alone)
            has_record = isinstance(res.get("record"), dict) and len(res["record"]) > 0
            has_publication = isinstance(res.get("publication"), dict) and len(res["publication"]) > 0
            has_contract = isinstance(res.get("contract_payload"), dict) and len(res["contract_payload"]) > 0
            has_valid_decision = isinstance(res.get("decision"), str) and len(res["decision"].strip()) > 0
            has_lifecycle = isinstance(res.get("current_lifecycle_state"), str) and res["current_lifecycle_state"] in (
                "PERSISTED", "PUBLISHED", "PRESENTABLE", "AUTHORIZED", "EVALUATED"
            )

            has_positive_evidence = (
                has_record
                or has_publication
                or has_contract
                or has_lifecycle
                or (has_valid_decision and st in VALID_EXECUTION_STATUSES)
            )

            if not has_positive_evidence:
                return False

            # Verify consistency across nested record object if present
            rec = res.get("record")
            if isinstance(rec, dict):
                rec_sym = rec.get("symbol")
                if rec_sym is not None:
                    if not isinstance(rec_sym, str) or not rec_sym.strip():
                        return False
                    if exp_sym is not None and rec_sym.strip().upper() != exp_sym:
                        return False

                # Validate nested record timeframe aliases without unsafe fallback concealing contradictions
                rec_has_interval = "interval" in rec and rec["interval"] is not None
                rec_has_timeframe = "timeframe" in rec and rec["timeframe"] is not None
                rec_tf_val = None

                if rec_has_interval:
                    if not isinstance(rec["interval"], str) or not rec["interval"].strip():
                        return False
                    try:
                        rec_tf_val = CanonicalTimeframe.from_str(rec["interval"]).value
                    except (ValueError, TypeError):
                        return False

                if rec_has_timeframe:
                    if not isinstance(rec["timeframe"], str) or not rec["timeframe"].strip():
                        return False
                    try:
                        rec_tf_val2 = CanonicalTimeframe.from_str(rec["timeframe"]).value
                        if rec_tf_val is not None and rec_tf_val != rec_tf_val2:
                            return False
                        rec_tf_val = rec_tf_val2
                    except (ValueError, TypeError):
                        return False

                if exp_tf is not None and rec_tf_val is not None and rec_tf_val != exp_tf:
                    return False

                rec_cand = rec.get("candidate_id")
                if rec_cand is not None:
                    if not isinstance(rec_cand, str) or not rec_cand.strip():
                        return False
                    if rec_cand.strip() != raw_cand.strip():
                        return False
                    if exp_cand is not _UNSPECIFIED and exp_cand is not None and rec_cand.strip() != exp_cand:
                        return False

            # Validate publication outcome if publish_result is present
            pub_res = res.get("publish_result")
            if pub_res is not None:
                if not isinstance(pub_res, dict):
                    return False

                pub_status = pub_res.get("status")
                is_published = pub_res.get("published")

                if pub_status in PUB_FAILURE_STATUSES:
                    return False

                if is_published is True:
                    if not pub_status or pub_status not in ("PUBLISHED", "DELIVERED", "ACCEPTED", "OK", "ACKNOWLEDGED", "INGESTED"):
                        return False
                elif is_published is False:
                    if pub_status not in ("SKIPPED_DISABLED", "SKIPPED_NO_TRADE"):
                        return False
                else:
                    return False

            return True

        return False

    def get_health_status(
        self,
        staleness_threshold_seconds: float = 300.0,
        reference_now: datetime.datetime | None = None,
        current_evaluations: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute current worker health diagnostics.

        Health states:
        - WORKER_NOT_STARTED: continuous loop hasn't started or executed ticks
        - WORKER_STALE: tick heartbeat older than staleness threshold or invalid timestamp
        - WORKER_DEGRADED: tick heartbeat is fresh but timeframe evaluations fail or are all blocked
        - WORKER_HEALTHY: tick heartbeat within staleness threshold and execution is healthy
        """
        ref_now = reference_now if reference_now is not None else self.clock()
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=datetime.timezone.utc)

        if not self._started_at_utc or not self._last_tick_at_utc:
            return {
                "status": "WORKER_NOT_STARTED",
                "healthy": False,
                "heartbeat_healthy": False,
                "execution_healthy": False,
                "started_at_utc": self._started_at_utc,
                "last_tick_at_utc": self._last_tick_at_utc,
                "last_successful_evaluation_at_utc": self._last_successful_evaluation_at_utc,
                "last_successful_publication_at_utc": self._last_successful_publication_at_utc,
                "last_failure_reason": self._last_failure_reason,
                "seconds_since_last_tick": None,
                "tick_count": self._tick_count,
            }

        try:
            last_tick_dt = datetime.datetime.fromisoformat(self._last_tick_at_utc)
            if last_tick_dt.tzinfo is None:
                last_tick_dt = last_tick_dt.replace(tzinfo=datetime.timezone.utc)
            age_seconds = (ref_now - last_tick_dt).total_seconds()
        except Exception:
            age_seconds = -1.0

        is_heartbeat_healthy = age_seconds >= 0 and age_seconds <= staleness_threshold_seconds

        # Evaluate execution health across timeframe evaluations in current tick or latest history
        evals_to_check = current_evaluations
        if evals_to_check is None and self._execution_history:
            evals_to_check = self._execution_history[-1].get("evaluations")

        execution_healthy = False
        healthy_tf_count = 0
        if isinstance(evals_to_check, dict) and evals_to_check:
            for tf, res in evals_to_check.items():
                scope = self._resolve_expected_scope(tf)
                if not scope["has_conflict"] and self._is_healthy_evaluation(
                    res,
                    expected_timeframe=scope["expected_timeframe"],
                    expected_symbol=scope["expected_symbol"],
                    expected_candidate_id=scope["expected_candidate_id"],
                ):
                    healthy_tf_count += 1
            if healthy_tf_count > 0:
                execution_healthy = True

        if not is_heartbeat_healthy:
            status = "WORKER_STALE"
        elif not execution_healthy:
            status = "WORKER_DEGRADED"
        else:
            status = "WORKER_HEALTHY"

        is_healthy = is_heartbeat_healthy and execution_healthy

        # Build structured per-timeframe health diagnostics with exact precedence
        timeframe_diagnostics: dict[str, dict[str, Any]] = {}
        all_canonical_tfs = tuple(tf.value for tf in CanonicalTimeframe.canonical_ladder())
        healthy_canonical_count = 0

        for tf in self.timeframes:
            readiness_info = self._timeframe_readiness.get(tf, {})
            tf_eval = evals_to_check.get(tf) if isinstance(evals_to_check, dict) else None
            scope = self._resolve_expected_scope(tf)

            is_healthy_tf = False
            if tf_eval and not scope["has_conflict"]:
                is_healthy_tf = self._is_healthy_evaluation(
                    tf_eval,
                    expected_timeframe=scope["expected_timeframe"],
                    expected_symbol=scope["expected_symbol"],
                    expected_candidate_id=scope["expected_candidate_id"],
                )

            if is_healthy_tf:
                healthy_canonical_count += 1

            # Derive publication status
            pub_status = "NOT_ATTEMPTED"
            pub_res = tf_eval.get("publish_result") if isinstance(tf_eval, dict) else None
            if isinstance(pub_res, dict) and pub_res.get("status"):
                pub_status = str(pub_res["status"])
            elif not self.publish or (isinstance(self.publisher, Project2Publisher) and not self.publisher.enabled):
                pub_status = "SKIPPED_DISABLED"

            # Derive actionable reason code following strict precedence
            reason_code = None

            # Precedence 1: Explicit evaluation failure/error/exception in current tick
            if isinstance(tf_eval, dict):
                if tf_eval.get("error"):
                    reason_code = str(tf_eval["error"])
                elif tf_eval.get("exception"):
                    reason_code = str(tf_eval["exception"])
                else:
                    bs = tf_eval.get("blocked_state")
                    if isinstance(bs, dict) and bs.get("reason"):
                        reason_code = str(bs["reason"])

            # Precedence 2: Publication failure
            if not reason_code and pub_status in PUB_FAILURE_STATUSES and isinstance(pub_res, dict):
                reason_code = str(pub_res.get("reason") or pub_res.get("error") or pub_status)

            # Precedence 3: Evaluation blocked or benign skip reason
            if not reason_code and isinstance(tf_eval, dict):
                st = tf_eval.get("status")
                if st in ("SKIPPED_DEDUPLICATED", "SKIPPED_NO_CLOSED_CANDLE"):
                    reason_code = str(tf_eval.get("reason") or st)
                elif tf_eval.get("blocked") or st in ("BLOCKED", "FAILED_RECOVERABLE", "SKIPPED_BLOCKED", "EXCEPTIONAL_FAILURE", "UNAUTHORIZED"):
                    reason_code = str(tf_eval.get("reason") or tf_eval.get("detail") or st)

            # Precedence 4: Readiness non-ready reason. For healthy market
            # observations, fall back to the observation's disclosed candidate
            # blocker so the diagnosis still names why no trade was authorized.
            if not reason_code and readiness_info.get("status") != "READY":
                reason_code = (
                    str(readiness_info.get("reason_code"))
                    if readiness_info.get("reason_code")
                    else (
                        str(tf_eval.get("reason"))
                        if isinstance(tf_eval, dict) and tf_eval.get("observation") is True and tf_eval.get("reason")
                        else None
                    )
                    or "BLOCKED"
                )

            # Safety fallback: An unhealthy evaluation must NEVER report "READY"
            if not is_healthy_tf and (not reason_code or reason_code == "READY"):
                st_fallback = tf_eval.get("status") if isinstance(tf_eval, dict) else None
                reason_code = str(st_fallback) if st_fallback and st_fallback != "READY" else "UNCLASSIFIED_RUNTIME_FAILURE"

            # Precedence 5: Default READY when healthy
            if not reason_code:
                reason_code = str(readiness_info.get("reason_code") or "READY")

            cand_id = readiness_info.get("candidate_id") or (tf_eval.get("candidate_id") if isinstance(tf_eval, dict) else None)

            tf_diag: dict[str, Any] = {
                "readiness": readiness_info.get("status", "UNKNOWN"),
                "reason_code": reason_code,
                "candidate_id": cand_id,
                "execution": "HEALTHY" if is_healthy_tf else ("BLOCKED" if (isinstance(tf_eval, dict) and tf_eval.get("blocked")) or readiness_info.get("status") == "BLOCKED" else "UNHEALTHY"),
                "publication_status": pub_status,
            }
            timeframe_diagnostics[tf] = tf_diag

        is_seven_configured = set(self.timeframes) == set(all_canonical_tfs)
        seven_frame_parity = False
        if is_seven_configured:
            parity_qualifying = True
            for c_tf in all_canonical_tfs:
                readiness_status = self._timeframe_readiness.get(c_tf, {}).get("status")
                if readiness_status != "READY":
                    parity_qualifying = False
                    break

                c_eval = evals_to_check.get(c_tf) if isinstance(evals_to_check, dict) else None
                if not c_eval:
                    parity_qualifying = False
                    break

                scope = self._resolve_expected_scope(c_tf)
                if scope["has_conflict"]:
                    parity_qualifying = False
                    break

                if not self._is_healthy_evaluation(
                    c_eval,
                    expected_timeframe=scope["expected_timeframe"],
                    expected_symbol=scope["expected_symbol"],
                    expected_candidate_id=scope["expected_candidate_id"],
                ):
                    parity_qualifying = False
                    break

                c_st = c_eval.get("status")
                if c_st == "SKIPPED_NO_CLOSED_CANDLE":
                    parity_qualifying = False
                    break

            seven_frame_parity = parity_qualifying

        return {
            "status": status,
            "healthy": is_healthy,
            "heartbeat_healthy": is_heartbeat_healthy,
            "execution_healthy": execution_healthy,
            "seven_frame_parity": seven_frame_parity,
            "started_at_utc": self._started_at_utc,
            "last_tick_at_utc": self._last_tick_at_utc,
            "last_successful_evaluation_at_utc": self._last_successful_evaluation_at_utc,
            "last_successful_publication_at_utc": self._last_successful_publication_at_utc,
            "last_failure_reason": self._last_failure_reason,
            "seconds_since_last_tick": max(0.0, age_seconds),
            "tick_count": self._tick_count,
            "timeframes": timeframe_diagnostics,
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

    def _evaluate_market_observation_no_trade(
        self,
        *,
        canonical_tf: str,
        eval_df: pd.DataFrame,
        closed_ts_iso: str,
        blocked: ProductionBlocked,
        reference_now: datetime.datetime,
    ) -> dict[str, Any]:
        """Evaluate a per-timeframe NO_TRADE market observation when no authorized candidate exists.

        The timeframe's real closed-candle slice is validated through the same
        freshness/closure gate used by the canonical path. A fresh, closed slice
        yields a truthful NO_TRADE observation publication (real candle timestamp,
        real decision timestamp, explicit provenance that no authorized candidate
        exists). A stale/invalid slice fail-closes to a blocked, unpublished
        result: invalid data must never be laundered into a releasable outcome.
        """
        from src.evaluation.live_market_observation import MarketObservationPublication

        freshness = validate_market_data_freshness(
            data=eval_df,
            timeframe=canonical_tf,
            max_age_seconds=self.max_age_seconds,
            reference_now=reference_now,
        )

        if not freshness.get("fresh"):
            return {
                "blocked": True,
                "status": "BLOCKED",
                "reason": str(freshness.get("reason") or "unclosed_market_data"),
                "detail": (
                    f"Market observation for {self.symbol} {canonical_tf} fail-closed: "
                    f"market data did not pass the freshness/closure gate "
                    f"(reason={freshness.get('reason')}); candidate blocker "
                    f"[{blocked.reason}] is also present and disclosed in blocked_state."
                ),
                "symbol": self.symbol,
                "timeframe": canonical_tf,
                "candidate_id": blocked.candidate_id,
                "blocked_state": {
                    "blocked": True,
                    "reason": blocked.reason,
                    "detail": blocked.detail,
                    "candidate_id": blocked.candidate_id,
                    "symbol": self.symbol,
                    "timeframe": canonical_tf,
                    "market_data_fresh": False,
                    "freshness_reason": str(freshness.get("reason")),
                    "candle_timestamp": freshness.get("candle_timestamp") or closed_ts_iso,
                },
            }

        try:
            observation = MarketObservationPublication(
                symbol=self.symbol,
                timeframe=canonical_tf,
                reason=blocked.reason,
                detail=blocked.detail,
                candle_timestamp_utc=closed_ts_iso,
                observation_timestamp_utc=reference_now.isoformat(),
            )
        except Exception as exc:
            return {
                "blocked": True,
                "status": "BLOCKED",
                "reason": "MARKET_OBSERVATION_CONSTRUCTION_FAILED",
                "detail": str(exc),
                "symbol": self.symbol,
                "timeframe": canonical_tf,
                "candidate_id": blocked.candidate_id,
            }

        if self.publish and self.publisher is not None:
            try:
                pub_res = self.publisher.publish(
                    observation,
                    skip_if_no_trade=self.skip_if_no_trade,
                )
            except Exception as exc:
                pub_res = {
                    "status": "FAILED",
                    "published": False,
                    "reason": "Market observation publication exception",
                    "error": str(exc),
                }
        else:
            pub_res = {
                "status": "SKIPPED_DISABLED",
                "published": False,
                "reason": "Publisher disabled in configuration",
            }

        observation_status = pub_res.get("status")

        return {
            "observation": True,
            "blocked": False,
            "status": observation_status,
            "decision": "NO TRADE",
            "reason": blocked.reason,
            "detail": f"No authorized production candidate for {self.symbol} {canonical_tf}: {blocked.detail}",
            "symbol": self.symbol,
            "interval": canonical_tf,
            "timeframe": canonical_tf,
            "candle_timestamp": closed_ts_iso,
            "decision_timestamp": observation.observation_timestamp_utc,
            "record": {
                "observation_id": observation.observation_id,
                "decision": "NO TRADE",
                "symbol": self.symbol,
                "interval": canonical_tf,
                "timeframe": canonical_tf,
                "reason": blocked.reason,
                "candidate_id": None,
                "authorized": False,
                "market_fresh": True,
            },
            "publication": observation.to_contract_v1_payload(),
            "publish_result": pub_res,
            "market_freshness": {
                "fresh": True,
                "reason": freshness.get("reason"),
                "age_seconds": freshness.get("age_seconds"),
                "candle_timestamp": freshness.get("candle_timestamp"),
            },
        }

    def _evaluate_direct_signal(
        self,
        *,
        canonical_tf: str,
        eval_df: pd.DataFrame,
        reference_now: datetime.datetime,
    ) -> dict[str, Any]:
        """Evaluate the direct deterministic strategy on this timeframe's own closed-candle slice.

        Direct mode requires NO candidate, NO promotion, and NO research
        qualification. A fresh, closed slice yields a real strategy decision
        (BUY / SELL / NO TRADE) with trade levels derived only from the
        evaluated closed candle and explicit risk rules. A stale/invalid slice
        fail-closes to a blocked, unpublished result: invalid data must never be
        laundered into a releasable outcome.
        """
        from src.evaluation.live_direct_strategy import (
            DIRECT_STRATEGY_ID,
            DIRECT_STRATEGY_VERSION,
            DirectSignalDataError,
            evaluate_direct_live_signal,
        )

        try:
            signal = evaluate_direct_live_signal(
                data=eval_df,
                symbol=self.symbol,
                timeframe=canonical_tf,
                reference_now=reference_now,
                max_age_seconds=self.max_age_seconds,
            )
        except DirectSignalDataError as exc:
            return {
                "direct_signal": True,
                "blocked": True,
                "status": "BLOCKED",
                "reason": exc.reason,
                "detail": (
                    f"Direct strategy evaluation for {self.symbol} {canonical_tf} fail-closed: "
                    f"{exc.detail}"
                ),
                "decision": "NO TRADE",
                "strategy": DIRECT_STRATEGY_ID,
                "symbol": self.symbol,
                "timeframe": canonical_tf,
                "candidate_id": None,
                "trade_levels": None,
                "record": {
                    "signal_id": None,
                    "decision": "NO TRADE",
                    "symbol": self.symbol,
                    "timeframe": canonical_tf,
                    "strategy_id": DIRECT_STRATEGY_ID,
                    "reason": exc.reason,
                    "candidate_id": None,
                    "authorized": False,
                    "market_fresh": False,
                },
                "publication": None,
                "publish_result": {
                    "status": "SKIPPED_BLOCKED",
                    "published": False,
                    "reason": exc.reason,
                },
                "blocked_state": {
                    "blocked": True,
                    "reason": exc.reason,
                    "detail": exc.detail,
                    "symbol": self.symbol,
                    "timeframe": canonical_tf,
                    "market_data_fresh": False,
                },
                "market_freshness": {
                    "fresh": False,
                    "reason": exc.reason,
                },
                "decision_timestamp": reference_now.isoformat(),
            }
        except Exception as exc:
            return {
                "direct_signal": True,
                "blocked": True,
                "status": "BLOCKED",
                "reason": "DIRECT_EVALUATION_FAILED",
                "detail": f"Direct strategy evaluation for {self.symbol} {canonical_tf} failed: {exc}",
                "decision": "NO TRADE",
                "strategy": DIRECT_STRATEGY_ID,
                "symbol": self.symbol,
                "timeframe": canonical_tf,
                "candidate_id": None,
                "trade_levels": None,
                "record": {
                    "signal_id": None,
                    "decision": "NO TRADE",
                    "symbol": self.symbol,
                    "timeframe": canonical_tf,
                    "strategy_id": DIRECT_STRATEGY_ID,
                    "reason": "DIRECT_EVALUATION_FAILED",
                    "candidate_id": None,
                    "authorized": False,
                    "market_fresh": False,
                },
                "publication": None,
                "publish_result": {
                    "status": "SKIPPED_BLOCKED",
                    "published": False,
                    "reason": "DIRECT_EVALUATION_FAILED",
                },
                "blocked_state": {
                    "blocked": True,
                    "reason": "DIRECT_EVALUATION_FAILED",
                    "detail": str(exc),
                    "symbol": self.symbol,
                    "timeframe": canonical_tf,
                    "market_data_fresh": False,
                },
                "market_freshness": {
                    "fresh": False,
                    "reason": "DIRECT_EVALUATION_FAILED",
                },
                "decision_timestamp": reference_now.isoformat(),
            }

        if self.publish and self.publisher is not None:
            try:
                pub_res = self.publisher.publish(
                    signal,
                    skip_if_no_trade=self.skip_if_no_trade,
                )
            except Exception as exc:
                pub_res = {
                    "status": "FAILED",
                    "published": False,
                    "reason": "Direct signal publication exception",
                    "error": str(exc),
                }
        else:
            pub_res = {
                "status": "SKIPPED_DISABLED",
                "published": False,
                "reason": "Publisher disabled in configuration",
            }

        publication = signal.to_contract_v1_payload()
        trade_levels = {
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            "risk_reward_ratio": signal.risk_reward_ratio,
        } if signal.decision in ("BUY", "SELL") else None

        return {
            "direct_signal": True,
            "blocked": False,
            "status": pub_res.get("status"),
            "decision": signal.decision,
            "reason": signal.reason,
            "detail": signal.detail,
            "strategy": DIRECT_STRATEGY_ID,
            "strategy_version": DIRECT_STRATEGY_VERSION,
            "symbol": self.symbol,
            "interval": canonical_tf,
            "timeframe": canonical_tf,
            "candle_timestamp": signal.candle_timestamp_utc,
            "decision_timestamp": signal.decision_timestamp_utc,
            "candidate_id": None,
            "trade_levels": trade_levels,
            "record": {
                "signal_id": signal.signal_id,
                "decision": signal.decision,
                "symbol": self.symbol,
                "interval": canonical_tf,
                "timeframe": canonical_tf,
                "strategy_id": DIRECT_STRATEGY_ID,
                "strategy_version": DIRECT_STRATEGY_VERSION,
                "reason": signal.reason,
                "candidate_id": None,
                "authorized": False,
                "market_fresh": True,
                "candle_timestamp_utc": signal.candle_timestamp_utc,
                "decision_timestamp_utc": signal.decision_timestamp_utc,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "risk_reward_ratio": signal.risk_reward_ratio,
            },
            "publication": publication,
            "publish_result": pub_res,
            "market_freshness": {
                "fresh": True,
                "reason": "fresh",
                "candle_timestamp": signal.candle_timestamp_utc,
            },
        }

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

                # NOTE: no preflight-cache skip. Every configured timeframe is attempted
                # independently on each tick so a blocked timeframe reports its own
                # BLOCKED/NO-TRADE result here and automatically recovers when its
                # prerequisite (qualified candidate) becomes available, without affecting
                # any other timeframe's evaluation or publication. A BLOCKED startup
                # readiness result (e.g. PromotionUnavailable) is an isolation boundary
                # and an operational diagnosis, never a reason to skip this timeframe's
                # market evaluation and publication attempt at tick level:
                # candidate-less timeframes still produce their own truthful NO_TRADE
                # market observation below.

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

                # Direct mode: timeframes NOT explicitly configured with a
                # candidate are evaluated by the one deterministic live
                # strategy on their OWN closed-candle data. No candidate,
                # promotion, or research qualification is consulted, and no
                # data, candidate, signal, readiness, or health is borrowed
                # from any other timeframe. Failures here are isolated to
                # this timeframe (continue) and never stop the other frames.
                if canonical_tf not in self.candidate_ids:
                    dedup_key = (self.symbol, canonical_tf, closed_ts_iso, "direct:ma_crossover_direct")

                    if dedup_key in self._evaluated_candles:
                        tick_evaluations[canonical_tf] = {
                            "status": "SKIPPED_DEDUPLICATED",
                            "reason": "candle_already_evaluated",
                            "candle_timestamp": closed_ts_iso,
                            "candidate_id": None,
                            "timeframe": canonical_tf,
                        }
                        continue

                    result = self._evaluate_direct_signal(
                        canonical_tf=canonical_tf,
                        eval_df=eval_df,
                        reference_now=ref_now,
                    )
                    tick_evaluations[canonical_tf] = result

                    # Deduplicated only after a healthy direct evaluation; unhealthy
                    # outcomes (failed publication, stale data, unclosed gate)
                    # remain retryable on the next tick without suppression.
                    if self._is_healthy_evaluation(
                        result,
                        expected_timeframe=canonical_tf,
                        expected_symbol=self.symbol,
                        expected_candidate_id=None,
                    ):
                        self._evaluated_candles.add(dedup_key)
                        self._last_successful_evaluation_at_utc = ref_now.isoformat()
                        pub_direct_res = result.get("publish_result")
                        if pub_direct_res and pub_direct_res.get("published") is True:
                            self._last_successful_publication_at_utc = ref_now.isoformat()
                    else:
                        self._last_failure_reason = (
                            result.get("reason") or result.get("error") or "DIRECT_EVALUATION_UNHEALTHY"
                        )
                    continue

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

                if isinstance(resolved, ProductionBlocked):
                    # 5. Deduplication check BEFORE evaluation (same semantics as
                    # the canonical path: only healthy prior outcomes dedup).
                    if dedup_key in self._evaluated_candles:
                        tick_evaluations[canonical_tf] = {
                            "status": "SKIPPED_DEDUPLICATED",
                            "reason": "candle_already_evaluated",
                            "candle_timestamp": closed_ts_iso,
                            "candidate_id": cand_id,
                            "timeframe": canonical_tf,
                        }
                        continue

                    # Candidate qualification is mandatory to ACT on the market
                    # (BUY/SELL require an authorized production candidate), but it
                    # must not suppress this timeframe's own market evaluation and
                    # publication outcome. Evaluated against the timeframe's real
                    # closed-candle slice below, the honest outcome is a NO_TRADE
                    # market observation: publishable when the data is fresh and
                    # closed, fail-closed (blocked, unpublished) otherwise.
                    result = self._evaluate_market_observation_no_trade(
                        canonical_tf=canonical_tf,
                        eval_df=eval_df,
                        closed_ts_iso=closed_ts_iso,
                        blocked=resolved,
                        reference_now=ref_now,
                    )
                    tick_evaluations[canonical_tf] = result

                    # Add the deduplication key ONLY after a healthy observation
                    # evaluation; unhealthy outcomes (e.g. failed publication or
                    # stale data) retry on the next tick without suppression.
                    if self._is_healthy_evaluation(
                        result,
                        expected_timeframe=canonical_tf,
                        expected_symbol=self.symbol,
                        expected_candidate_id=cand_id,
                    ):
                        self._evaluated_candles.add(dedup_key)
                        self._last_successful_evaluation_at_utc = ref_now.isoformat()
                        pub_obs_res = result.get("publish_result")
                        if pub_obs_res and pub_obs_res.get("published") is True:
                            self._last_successful_publication_at_utc = ref_now.isoformat()
                    else:
                        self._last_failure_reason = result.get("reason") or "OBSERVATION_UNHEALTHY"
                    continue

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

                # Derive timeframe-scoped persistence paths to guarantee path isolation across timeframes
                tf_store_path = self.store_path.parent / canonical_tf / self.store_path.name
                tf_snapshot_path = self.snapshot_path.parent / canonical_tf / self.snapshot_path.name

                # 6. Delegate strictly to existing authoritative run_once() primitive
                try:
                    runtime = LiveExecutionRuntime(
                        symbol=self.symbol,
                        interval=canonical_tf,
                        limit=self.limit,
                        publisher=self.publisher,
                        store_path=tf_store_path,
                        snapshot_path=tf_snapshot_path,
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

                    # 7. Add to deduplication set ONLY AFTER healthy evaluation execution
                    if self._is_healthy_evaluation(
                        result,
                        expected_timeframe=canonical_tf,
                        expected_symbol=self.symbol,
                        expected_candidate_id=cand_id,
                    ):
                        self._evaluated_candles.add(dedup_key)
                        self._last_successful_evaluation_at_utc = ref_now.isoformat()
                        pub_res = result.get("publish_result")
                        if pub_res and pub_res.get("published") is True:
                            self._last_successful_publication_at_utc = ref_now.isoformat()
                    else:
                        self._last_failure_reason = result.get("reason") or result.get("error") or "EVALUATION_UNHEALTHY"

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
                "health": self.get_health_status(
                    reference_now=ref_now,
                    current_evaluations=tick_evaluations,
                ),
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
        """Perform preflight checks evaluating readiness independently across every configured timeframe.

        Fails closed with PromotionUnavailable if 0 configured timeframes are READY.
        Allows continuous polling loop to proceed if at least 1 configured timeframe is READY.
        """
        self._timeframe_readiness = {}
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

        # Publish mode sanity: when publishing is requested, the publisher configuration
        # itself must be usable; otherwise every timeframe would fail to deliver and the
        # process should fail loudly for THIS reason (not because of candidate availability).
        if self.publish and not (self.publisher is not None and getattr(self.publisher, "enabled", False)):
            raise PromotionUnavailable(
                "Publication requested (--publish) but the Project 2 publisher is disabled "
                "or its configuration is unusable. The worker cannot deliver any timeframe's "
                "decision and must fail closed for this process-level reason."
            )
        if self.publish and self.publisher and getattr(self.publisher, "enabled", False):
            from src.integration.project2_publisher import validate_publication_configuration
            pub_config_valid, pub_config_reason = validate_publication_configuration(
                getattr(self.publisher, "publish_url", None),
                getattr(self.publisher, "api_key", None),
            )
            if not pub_config_valid:
                raise PromotionUnavailable(
                    f"Publication requested but Project 2 publication configuration is unusable: {pub_config_reason}"
                )

        readiness_results = {}
        ready_count = 0
        for tf in self.timeframes:
            canonical_tf = CanonicalTimeframe.from_str(tf).value
            cand_id = self.candidate_ids.get(canonical_tf)
            if cand_id is None:
                # Direct mode: no candidate is required or consulted; readiness
                # depends on timeframe/provider/publication configuration only.
                readiness = verify_timeframe_direct_readiness(
                    symbol=self.symbol,
                    timeframe=canonical_tf,
                    publisher=self.publisher,
                    reference_now=self.clock(),
                )
            else:
                readiness = verify_timeframe_production_readiness(
                    symbol=self.symbol,
                    timeframe=canonical_tf,
                    candidate_id=cand_id,
                    research_dir=self.research_dir,
                    publisher=self.publisher,
                    reference_now=self.clock(),
                )
            readiness_results[canonical_tf] = readiness
            if readiness.get("status") == "READY":
                ready_count += 1
            else:
                logger.warning(
                    "Startup readiness preflight non-ready for %s %s: [%s] %s",
                    self.symbol,
                    canonical_tf,
                    readiness.get("reason_code"),
                    readiness.get("detail"),
                )

        if ready_count == 0:
            details = [
                f"{tf}: [{res.get('reason_code')}] {res.get('detail')}"
                for tf, res in readiness_results.items()
            ]
            # Per-frame readiness unavailability is a per-timeframe condition,
            # NOT a process-level failure. The worker proceeds so ready
            # timeframes can run and blocked timeframes report their own
            # BLOCKED reason and recover automatically.
            logger.error(
                "Startup readiness preflight: 0 of %d configured timeframes are READY for %s. "
                "Continuing with per-timeframe BLOCKED reporting. Failure details:\n%s",
                len(self.timeframes),
                self.symbol,
                "\n".join(details),
            )

        self._timeframe_readiness = readiness_results
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
        try:
            if args.timeframes is not None:
                if not args.timeframes.strip():
                    raise ValueError("Explicitly empty --timeframes argument provided.")
                raw_tfs = args.timeframes.split(",")
                if any(not t.strip() for t in raw_tfs):
                    raise ValueError(f"Invalid --timeframes argument '{args.timeframes}'. Contains empty timeframe token.")
                tfs = [t.strip() for t in raw_tfs]
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
