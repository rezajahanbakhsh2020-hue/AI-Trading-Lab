"""Direct deterministic live-signal strategy for canonical XAUUSD timeframes.

Simplification contract (Issue #100):
- Each canonical timeframe (1m, 5m, 15m, 30m, 1H, 4H, 1D) is evaluated
  independently against its OWN market data via ONE transparent, deterministic
  rule: a fast/high moving-average crossover of closes.
- Live signal generation requires NO candidate_id, NO persisted promotion, NO
  governance fingerprint, and NO research qualification.
- BUY/SELL are emitted only when the rule is met on a closed, fresh candle;
  every other outcome is NO TRADE with a machine-readable reason.
- All trade levels (entry, stop-loss, take-profit, risk/reward) derive solely
  from real closed-candle values and explicit risk rules. NO TRADE emits null
  levels. No prices, timestamps, confidence, or stability values are invented.
- Stale, unclosed, invalid, or missing market data fail closed:
  DirectSignalDataError is raised and the caller must NOT publish.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.evaluation.live_execution_runtime import (
    validate_and_prepare_market_snapshot,
    validate_market_data_freshness,
)
from src.evaluation.live_market_observation import validate_utc_iso8601

DIRECT_STRATEGY_ID = "ma_crossover_direct"
DIRECT_STRATEGY_VERSION = "1.0"
DIRECT_PROVENANCE_TYPE = "direct_strategy_signal"

FAST_MA_PERIOD = 5
SLOW_MA_PERIOD = 20
RISK_REWARD_RATIO = 2.0

BUY = "BUY"
SELL = "SELL"
NO_TRADE = "NO TRADE"


class DirectSignalDataError(Exception):
    """Fail-closed error raised when market data is unusable for live decisions.

    Carries a machine-readable reason (stale_market_data, missing_market_data,
    invalid_market_data) that callers surface verbatim without fabricating a trade.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def _ma(values: pd.Series | list[float], end: int, period: int) -> float:
    """Arithmetic mean of the `period` closes ending at (exclusive) index `end`."""
    window = values[end - period : end]
    return float(sum(window) / period)


@dataclass(frozen=True)
class DirectSignalPublication:
    """Immutable Contract v1.0 event for a direct-strategy live decision.

    Event time is the provider's real closed-candle (open) timestamp; the
    decision/publication time is separate and explicit. BUY/SELL carry trade
    levels derived only from the evaluated candle and the explicit risk rule.
    NO TRADE carries null levels and never references candidates or promotion.
    """

    symbol: str
    timeframe: str
    decision: str
    reason: str
    detail: str
    candle_timestamp_utc: str
    decision_timestamp_utc: str
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward_ratio: float | None = None
    signal_id: str = field(init=False)

    def __post_init__(self) -> None:
        req_symbol = str(self.symbol).strip().upper() if self.symbol else ""
        req_timeframe = str(self.timeframe).strip() if self.timeframe else ""
        req_decision = str(self.decision).strip() if self.decision else ""
        req_reason = str(self.reason).strip() if self.reason else ""
        req_detail = str(self.detail).strip() if self.detail else ""

        if req_symbol != req_symbol.upper() or not req_symbol:
            raise ValueError("symbol must be a non-empty uppercase string.")
        if not req_timeframe:
            raise ValueError("timeframe must be a non-empty string.")
        if req_decision not in (BUY, SELL, NO_TRADE):
            raise ValueError(f"decision must be one of {BUY!r}, {SELL!r}, {NO_TRADE!r}.")
        if not req_reason:
            raise ValueError("reason must be a non-empty machine-readable string.")

        is_trade = req_decision in (BUY, SELL)
        trade_fields = (self.entry_price, self.stop_loss, self.take_profit, self.risk_reward_ratio)
        if is_trade and any(v is None for v in trade_fields):
            raise ValueError(f"{req_decision} requires non-null entry_price, stop_loss, take_profit, risk_reward_ratio.")
        if not is_trade and any(v is not None for v in trade_fields):
            raise ValueError(f"{NO_TRADE} must carry null trade levels.")

        for name in ("entry_price", "stop_loss", "take_profit"):
            val = getattr(self, name)
            if val is not None and (not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0):
                raise ValueError(f"{name} must be a positive finite real price when present.")

        if self.risk_reward_ratio is not None:
            if not isinstance(self.risk_reward_ratio, (int, float)) or isinstance(self.risk_reward_ratio, bool) or self.risk_reward_ratio <= 0:
                raise ValueError("risk_reward_ratio must be a positive finite number when present.")

        object.__setattr__(self, "symbol", req_symbol)
        object.__setattr__(self, "timeframe", req_timeframe)
        object.__setattr__(self, "decision", req_decision)
        object.__setattr__(self, "reason", req_reason)
        object.__setattr__(self, "detail", req_detail)

        object.__setattr__(self, "candle_timestamp_utc", validate_utc_iso8601("candle_timestamp_utc", self.candle_timestamp_utc))
        object.__setattr__(self, "decision_timestamp_utc", validate_utc_iso8601("decision_timestamp_utc", self.decision_timestamp_utc))

        object.__setattr__(self, "signal_id", self._derive_signal_id())

    def _derive_signal_id(self) -> str:
        raw = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "decision": self.decision,
            "reason": self.reason,
            "candle_timestamp_utc": self.candle_timestamp_utc,
            "decision_timestamp_utc": self.decision_timestamp_utc,
            "strategy_id": DIRECT_STRATEGY_ID,
            "strategy_version": DIRECT_STRATEGY_VERSION,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
        }
        serialized = json.dumps(raw, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:32]

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "decision": self.decision,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "reason": self.reason,
            "detail": self.detail,
            "strategy_id": DIRECT_STRATEGY_ID,
            "strategy_version": DIRECT_STRATEGY_VERSION,
            "candle_timestamp_utc": self.candle_timestamp_utc,
            "decision_timestamp_utc": self.decision_timestamp_utc,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_reward_ratio": self.risk_reward_ratio,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "AI-Trading-Lab",
            "provenance_type": DIRECT_PROVENANCE_TYPE,
            "is_live": True,
            "produced_at": self.decision_timestamp_utc,
            "candidate_id": None,
            "strategy_name": DIRECT_STRATEGY_ID,
            "strategy_version": DIRECT_STRATEGY_VERSION,
            "operational_stability_score": None,
            "authorized": False,
            "authorization_state": "direct_strategy_not_promotion_gated",
            "blocked_reason_code": None,
            "blocked_detail": None,
            "publication_contract_version": "1.0",
        }

    def to_contract_v1_payload(self) -> dict[str, Any]:
        """Convert the direct signal into a Contract v1.0 payload for Project 2."""
        trend = "NEUTRAL"
        if self.decision == BUY:
            trend = "BULLISH"
        elif self.decision == SELL:
            trend = "BEARISH"

        signal_id = self.signal_id
        payload = {
            "contract_version": "1.0",
            "event_id": signal_id,
            "event_type": "TRADING_SIGNAL",
            "timestamp": self.candle_timestamp_utc,
            "instrument": {
                "symbol": self.symbol,
                "interval": self.timeframe,
            },
            "signal": {
                "publication_id": signal_id,
                "signal_id": signal_id,
                "decision_id": signal_id,
                "decision": self.decision,
                "strategy": DIRECT_STRATEGY_ID,
                "candidate_id": None,
                "confidence": None,
                "stability_score": None,
                "invalidation": self.stop_loss,
                "signal_label": self.decision,
                "trend": trend,
                "reason": self.reason,
            },
            "trade_setup": {
                "entry_price": self.entry_price,
                "stop_loss": self.stop_loss,
                "tp1": None,
                "tp2": None,
                "tp3": None,
                "take_profit": self.take_profit,
                "risk_reward_ratio": self.risk_reward_ratio,
                "trailing_stop": None,
            },
            "provenance": self.provenance(),
        }
        return payload


def evaluate_direct_live_signal(
    data: pd.DataFrame,
    *,
    symbol: str = "XAUUSD",
    timeframe: str,
    reference_now: datetime.datetime | None = None,
    max_age_seconds: float,
) -> DirectSignalPublication:
    """Evaluate the direct MA-crossover rule on ONE timeframe's own closed-candle data.

    Deterministic rule (explicit parameters):
        fast_ma  = mean of the last FAST_MA_PERIOD closed-candle close prices
        slow_ma  = mean of the last SLOW_MA_PERIOD closed-candle close prices
        bullish crossover (BUY):  fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now
        bearish crossover (SELL): fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now
        otherwise NO TRADE with reason "no_crossover".

    Risk rule (explicit parameters):
        entry  = close of the evaluated closed candle
        stop   = candle low (BUY) or candle high (SELL)
        reward = RISK_REWARD_RATIO * risk, risk = |entry - stop|
        Degenerate zero-risk geometry yields NO TRADE ("invalid_risk_geometry").

    Fail-closed: stale/future/unclosed/invalid/missing data raise DirectSignalDataError
    and never produce a publication.
    """
    symbol_norm = str(symbol).strip().upper()
    if not symbol_norm:
        raise DirectSignalDataError("invalid_symbol", "symbol must be a non-empty string.")
    try:
        prepared = validate_and_prepare_market_snapshot(data)
    except ValueError as exc:
        raise DirectSignalDataError("invalid_market_data", str(exc)) from exc

    freshness = validate_market_data_freshness(
        data=prepared,
        timeframe=timeframe,
        max_age_seconds=max_age_seconds,
        reference_now=reference_now,
    )
    if not freshness.get("fresh"):
        raise DirectSignalDataError(
            str(freshness.get("reason") or "stale_market_data"),
            f"Market data did not pass the freshness/closure gate for {timeframe}.",
        )

    now_dt = reference_now if reference_now is not None else datetime.datetime.now(datetime.timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=datetime.timezone.utc)

    candle = prepared.iloc[-1]
    closes = [float(v) for v in prepared["close"].tolist()]

    if len(closes) < SLOW_MA_PERIOD + 1:
        return DirectSignalPublication(
            symbol=symbol_norm,
            timeframe=timeframe,
            decision=NO_TRADE,
            reason="insufficient_history",
            detail=(
                f"Need >= {SLOW_MA_PERIOD + 1} closed candles for the MA-crossover rule; "
                f"received {len(closes)}."
            ),
            candle_timestamp_utc=_iso(candle["timestamp"]),
            decision_timestamp_utc=now_dt.isoformat(),
        )

    fast_now = _ma(closes, len(closes), FAST_MA_PERIOD)
    fast_prev = _ma(closes, len(closes) - 1, FAST_MA_PERIOD)
    slow_now = _ma(closes, len(closes), SLOW_MA_PERIOD)
    slow_prev = _ma(closes, len(closes) - 1, SLOW_MA_PERIOD)

    entry = float(candle["close"])
    if fast_prev <= slow_prev and fast_now > slow_now:
        decision = BUY
        stop = float(candle["low"])
    elif fast_prev >= slow_prev and fast_now < slow_now:
        decision = SELL
        stop = float(candle["high"])
    else:
        decision = NO_TRADE
        stop = None

    if decision == NO_TRADE:
        return DirectSignalPublication(
            symbol=symbol_norm,
            timeframe=timeframe,
            decision=NO_TRADE,
            reason="no_crossover",
            detail=(
                f"No MA crossover on {timeframe} closed candle "
                f"(fast={fast_now:.6f}, slow={slow_now:.6f}, fast_prev={fast_prev:.6f}, slow_prev={slow_prev:.6f})."
            ),
            candle_timestamp_utc=_iso(candle["timestamp"]),
            decision_timestamp_utc=now_dt.isoformat(),
        )

    risk = abs(entry - stop)
    if risk <= 0:
        return DirectSignalPublication(
            symbol=symbol_norm,
            timeframe=timeframe,
            decision=NO_TRADE,
            reason="invalid_risk_geometry",
            detail=f"Evaluation candle for {timeframe} yields zero stop distance (entry={entry}, stop={stop}).",
            candle_timestamp_utc=_iso(candle["timestamp"]),
            decision_timestamp_utc=now_dt.isoformat(),
        )

    reward = risk * RISK_REWARD_RATIO
    take_profit = entry + reward if decision == BUY else entry - reward

    reason = "ma_crossover_bullish" if decision == BUY else "ma_crossover_bearish"
    return DirectSignalPublication(
        symbol=symbol_norm,
        timeframe=timeframe,
        decision=decision,
        reason=reason,
        detail=(
            f"{reason} on {timeframe} closed candle "
            f"(fast={fast_now:.6f}, slow={slow_now:.6f}, fast_prev={fast_prev:.6f}, slow_prev={slow_prev:.6f})."
        ),
        candle_timestamp_utc=_iso(candle["timestamp"]),
        decision_timestamp_utc=now_dt.isoformat(),
        entry_price=entry,
        stop_loss=stop,
        take_profit=take_profit,
        risk_reward_ratio=RISK_REWARD_RATIO,
    )


def _iso(ts: Any) -> str:
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().isoformat()
    if isinstance(ts, datetime.datetime):
        return ts.isoformat()
    return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).isoformat()
