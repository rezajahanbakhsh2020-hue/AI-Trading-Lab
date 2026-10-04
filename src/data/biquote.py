from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BIQUOTE_BASE_URL = "https://biquote.io"
XAUUSD_SYMBOL = "XAUUSD"

SUPPORTED_INTERVALS = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
)

DEFAULT_INTERVAL = "5m"
DEFAULT_LIMIT = 200
MIN_LIMIT = 1
MAX_LIMIT = 1000
DEFAULT_TIMEOUT = 10


def _get_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "AI-Trading-Lab/1.0",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise RuntimeError(
            f"BiQuote HTTP error: {exc.code} {exc.reason}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Unable to reach BiQuote: {exc.reason}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BiQuote returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("BiQuote returned an invalid response structure.")

    return payload


def fetch_xauusd_ohlc(
    interval: str = DEFAULT_INTERVAL,
    limit: int = DEFAULT_LIMIT,
    timeout: int = DEFAULT_TIMEOUT,
) -> pd.DataFrame:
    """Fetch real XAU/USD OHLC candles from BiQuote."""

    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval: {interval}. "
            f"Supported intervals: {', '.join(SUPPORTED_INTERVALS)}"
        )

    if not isinstance(limit, int):
        raise ValueError("limit must be an integer.")

    if not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise ValueError(
            f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}."
        )

    params = urlencode(
        {
            "interval": interval,
            "limit": limit,
        }
    )

    url = (
        f"{BIQUOTE_BASE_URL}/api/"
        f"{XAUUSD_SYMBOL}/ohlc?{params}"
    )

    payload = _get_json(url, timeout=timeout)

    if payload.get("symbol") != XAUUSD_SYMBOL:
        raise RuntimeError("BiQuote returned an unexpected symbol.")

    if payload.get("interval") != interval:
        raise RuntimeError("BiQuote returned an unexpected interval.")

    bars = payload.get("bars")

    if not isinstance(bars, list) or not bars:
        raise RuntimeError("BiQuote returned no XAU/USD candles.")

    frame = pd.DataFrame(bars)

    required_columns = {
        "openTime",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required_columns.difference(frame.columns)
    if missing:
        raise RuntimeError(
            "BiQuote response is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["openTime"] = pd.to_datetime(
        frame["openTime"],
        utc=True,
        errors="coerce",
    )

    if frame["openTime"].isna().any():
        raise RuntimeError("BiQuote returned invalid candle timestamps.")

    numeric_columns = ("open", "high", "low", "close")

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[list(numeric_columns)].isna().any().any():
        raise RuntimeError("BiQuote returned invalid OHLC values.")

    if "volume" in frame.columns:
        frame["volume"] = pd.to_numeric(
            frame["volume"],
            errors="coerce",
        )

    if "tickVolume" in frame.columns:
        frame["tickVolume"] = pd.to_numeric(
            frame["tickVolume"],
            errors="coerce",
        )

    if "isOpen" in frame.columns:
        frame["isOpen"] = frame["isOpen"].astype(bool)

    frame = (
        frame.sort_values("openTime")
        .drop_duplicates(subset=["openTime"], keep="last")
        .reset_index(drop=True)
    )

    return frame


def fetch_xauusd_quote(
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """Fetch the latest real XAU/USD quote from BiQuote."""

    url = f"{BIQUOTE_BASE_URL}/api/{XAUUSD_SYMBOL}"
    payload = _get_json(url, timeout=timeout)

    if payload.get("symbol") != XAUUSD_SYMBOL:
        raise RuntimeError("BiQuote returned an unexpected quote symbol.")

    if "mid" not in payload:
        raise RuntimeError("BiQuote quote does not contain a mid price.")

    try:
        payload["mid"] = float(payload["mid"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BiQuote returned an invalid mid price.") from exc

    return payload
