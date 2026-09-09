from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch
from urllib.error import URLError

import pandas as pd
import pytest

import app_live


SAMPLE_BARS = [
    {
        "openTime": "2026-09-09T07:00:00Z",
        "open": 3500.0,
        "high": 3510.0,
        "low": 3490.0,
        "close": 3505.0,
        "volume": 0,
        "tickVolume": 100,
        "isOpen": False,
    },
    {
        "openTime": "2026-09-09T08:00:00Z",
        "open": 3505.0,
        "high": 3520.0,
        "low": 3500.0,
        "close": 3515.0,
        "volume": 0,
        "tickVolume": 120,
        "isOpen": True,
    },
]


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def make_ohlc_payload(
    *,
    symbol: str = "XAUUSD",
    interval: str = "5m",
    bars: list[dict] | None = None,
) -> dict:
    return {
        "symbol": symbol,
        "interval": interval,
        "bars": SAMPLE_BARS if bars is None else bars,
    }


def make_quote_payload(
    *,
    symbol: str = "XAUUSD",
    mid: float = 3516.25,
) -> dict:
    return {
        "symbol": symbol,
        "description": "Gold vs US Dollar",
        "bid": mid - 0.05,
        "ask": mid + 0.05,
        "mid": mid,
        "spread": 0.10,
        "last": 0.0,
        "volume": 0,
        "marketState": "open",
        "stale": False,
        "quoteAgeSeconds": 0,
    }


def test_fetch_xauusd_ohlc_returns_valid_dataframe():
    payload = make_ohlc_payload()

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        data = app_live.fetch_xauusd_ohlc(
            interval="5m",
            limit=2,
        )

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert {
        "openTime",
        "open",
        "high",
        "low",
        "close",
    }.issubset(data.columns)

    assert data["openTime"].is_monotonic_increasing
    assert float(data.iloc[-1]["close"]) == 3515.0


def test_fetch_xauusd_ohlc_rejects_invalid_interval():
    with pytest.raises(ValueError, match="Unsupported interval"):
        app_live.fetch_xauusd_ohlc(interval="2h")


def test_fetch_xauusd_ohlc_rejects_invalid_limit():
    with pytest.raises(ValueError, match="between"):
        app_live.fetch_xauusd_ohlc(limit=0)

    with pytest.raises(ValueError, match="between"):
        app_live.fetch_xauusd_ohlc(limit=1001)


def test_fetch_xauusd_ohlc_rejects_non_integer_limit():
    with pytest.raises(ValueError, match="integer"):
        app_live.fetch_xauusd_ohlc(limit=10.5)


def test_fetch_xauusd_ohlc_rejects_wrong_symbol():
    payload = make_ohlc_payload(symbol="EURUSD")

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="unexpected symbol",
        ):
            app_live.fetch_xauusd_ohlc(
                interval="5m",
                limit=2,
            )


def test_fetch_xauusd_ohlc_rejects_missing_bars():
    payload = {
        "symbol": "XAUUSD",
        "interval": "5m",
        "bars": [],
    }

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="no XAU/USD candles",
        ):
            app_live.fetch_xauusd_ohlc(
                interval="5m",
                limit=2,
            )


def test_fetch_xauusd_ohlc_rejects_missing_required_column():
    bars = [
        {
            "openTime": "2026-09-09T07:00:00Z",
            "open": 3500.0,
            "high": 3510.0,
            "low": 3490.0,
            "isOpen": False,
        }
    ]

    payload = make_ohlc_payload(bars=bars)

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="missing required columns",
        ):
            app_live.fetch_xauusd_ohlc(
                interval="5m",
                limit=1,
            )


def test_fetch_xauusd_ohlc_rejects_invalid_timestamp():
    bars = [
        {
            "openTime": "not-a-date",
            "open": 3500.0,
            "high": 3510.0,
            "low": 3490.0,
            "close": 3505.0,
        }
    ]

    payload = make_ohlc_payload(bars=bars)

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="invalid candle timestamps",
        ):
            app_live.fetch_xauusd_ohlc(
                interval="5m",
                limit=1,
            )


def test_fetch_xauusd_ohlc_rejects_invalid_ohlc_values():
    bars = [
        {
            "openTime": "2026-09-09T07:00:00Z",
            "open": "bad",
            "high": 3510.0,
            "low": 3490.0,
            "close": 3505.0,
        }
    ]

    payload = make_ohlc_payload(bars=bars)

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="invalid OHLC values",
        ):
            app_live.fetch_xauusd_ohlc(
                interval="5m",
                limit=1,
            )


def test_fetch_xauusd_ohlc_handles_unsorted_bars():
    bars = list(reversed(SAMPLE_BARS))
    payload = make_ohlc_payload(bars=bars)

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        data = app_live.fetch_xauusd_ohlc(
            interval="5m",
            limit=2,
        )

    assert data["openTime"].is_monotonic_increasing
    assert float(data.iloc[0]["open"]) == 3500.0
    assert float(data.iloc[1]["open"]) == 3505.0


def test_fetch_xauusd_ohlc_deduplicates_timestamps():
    bars = [
        SAMPLE_BARS[0],
        SAMPLE_BARS[0],
        SAMPLE_BARS[1],
    ]

    payload = make_ohlc_payload(bars=bars)

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        data = app_live.fetch_xauusd_ohlc(
            interval="5m",
            limit=3,
        )

    assert len(data) == 2
    assert data["openTime"].is_unique


def test_fetch_xauusd_quote_returns_mid_price():
    payload = make_quote_payload(mid=3516.25)

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        quote = app_live.fetch_xauusd_quote()

    assert quote["symbol"] == "XAUUSD"
    assert quote["mid"] == 3516.25


def test_fetch_xauusd_quote_rejects_wrong_symbol():
    payload = make_quote_payload(symbol="EURUSD")

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="unexpected quote symbol",
        ):
            app_live.fetch_xauusd_quote()


def test_fetch_xauusd_quote_rejects_missing_mid():
    payload = make_quote_payload()
    payload.pop("mid")

    with patch(
        "app_live.urlopen",
        return_value=FakeResponse(payload),
    ):
        with pytest.raises(
            RuntimeError,
            match="mid price",
        ):
            app_live.fetch_xauusd_quote()


def test_fetch_xauusd_quote_rejects_network_error():
    with patch(
        "app_live.urlopen",
        side_effect=URLError("network unavailable"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Unable to reach BiQuote",
        ):
            app_live.fetch_xauusd_quote()


def test_build_live_chart_creates_candlestick():
    data = pd.DataFrame(SAMPLE_BARS)
    data["openTime"] = pd.to_datetime(
        data["openTime"],
        utc=True,
    )

    figure = app_live.build_live_chart(data)

    assert len(figure.data) == 1
    assert figure.data[0].type == "candlestick"
    assert len(figure.data[0].x) == 2


def test_build_live_chart_rejects_missing_column():
    data = pd.DataFrame(
        {
            "openTime": pd.to_datetime(
                ["2026-09-09T07:00:00Z"],
                utc=True,
            ),
            "open": [3500.0],
            "high": [3510.0],
            "low": [3490.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        app_live.build_live_chart(data)


def test_price_change_returns_expected_values():
    data = pd.DataFrame(
        {
            "close": [3500.0, 3515.0],
        }
    )

    change, change_percent = app_live._price_change(data)

    assert change == 15.0
    assert change_percent == pytest.approx(
        (15.0 / 3500.0) * 100.0
    )


def test_price_change_handles_single_row():
    data = pd.DataFrame(
        {
            "close": [3500.0],
        }
    )

    change, change_percent = app_live._price_change(data)

    assert change == 0.0
    assert change_percent == 0.0
