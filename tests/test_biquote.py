from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.biquote import (
    DEFAULT_INTERVAL,
    DEFAULT_LIMIT,
    fetch_xauusd_ohlc,
    fetch_xauusd_quote,
)
from src.data.provider import BiQuoteProvider, UnsupportedInstrumentError


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


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


def test_biquote_provider_default_fetchers_use_headless_biquote() -> None:
    """Verify BiQuoteProvider without explicit fetchers uses headless src.data.biquote."""
    payload_ohlc = make_ohlc_payload()
    payload_quote = make_quote_payload()

    provider = BiQuoteProvider()
    assert provider.supports_symbol("XAUUSD")

    with patch("src.data.biquote.urlopen", side_effect=[FakeResponse(payload_quote), FakeResponse(payload_ohlc)]):
        quote = provider.get_quote("XAUUSD")
        assert quote["symbol"] == "XAUUSD"
        assert quote["mid"] == 3516.25

        df = provider.get_candles("XAUUSD", timeframe="5m", limit=2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2


def test_biquote_provider_controlled_fetcher() -> None:
    """Verify BiQuoteProvider works with controlled custom fetchers."""
    dummy_df = pd.DataFrame({
        "openTime": pd.date_range("2026-01-01", periods=2, tz="UTC"),
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [99.0, 100.0],
        "close": [102.0, 103.0],
    })

    provider = BiQuoteProvider(
        ohlc_fetcher=lambda interval, limit: dummy_df,
        quote_fetcher=lambda: {"symbol": "XAUUSD", "mid": 2000.0},
    )

    df = provider.get_candles("XAUUSD")
    assert len(df) == 2

    quote = provider.get_quote("XAUUSD")
    assert quote["mid"] == 2000.0


def test_biquote_validation_contract() -> None:
    """Verify headless BiQuote validation rules fail closed properly."""
    # Unsupported interval
    with pytest.raises(ValueError, match="Unsupported interval"):
        fetch_xauusd_ohlc(interval="2h")

    # Invalid limit bounds
    with pytest.raises(ValueError, match="between"):
        fetch_xauusd_ohlc(limit=0)

    # Invalid non-integer limit
    with pytest.raises(ValueError, match="integer"):
        fetch_xauusd_ohlc(limit=10.5)  # type: ignore

    # Unexpected symbol returned from endpoint
    with patch("src.data.biquote.urlopen", return_value=FakeResponse(make_ohlc_payload(symbol="EURUSD"))):
        with pytest.raises(RuntimeError, match="unexpected symbol"):
            fetch_xauusd_ohlc()

    # Empty bars
    with patch("src.data.biquote.urlopen", return_value=FakeResponse({"symbol": "XAUUSD", "interval": "5m", "bars": []})):
        with pytest.raises(RuntimeError, match="no XAU/USD candles"):
            fetch_xauusd_ohlc()


def test_biquote_symbol_mismatch_fails_closed() -> None:
    """Verify requesting an unsupported symbol on BiQuoteProvider raises UnsupportedInstrumentError."""
    provider = BiQuoteProvider(supported_symbols=("XAUUSD",))
    with pytest.raises(UnsupportedInstrumentError, match="No market-data provider supports EURUSD"):
        provider.get_candles("EURUSD")

    with pytest.raises(UnsupportedInstrumentError, match="No market-data provider supports EURUSD"):
        provider.get_quote("EURUSD")


def test_subprocess_headless_provider_and_runtime_exercise_no_ui_modules() -> None:
    """Subprocess test exercising headless BiQuoteProvider and LiveExecutionRuntime to ensure no UI modules are loaded."""
    code = """
import sys
from unittest.mock import patch
import json
import pandas as pd

from src.data.provider import BiQuoteProvider, resolve_provider_for_symbol
from src.evaluation.live_execution_runtime import LiveExecutionRuntime, load_live_market_data

# 1. Assert modules not in sys.modules upon import
assert 'plotly' not in sys.modules, f'plotly loaded prematurely'
assert 'streamlit' not in sys.modules, f'streamlit loaded prematurely'
assert 'app_live' not in sys.modules, f'app_live loaded prematurely'

# 2. Exercise provider path
sample_payload = {
    'symbol': 'XAUUSD',
    'interval': '5m',
    'bars': [
        {'openTime': '2026-09-09T07:00:00Z', 'open': 3500.0, 'high': 3510.0, 'low': 3490.0, 'close': 3505.0},
        {'openTime': '2026-09-09T08:00:00Z', 'open': 3505.0, 'high': 3520.0, 'low': 3500.0, 'close': 3515.0}
    ]
}

class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def read(self):
        return json.dumps(self.payload).encode('utf-8')
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

with patch('src.data.biquote.urlopen', return_value=FakeResponse(sample_payload)):
    df = load_live_market_data('XAUUSD', '5m', 2)
    assert not df.empty, 'Market data empty'

# 3. Assert modules STILL not in sys.modules after exercising provider path
assert 'plotly' not in sys.modules, 'plotly loaded during execution'
assert 'streamlit' not in sys.modules, 'streamlit loaded during execution'
assert 'app_live' not in sys.modules, 'app_live loaded during execution'

print('SUCCESS')
"""

    cmd = [sys.executable, "-c", code]
    res = subprocess.run(cmd, capture_output=True, text=True, env={"PYTHONPATH": "."})
    assert res.returncode == 0, f"Subprocess headless test failed: {res.stderr}\n{res.stdout}"
    assert "SUCCESS" in res.stdout
