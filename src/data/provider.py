from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, runtime_checkable

import pandas as pd


DataProvider = Callable[[], pd.DataFrame]


class UnsupportedInstrumentError(ValueError):
    """Raised when an instrument symbol is unsupported by market data providers."""

    pass


@runtime_checkable
class MarketDataProvider(Protocol):
    """
    Explicit capability-based market data provider contract.
    """

    def supports_symbol(self, symbol: str) -> bool:
        ...

    def get_quote(self, symbol: str) -> dict[str, Any]:
        ...

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame:
        ...


def resolve_requested_symbol(request: Any) -> str:
    """
    Resolve and normalize requested instrument symbol without default fallback substitution.
    Preserves canonical symbol identity throughout the execution path.
    """
    if isinstance(request, dict):
        symbol = request.get("symbol")
    elif hasattr(request, "symbol"):
        symbol = getattr(request, "symbol")
    else:
        symbol = request

    if symbol is None or not str(symbol).strip():
        raise ValueError("Requested symbol must be a non-empty string.")

    return str(symbol).strip().upper()


def resolve_provider_for_symbol(
    symbol: str,
    available_providers: Iterable[MarketDataProvider],
) -> MarketDataProvider | None:
    """
    Select market data provider based strictly on capability truth (candidate.supports_symbol).
    Never selects a provider merely because it exists.
    Never rewrites the requested symbol to satisfy a provider.
    """
    canonical_symbol = resolve_requested_symbol(symbol)
    return next(
        (
            candidate
            for candidate in available_providers
            if candidate.supports_symbol(canonical_symbol)
        ),
        None,
    )


class FunctionMarketDataProvider:
    """
    Adapter converting symbol-specific or generic market data functions into a MarketDataProvider.
    """

    def __init__(
        self,
        symbol: str,
        ohlc_fetcher: Callable[..., pd.DataFrame],
        quote_fetcher: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if not symbol or not str(symbol).strip():
            raise ValueError("symbol must be a non-empty string.")
        self._canonical_symbol = str(symbol).strip().upper()
        self._ohlc_fetcher = ohlc_fetcher
        self._quote_fetcher = quote_fetcher

    @property
    def canonical_symbol(self) -> str:
        return self._canonical_symbol

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or not str(symbol).strip():
            return False
        return str(symbol).strip().upper() == self._canonical_symbol

    def get_quote(self, symbol: str) -> dict[str, Any]:
        requested = resolve_requested_symbol(symbol)
        if not self.supports_symbol(requested):
            raise UnsupportedInstrumentError(
                f"No market-data provider supports {requested}"
            )
        if self._quote_fetcher is not None:
            quote = self._quote_fetcher()
        else:
            quote = {"symbol": requested, "connected": True}

        if quote.get("symbol") and str(quote.get("symbol")).strip().upper() != requested:
            raise UnsupportedInstrumentError(
                f"Provider returned quote for symbol '{quote.get('symbol')}' "
                f"which does not match requested symbol '{requested}'."
            )
        return quote

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame:
        requested = resolve_requested_symbol(symbol)
        if not self.supports_symbol(requested):
            raise UnsupportedInstrumentError(
                f"No market-data provider supports {requested}"
            )

        try:
            df = self._ohlc_fetcher(symbol=requested, timeframe=timeframe, limit=limit)
        except TypeError:
            try:
                df = self._ohlc_fetcher(interval=timeframe, limit=limit)
            except TypeError:
                df = self._ohlc_fetcher()

        return df


class BiQuoteProvider:
    """
    BiQuote market data provider implementing MarketDataProvider.
    """

    def __init__(
        self,
        supported_symbols: Iterable[str] = ("XAUUSD",),
        ohlc_fetcher: Callable[..., pd.DataFrame] | None = None,
        quote_fetcher: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._supported_symbols = {str(s).strip().upper() for s in supported_symbols if s and str(s).strip()}
        self._ohlc_fetcher = ohlc_fetcher
        self._quote_fetcher = quote_fetcher

    def supports_symbol(self, symbol: str) -> bool:
        if not symbol or not str(symbol).strip():
            return False
        return str(symbol).strip().upper() in self._supported_symbols

    def get_quote(self, symbol: str) -> dict[str, Any]:
        requested = resolve_requested_symbol(symbol)
        if not self.supports_symbol(requested):
            raise UnsupportedInstrumentError(
                f"No market-data provider supports {requested}"
            )

        if self._quote_fetcher is None:
            from app_live import fetch_xauusd_quote
            quote_fn = fetch_xauusd_quote
        else:
            quote_fn = self._quote_fetcher

        quote = quote_fn()
        if quote.get("symbol") and str(quote.get("symbol")).strip().upper() != requested:
            raise UnsupportedInstrumentError(
                f"Provider returned quote for symbol '{quote.get('symbol')}' "
                f"which does not match requested symbol '{requested}'."
            )
        return quote

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 200,
    ) -> pd.DataFrame:
        requested = resolve_requested_symbol(symbol)
        if not self.supports_symbol(requested):
            raise UnsupportedInstrumentError(
                f"No market-data provider supports {requested}"
            )

        if self._ohlc_fetcher is None:
            from app_live import fetch_xauusd_ohlc
            ohlc_fn = fetch_xauusd_ohlc
        else:
            ohlc_fn = self._ohlc_fetcher

        return ohlc_fn(interval=timeframe, limit=limit)


def validate_data_provider(
    provider: DataProvider,
) -> None:
    """
    Validate that the supplied object can be used as a data provider.
    """

    if not callable(provider):
        raise TypeError("provider must be callable.")


def load_from_provider(
    provider: DataProvider,
) -> pd.DataFrame:
    """
    Load market data from a callable data provider.
    """

    validate_data_provider(provider)

    df = provider()

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Data provider must return a pandas DataFrame."
        )

    return df.copy()
