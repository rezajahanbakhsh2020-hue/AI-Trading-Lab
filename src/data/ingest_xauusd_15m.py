"""
Ingest and validate historical XAUUSD 15-minute market data for Project 1 research.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import urllib.parse
import urllib.request

import pandas as pd

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/getdata-finance/"
    "xauusd-15m-ohlcv-metals-historical-data/main/XAUUSD_15m.csv"
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "xauusd_15m"
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "ds_moving_average_15m.csv"


def download_and_validate_xauusd_15m(
    source_url: str = DEFAULT_SOURCE_URL,
    output_path: str | Path = DEFAULT_OUTPUT_FILE,
) -> tuple[Path, dict]:
    """Download, validate, and persist canonical XAUUSD 15m historical dataset.

    Returns output Path and dataset metadata summary dict.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Download raw CSV data
    req = urllib.request.Request(
        source_url,
        headers={"User-Agent": "AI-Trading-Lab/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_bytes = resp.read()

    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Parse and validate DataFrame
    df = pd.read_csv(io.BytesIO(raw_bytes))

    required_raw_cols = {"datetime", "open", "high", "low", "close"}
    missing = required_raw_cols - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV missing required columns: {sorted(missing)}")

    # Timeframe and Timestamp Validation
    parsed_ts = pd.to_datetime(df["datetime"], utc=True)
    if parsed_ts.isna().any():
        raise ValueError("Source CSV contains invalid or unparseable timestamps.")

    if not parsed_ts.is_monotonic_increasing:
        raise ValueError("Source CSV timestamps are not monotonic increasing.")

    if not parsed_ts.is_unique:
        raise ValueError("Source CSV contains duplicate timestamps.")

    # Validate OHLC geometry
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Source CSV contains non-numeric or NaN OHLC values.")

    invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"])
    invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"])

    if invalid_high.any() or invalid_low.any():
        raise ValueError("Source CSV contains invalid OHLC candle geometry.")

    # Prepare canonical schema
    canonical_df = pd.DataFrame(
        {
            "timestamp": df["datetime"],
            "open": df["open"],
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
        }
    )

    if "volume" in df.columns:
        canonical_df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    canonical_df.to_csv(out_p, index=False)

    metadata = {
        "source_url": source_url,
        "output_path": str(out_p),
        "sha256": sha256_hash,
        "row_count": len(canonical_df),
        "start_timestamp_utc": parsed_ts.min().isoformat(),
        "end_timestamp_utc": parsed_ts.max().isoformat(),
    }

    return out_p, metadata


if __name__ == "__main__":
    out_path, meta = download_and_validate_xauusd_15m()
    print("Downloaded and validated XAUUSD 15m dataset successfully:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
