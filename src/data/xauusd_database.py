from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


DEFAULT_DB_PATH = Path("data/xauusd.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL,
    source TEXT,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_market_data_lookup
ON market_data(symbol, timeframe, timestamp);
"""


def create_database(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Create the XAU/USD SQLite database and market-data schema."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA)

    return path


def _parse_number(value: str) -> Optional[float]:
    value = value.strip()

    if not value:
        return None

    return float(value)


def _parse_row(line: str) -> tuple[str, float, float, float, float, Optional[float]]:
    parts = [part.strip() for part in line.rstrip("\r\n").split(";")]

    if len(parts) != 6:
        raise ValueError(
            "Each XAU/USD 15m row must contain exactly 6 semicolon-separated fields."
        )

    timestamp, open_price, high_price, low_price, close_price, volume = parts

    if not timestamp:
        raise ValueError("Timestamp cannot be empty.")

    values = (
        timestamp,
        _parse_number(open_price),
        _parse_number(high_price),
        _parse_number(low_price),
        _parse_number(close_price),
        _parse_number(volume),
    )

    if any(value is None for value in values[1:5]):
        raise ValueError("OHLC values cannot be empty.")

    return values  # type: ignore[return-value]


def import_xauusd_15m(
    csv_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
) -> int:
    """
    Import XAU/USD 15-minute candles into SQLite.

    The source CSV is read only. It is never modified.

    Duplicate candles are ignored using:
        symbol + timeframe + timestamp

    Returns the number of newly inserted rows.
    """
    source_path = Path(csv_path)

    if not source_path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    database_path = create_database(db_path)
    inserted = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        header = file.readline().strip()

        expected_header = "Date;Open;High;Low;Close;Volume"

        if header != expected_header:
            raise ValueError(
                f"Unexpected XAU/USD CSV header: {header!r}. "
                f"Expected: {expected_header!r}"
            )

        with sqlite3.connect(database_path) as connection:
            connection.execute("BEGIN")

            for line_number, line in enumerate(file, start=2):
                if not line.strip():
                    continue

                (
                    timestamp,
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume,
                ) = _parse_row(line)

                if high_price < max(open_price, close_price):
                    raise ValueError(
                        f"Invalid OHLC data at line {line_number}: high is too low."
                    )

                if low_price > min(open_price, close_price):
                    raise ValueError(
                        f"Invalid OHLC data at line {line_number}: low is too high."
                    )

                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO market_data (
                        symbol,
                        timeframe,
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        timeframe,
                        timestamp,
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume,
                        source_path.name,
                    ),
                )

                inserted += cursor.rowcount

            connection.commit()

    return inserted


def count_candles(
    db_path: str | Path = DEFAULT_DB_PATH,
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
) -> int:
    """Return the number of stored candles for a symbol/timeframe."""
    database_path = Path(db_path)

    if not database_path.exists():
        return 0

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*)
            FROM market_data
            WHERE symbol = ? AND timeframe = ?
            """,
            (symbol, timeframe),
        ).fetchone()

    return int(row[0])


def get_latest_timestamp(
    db_path: str | Path = DEFAULT_DB_PATH,
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
) -> Optional[str]:
    """Return the latest stored candle timestamp."""
    database_path = Path(db_path)

    if not database_path.exists():
        return None

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT MAX(timestamp)
            FROM market_data
            WHERE symbol = ? AND timeframe = ?
            """,
            (symbol, timeframe),
        ).fetchone()

    return row[0] if row else None


def get_earliest_timestamp(
    db_path: str | Path = DEFAULT_DB_PATH,
    symbol: str = "XAUUSD",
    timeframe: str = "15m",
) -> Optional[str]:
    """Return the earliest stored candle timestamp."""
    database_path = Path(db_path)

    if not database_path.exists():
        return None

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT MIN(timestamp)
            FROM market_data
            WHERE symbol = ? AND timeframe = ?
            """,
            (symbol, timeframe),
        ).fetchone()

    return row[0] if row else None


__all__ = [
    "DEFAULT_DB_PATH",
    "SCHEMA",
    "create_database",
    "import_xauusd_15m",
    "count_candles",
    "get_latest_timestamp",
    "get_earliest_timestamp",
]
