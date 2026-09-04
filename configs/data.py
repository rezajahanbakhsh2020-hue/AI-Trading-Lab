"""
Market data configuration for AI-Trading-Lab.
"""

XAUUSD_CONFIG = {
    "symbol": "XAU/USD",
    "raw_path": "data/raw/xauusd.csv",
    "processed_path": "data/processed/xauusd.csv",
    "features_path": "data/features/xauusd.csv",
    "timestamp_column": "timestamp",
    "price_columns": [
        "open",
        "high",
        "low",
        "close",
    ],
}
