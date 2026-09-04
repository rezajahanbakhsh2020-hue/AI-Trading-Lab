from configs.data import XAUUSD_CONFIG


def test_xauusd_config_contains_expected_symbol():
    assert XAUUSD_CONFIG["symbol"] == "XAU/USD"


def test_xauusd_config_contains_data_paths():
    assert XAUUSD_CONFIG["raw_path"] == "data/raw/xauusd.csv"
    assert XAUUSD_CONFIG["processed_path"] == "data/processed/xauusd.csv"
    assert XAUUSD_CONFIG["features_path"] == "data/features/xauusd.csv"


def test_xauusd_config_contains_timestamp_column():
    assert XAUUSD_CONFIG["timestamp_column"] == "timestamp"


def test_xauusd_config_contains_required_price_columns():
    assert XAUUSD_CONFIG["price_columns"] == [
        "open",
        "high",
        "low",
        "close",
    ]


def test_xauusd_config_is_dictionary():
    assert isinstance(XAUUSD_CONFIG, dict)
