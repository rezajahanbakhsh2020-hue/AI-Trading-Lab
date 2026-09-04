from datetime import date

import pandas as pd
import pytest

from src.data.xauusd_provider import (
    create_xauusd_csv_provider,
    filter_xauusd_date_range,
)


def create_sample_csv(tmp_path):
    path = tmp_path / "xauusd.csv"

    df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "open": [
                2650.0,
                2660.0,
                2670.0,
                2680.0,
            ],
            "high": [
                2660.0,
                2670.0,
                2680.0,
                2690.0,
            ],
            "low": [
                2640.0,
                2650.0,
                2660.0,
                2670.0,
            ],
            "close": [
                2655.0,
                2665.0,
                2675.0,
                2685.0,
            ],
        }
    )

    df.to_csv(path, index=False)

    return path


def test_xauusd_csv_provider_loads_data(tmp_path):
    path = create_sample_csv(tmp_path)

    provider = create_xauusd_csv_provider(str(path))
    result = provider()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4
    assert list(result.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    ]


def test_xauusd_csv_provider_rejects_missing_columns(tmp_path):
    path = tmp_path / "xauusd.csv"

    pd.DataFrame(
        {
            "timestamp": ["2026-01-01"],
            "open": [2650.0],
            "close": [2660.0],
        }
    ).to_csv(path, index=False)

    provider = create_xauusd_csv_provider(str(path))

    with pytest.raises(
        ValueError,
        match="Missing required columns:",
    ):
        provider()


def test_filter_xauusd_date_range_start_only(tmp_path):
    path = create_sample_csv(tmp_path)

    provider = create_xauusd_csv_provider(str(path))
    df = provider()

    result = filter_xauusd_date_range(
        df,
        start_date=date(2026, 1, 3),
    )

    assert len(result) == 2
    assert result["timestamp"].dt.date.tolist() == [
        date(2026, 1, 3),
        date(2026, 1, 4),
    ]


def test_filter_xauusd_date_range_end_only(tmp_path):
    path = create_sample_csv(tmp_path)

    provider = create_xauusd_csv_provider(str(path))
    df = provider()

    result = filter_xauusd_date_range(
        df,
        end_date=date(2026, 1, 2),
    )

    assert len(result) == 2
    assert result["timestamp"].dt.date.tolist() == [
        date(2026, 1, 1),
        date(2026, 1, 2),
    ]


def test_filter_xauusd_date_range_is_inclusive(tmp_path):
    path = create_sample_csv(tmp_path)

    provider = create_xauusd_csv_provider(str(path))
    df = provider()

    result = filter_xauusd_date_range(
        df,
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    assert len(result) == 2
    assert result["timestamp"].dt.date.tolist() == [
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]


def test_filter_xauusd_date_range_does_not_modify_input(tmp_path):
    path = create_sample_csv(tmp_path)

    provider = create_xauusd_csv_provider(str(path))
    df = provider()
    original = df.copy()

    filter_xauusd_date_range(
        df,
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    pd.testing.assert_frame_equal(df, original)
