import pandas as pd
import pytest

from src.strategies.baseline import baseline_signal


def test_baseline_signal_creates_moving_averages():
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
    })

    result = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    assert "fast_ma" in result.columns
    assert "slow_ma" in result.columns
    assert "signal" in result.columns


def test_baseline_signal_calculates_moving_averages():
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
    })

    result = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    assert result["fast_ma"].iloc[1] == pytest.approx(1.5)
    assert result["fast_ma"].iloc[2] == pytest.approx(2.5)

    assert result["slow_ma"].iloc[2] == pytest.approx(2.0)
    assert result["slow_ma"].iloc[3] == pytest.approx(3.0)


def test_baseline_signal_generates_signal():
    df = pd.DataFrame({
        "close": [5, 4, 3, 4, 5, 6],
    })

    result = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    assert result["signal"].tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
    ]


def test_baseline_signal_does_not_modify_input():
    df = pd.DataFrame({
        "close": [1, 2, 3, 4, 5],
    })

    original = df.copy()

    baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    pd.testing.assert_frame_equal(df, original)


def test_baseline_signal_requires_close_column():
    df = pd.DataFrame({
        "open": [1, 2, 3],
    })

    with pytest.raises(
        ValueError,
        match="Column 'close' not found",
    ):
        baseline_signal(df)


def test_baseline_signal_requires_fast_window_smaller_than_slow():
    df = pd.DataFrame({
        "close": [1, 2, 3],
    })

    with pytest.raises(
        ValueError,
        match="fast_window must be smaller than slow_window",
    ):
        baseline_signal(
            df,
            fast_window=50,
            slow_window=20,
        )


def test_baseline_signal_accepts_equal_prices():
    df = pd.DataFrame({
        "close": [100, 100, 100, 100, 100],
    })

    result = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    assert result["signal"].tolist() == [0, 0, 0, 0, 0]


def test_baseline_signal_preserves_existing_columns():
    df = pd.DataFrame({
        "timestamp": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        "close": [100, 101, 102],
    })

    result = baseline_signal(
        df,
        fast_window=2,
        slow_window=3,
    )

    assert "timestamp" in result.columns
    assert "close" in result.columns
