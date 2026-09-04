import pandas as pd
import pytest

from src.strategies.momentum import momentum_signal


def test_momentum_signal_creates_columns():
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
    })

    result = momentum_signal(
        df,
        window=2,
    )

    assert "momentum" in result.columns
    assert "signal" in result.columns


def test_momentum_signal_generates_positive_signal():
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104],
    })

    result = momentum_signal(
        df,
        window=2,
    )

    assert result["signal"].tolist() == [
        0,
        0,
        1,
        1,
        1,
    ]


def test_momentum_signal_generates_flat_signal_when_momentum_is_negative():
    df = pd.DataFrame({
        "close": [100, 99, 98, 97, 96],
    })

    result = momentum_signal(
        df,
        window=2,
    )

    assert result["signal"].tolist() == [
        0,
        0,
        0,
        0,
        0,
    ]


def test_momentum_signal_zero_momentum_is_flat():
    df = pd.DataFrame({
        "close": [100, 100, 100, 100],
    })

    result = momentum_signal(
        df,
        window=1,
    )

    assert result["signal"].tolist() == [
        0,
        0,
        0,
        0,
    ]


def test_momentum_signal_does_not_modify_input():
    df = pd.DataFrame({
        "close": [100, 101, 102, 103],
    })

    original = df.copy()

    momentum_signal(
        df,
        window=2,
    )

    pd.testing.assert_frame_equal(df, original)


def test_momentum_signal_requires_close_column():
    df = pd.DataFrame({
        "open": [100, 101, 102],
    })

    with pytest.raises(
        ValueError,
        match="Column 'close' not found",
    ):
        momentum_signal(df)


def test_momentum_signal_requires_positive_window():
    df = pd.DataFrame({
        "close": [100, 101, 102],
    })

    with pytest.raises(
        ValueError,
        match="window must be positive",
    ):
        momentum_signal(
            df,
            window=0,
        )


def test_momentum_signal_requires_integer_window():
    df = pd.DataFrame({
        "close": [100, 101, 102],
    })

    with pytest.raises(
        TypeError,
        match="window must be an integer",
    ):
        momentum_signal(
            df,
            window=2.5,
        )
