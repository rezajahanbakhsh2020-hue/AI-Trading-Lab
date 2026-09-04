import pandas as pd
import pytest

from src.strategy.baseline import generate_baseline_signal


def test_generate_baseline_signal():
    df = pd.DataFrame({
        "return": [0.01, -0.02, 0.03, 0.00],
    })

    result = generate_baseline_signal(df)

    assert "signal" in result.columns
    assert result["signal"].tolist() == [0, 1, 0, 1]


def test_generate_baseline_signal_does_not_modify_input():
    df = pd.DataFrame({
        "return": [0.01, -0.02, 0.03],
    })

    original = df.copy()

    generate_baseline_signal(df)

    pd.testing.assert_frame_equal(df, original)


def test_generate_baseline_signal_missing_return():
    df = pd.DataFrame({
        "close": [100.0, 101.0],
    })

    with pytest.raises(ValueError):
        generate_baseline_signal(df)


def test_generate_baseline_signal_custom_column():
    df = pd.DataFrame({
        "returns": [0.01, -0.02, 0.03],
    })

    result = generate_baseline_signal(
        df,
        return_column="returns",
    )

    assert result["signal"].tolist() == [0, 1, 0]
