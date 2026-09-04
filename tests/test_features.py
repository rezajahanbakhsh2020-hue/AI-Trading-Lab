import pandas as pd
import pytest

from src.features.indicators import add_returns


def test_add_returns():
    df = pd.DataFrame(
        {
            "close": [100.0, 110.0, 121.0],
        }
    )

    result = add_returns(df)

    assert pd.isna(result["return"].iloc[0])
    assert result["return"].iloc[1] == pytest.approx(0.10)
    assert result["return"].iloc[2] == pytest.approx(0.10)
