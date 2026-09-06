import pandas as pd


def classify_volatility_regime(
    df: pd.DataFrame,
    return_column: str = "return",
    window: int = 20,
    low_threshold: float = 0.75,
    high_threshold: float = 1.25,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if return_column not in df.columns:
        raise ValueError(
            f"Missing required column: {return_column}"
        )

    if window <= 0:
        raise ValueError("window must be positive.")

    if low_threshold <= 0:
        raise ValueError("low_threshold must be positive.")

    if high_threshold <= low_threshold:
        raise ValueError(
            "high_threshold must be greater than low_threshold."
        )

    result = df.copy()

    volatility = (
        result[return_column]
        .rolling(window=window, min_periods=window)
        .std()
    )

    volatility_median = (
        volatility
        .rolling(window=window, min_periods=window)
        .median()
    )

    volatility_ratio = (
        volatility / volatility_median
    )

    regime = pd.Series(
        "normal_volatility",
        index=result.index,
        dtype="object",
    )

    regime.loc[
        volatility_ratio < low_threshold
    ] = "low_volatility"

    regime.loc[
        volatility_ratio > high_threshold
    ] = "high_volatility"

    regime.loc[
        volatility_ratio.isna()
    ] = pd.NA

    result["volatility"] = volatility
    result["volatility_median"] = volatility_median
    result["volatility_ratio"] = volatility_ratio
    result["volatility_regime"] = regime

    return result
