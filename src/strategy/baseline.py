import pandas as pd


def generate_baseline_signal(
    df: pd.DataFrame,
    return_column: str = "return",
) -> pd.DataFrame:
    """
    Generate a simple baseline long-only trading signal.

    Signal:
        1 -> hold the asset
        0 -> stay out of the market

    The signal is based on the previous period's return
    to avoid look-ahead bias.
    """

    result = df.copy()

    if return_column not in result.columns:
        raise ValueError(f"Column '{return_column}' not found.")

    result["signal"] = (
        result[return_column]
        .shift(1)
        .gt(0)
        .astype(int)
    )

    return result
