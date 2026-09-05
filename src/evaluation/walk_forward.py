from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    """Describe one chronological train/test window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_walk_forward_windows(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[WalkForwardWindow]:
    """
    Generate chronological walk-forward train/test windows.

    The test period always occurs after the training period, preventing
    future observations from entering the training window.

    Parameters
    ----------
    df:
        Input time-series DataFrame.

    train_size:
        Number of observations used for training.

    test_size:
        Number of observations used for out-of-sample testing.

    step:
        Number of observations to move forward after each window.
        Defaults to test_size.

    Returns
    -------
    list[WalkForwardWindow]
        Ordered walk-forward windows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if train_size <= 0:
        raise ValueError("train_size must be greater than zero.")

    if test_size <= 0:
        raise ValueError("test_size must be greater than zero.")

    if step is None:
        step = test_size

    if step <= 0:
        raise ValueError("step must be greater than zero.")

    total_size = len(df)

    if total_size < train_size + test_size:
        return []

    windows: list[WalkForwardWindow] = []

    start = 0

    while start + train_size + test_size <= total_size:
        train_start = start
        train_end = start + train_size

        test_start = train_end
        test_end = test_start + test_size

        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

        start += step

    return windows
