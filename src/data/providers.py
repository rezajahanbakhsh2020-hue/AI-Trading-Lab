from pathlib import Path

import pandas as pd


def csv_provider(
    path: str | Path,
):
    """
    Create a data provider that loads market data from a CSV file.
    """

    csv_path = Path(path)

    def provider() -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"File not found: {csv_path}"
            )

        return pd.read_csv(csv_path)

    return provider
