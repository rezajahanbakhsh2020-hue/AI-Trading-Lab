from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DEFAULT_PRODUCTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "production"
)


def load_production_result(
    run_dir: str | Path,
) -> dict:
    run_path = Path(run_dir)

    backtest_path = run_path / "backtest.csv"
    metadata_path = run_path / "metadata.json"

    if not backtest_path.exists():
        raise FileNotFoundError(
            f"Production backtest not found: "
            f"{backtest_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Production metadata not found: "
            f"{metadata_path}"
        )

    backtest = pd.read_csv(backtest_path)

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    return {
        "run_dir": str(run_path),
        "backtest": backtest,
        "metadata": metadata,
    }


def find_production_results(
    results_dir: str | Path = DEFAULT_PRODUCTION_DIR,
) -> list[Path]:
    root = Path(results_dir)

    if not root.exists():
        return []

    results: list[Path] = []

    for path in root.iterdir():
        if not path.is_dir():
            continue

        if (
            (path / "backtest.csv").exists()
            and (path / "metadata.json").exists()
        ):
            results.append(path)

    return sorted(
        results,
        reverse=True,
    )


def compare_production_results(
    results_dir: str | Path = DEFAULT_PRODUCTION_DIR,
) -> pd.DataFrame:
    results = find_production_results(
        results_dir
    )

    columns = [
        "run_id",
        "created_at_utc",
        "strategy",
        "stability_score",
        "observations",
    ]

    rows: list[dict] = []

    for run_dir in results:
        result = load_production_result(
            run_dir
        )

        metadata = result["metadata"]

        rows.append(
            {
                "run_id": run_dir.name,
                "created_at_utc": metadata.get(
                    "created_at_utc"
                ),
                "strategy": metadata.get(
                    "strategy"
                ),
                "stability_score": metadata.get(
                    "stability_score"
                ),
                "observations": metadata.get(
                    "observations"
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "created_at_utc",
            ascending=False,
        )
        .reset_index(drop=True)
    )
