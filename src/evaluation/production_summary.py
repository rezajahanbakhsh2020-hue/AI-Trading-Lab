from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.production_comparison import (
    load_production_result,
    find_production_results,
)
from src.evaluation.production_report import (
    build_production_report,
)


DEFAULT_PRODUCTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "production"
)


def build_production_summary(
    results_dir: str | Path = DEFAULT_PRODUCTION_DIR,
) -> pd.DataFrame:
    production_results = find_production_results(
        results_dir
    )

    columns = [
        "run_id",
        "created_at_utc",
        "strategy",
        "stability_score",
        "observations",
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
    ]

    if not production_results:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []

    for run_dir in production_results:
        result = load_production_result(run_dir)

        metadata = result["metadata"]
        report = build_production_report(
            result["backtest"]
        )

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
                "observations": report[
                    "observations"
                ],
                "total_return": report[
                    "total_return"
                ],
                "max_drawdown": report[
                    "max_drawdown"
                ],
                "sharpe_ratio": report[
                    "sharpe_ratio"
                ],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "created_at_utc",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_latest_production_result(
    results_dir: str | Path = DEFAULT_PRODUCTION_DIR,
) -> dict | None:
    production_results = find_production_results(
        results_dir
    )

    if not production_results:
        return None

    return load_production_result(
        production_results[0]
    )
