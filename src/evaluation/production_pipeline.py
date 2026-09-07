from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.evaluation.production_backtest import (
    run_production_backtest,
)
from src.evaluation.production_report import (
    build_production_report,
)
from src.evaluation.production_summary import (
    build_production_summary,
)

DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)

DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)

DEFAULT_PRODUCTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "production"
)


def run_production_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    production_dir: str | Path = DEFAULT_PRODUCTION_DIR,
    save_result: bool = True,
) -> dict:
    result = run_production_backtest(
        data_path=data_path,
        results_dir=results_dir,
        save_result=save_result,
    )

    report = build_production_report(
        result["backtest"]
    )

    summary = build_production_summary(
        production_dir
    )

    return {
        "strategy": result["strategy"],
        "stability_score": result[
            "stability_score"
        ],
        "stability_report": result[
            "stability_report"
        ],
        "backtest": result["backtest"],
        "report": report,
        "summary": summary,
        "saved_result": result.get(
            "saved_result"
        ),
    }
