from __future__ import annotations

from pathlib import Path

from src.evaluation.live_workflow import (
    run_xauusd_walk_forward,
)
from src.evaluation.production_backtest import (
    run_production_backtest,
)
from src.evaluation.production_report import (
    build_production_report,
)
from src.evaluation.production_summary import (
    build_production_summary,
)
from src.evaluation.result_store import (
    DEFAULT_RESULTS_DIR as DEFAULT_WALK_FORWARD_DIR,
)


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "xauusd_daily_2025.csv"
)

DEFAULT_RESULTS_DIR = DEFAULT_WALK_FORWARD_DIR

DEFAULT_PRODUCTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "production"
)


def _has_walk_forward_results(
    results_dir: str | Path,
) -> bool:
    root = Path(results_dir)

    if not root.exists():
        return False

    for path in root.iterdir():
        if not path.is_dir():
            continue

        if (
            (path / "final_report.csv").exists()
            and (path / "metadata.json").exists()
        ):
            return True

    return False


def _prepare_production_selection(
    data_path: str | Path,
    results_dir: str | Path,
) -> None:
    if _has_walk_forward_results(results_dir):
        return

    run_xauusd_walk_forward(
        path=data_path,
        save_result=True,
    )


def run_production_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    production_dir: str | Path = DEFAULT_PRODUCTION_DIR,
    save_result: bool = True,
) -> dict:
    _prepare_production_selection(
        data_path=data_path,
        results_dir=results_dir,
    )

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
