from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "walk_forward"
)


def save_walk_forward_result(
    result: dict,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> dict[str, str]:
    """
    Persist one completed walk-forward experiment.

    Saves:
        final_report.csv
        eligible_strategies.csv
        metadata.json
    """
    required_keys = {
        "final_report",
        "eligible_strategies",
        "best_strategy",
    }

    missing = required_keys - set(result)

    if missing:
        raise ValueError(
            "result is missing required keys: "
            f"{sorted(missing)}"
        )

    final_report = result["final_report"]
    eligible = result["eligible_strategies"]

    if not isinstance(
        final_report,
        pd.DataFrame,
    ):
        raise TypeError(
            "result['final_report'] must be "
            "a pandas DataFrame."
        )

    if not isinstance(
        eligible,
        pd.DataFrame,
    ):
        raise TypeError(
            "result['eligible_strategies'] must be "
            "a pandas DataFrame."
        )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    run_dir = output_path / timestamp
    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    final_report_path = (
        run_dir / "final_report.csv"
    )
    eligible_path = (
        run_dir / "eligible_strategies.csv"
    )
    metadata_path = run_dir / "metadata.json"

    final_report.to_csv(
        final_report_path,
        index=False,
    )

    eligible.to_csv(
        eligible_path,
        index=False,
    )

    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "best_strategy": result[
            "best_strategy"
        ],
        "strategies": list(
            result.get("comparison", {}).keys()
        ),
        "rows": int(
            len(result.get("data", []))
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "run_dir": str(run_dir),
        "final_report": str(
            final_report_path
        ),
        "eligible_strategies": str(
            eligible_path
        ),
        "metadata": str(metadata_path),
    }
