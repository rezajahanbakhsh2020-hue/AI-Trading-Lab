from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


DEFAULT_PRODUCTION_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "production"
)


def save_production_result(
    result: dict,
    output_dir: str | Path = DEFAULT_PRODUCTION_DIR,
) -> dict[str, str]:
    required_keys = {
        "strategy",
        "stability_score",
        "backtest",
    }

    missing = required_keys - set(result)

    if missing:
        raise ValueError(
            "result is missing required keys: "
            f"{sorted(missing)}"
        )

    backtest = result["backtest"]

    if not isinstance(backtest, pd.DataFrame):
        raise TypeError(
            "backtest must be a pandas DataFrame."
        )

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = (
        output_path
        / f"{timestamp}_{uuid4().hex[:8]}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    backtest_path = (
        run_dir / "backtest.csv"
    )

    metadata_path = (
        run_dir / "metadata.json"
    )

    backtest.to_csv(
        backtest_path,
        index=False,
    )

    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "strategy": result["strategy"],
        "stability_score": float(
            result["stability_score"]
        ),
        "observations": int(
            len(backtest)
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
        "backtest": str(backtest_path),
        "metadata": str(metadata_path),
    }
