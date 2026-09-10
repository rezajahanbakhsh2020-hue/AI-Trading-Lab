from __future__ import annotations

from pathlib import Path
from typing import Any
import json


DEFAULT_PRODUCTION_RESULTS_DIR = Path("results/production")


def load_latest_production_result(
    results_dir: str | Path = DEFAULT_PRODUCTION_RESULTS_DIR,
) -> dict[str, Any]:
    """Load the newest saved production result."""

    directory = Path(results_dir)

    if not directory.exists():
        raise FileNotFoundError(
            f"Production results directory not found: {directory}"
        )

    candidates = sorted(
        directory.rglob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No production JSON result found in {directory}"
        )

    for path in candidates:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if isinstance(payload, dict):
            return {
                "path": str(path),
                "result": payload,
            }

    raise ValueError(
        "No valid production result dictionary was found."
    )


def extract_production_selection(
    production_result: dict[str, Any],
) -> dict[str, Any]:
    """Extract stable strategy and stability score from production output."""

    if not isinstance(production_result, dict):
        raise TypeError(
            "production_result must be a dictionary."
        )

    payload = production_result.get("result", production_result)

    if not isinstance(payload, dict):
        raise ValueError(
            "Production result payload must be a dictionary."
        )

    stable_strategy = payload.get("stable_strategy")
    stability_score = payload.get("stability_score")

    if stable_strategy is None:
        stable_strategy = payload.get("strategy")

    if stability_score is None:
        stability_score = payload.get("stability", {}).get(
            "stability_score"
        ) if isinstance(payload.get("stability"), dict) else None

    if stable_strategy is None:
        raise ValueError(
            "Production result does not contain a stable strategy."
        )

    if stability_score is None:
        raise ValueError(
            "Production result does not contain a stability score."
        )

    return {
        "stable_strategy": str(stable_strategy),
        "stability_score": float(stability_score),
        "source_path": production_result.get("path"),
    }


def load_production_selection(
    results_dir: str | Path = DEFAULT_PRODUCTION_RESULTS_DIR,
) -> dict[str, Any]:
    """Load and normalize the latest production selection."""

    latest = load_latest_production_result(results_dir)

    return extract_production_selection(latest)
