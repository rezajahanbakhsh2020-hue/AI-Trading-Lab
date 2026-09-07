from pathlib import Path

import run_production


def test_production_entrypoint_exists() -> None:
    assert callable(run_production.main)


def test_production_entrypoint_uses_project_paths() -> None:
    root = Path(run_production.__file__).resolve().parent

    assert (
        root / "data" / "raw" / "xauusd_daily_2025.csv"
    ).parent.exists()

    assert (
        root / "results" / "walk_forward"
    ).name == "walk_forward"

    assert (
        root / "results" / "production"
    ).name == "production"
