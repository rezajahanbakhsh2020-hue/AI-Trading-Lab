from __future__ import annotations

import json

from src.evaluation.final_release import validate_final_release
from src.evaluation.production_end_to_end import (
    load_production_market_data,
    run_production_end_to_end,
)


DATA_PATH = "data/raw/xauusd_daily_2025.csv"
RESULTS_DIR = "results/production"
SYMBOL = "XAUUSD"
INTERVAL = "1d"
# Explicit canonical candidate binding (mirrors render.yaml: 1D mapping).
CANDIDATE_ID = "cand_moving_average_1d_verified_20261020"
STRATEGY_ID = "moving_average"


def main() -> None:
    data = load_production_market_data(DATA_PATH)

    result = run_production_end_to_end(
        data,
        results_dir=RESULTS_DIR,
        symbol=SYMBOL,
        interval=INTERVAL,
        candidate_id=CANDIDATE_ID,
        strategy_id=STRATEGY_ID,
    )

    final_release = validate_final_release(result)

    output = {
        "final_release_ready": final_release[
            "final_release_ready"
        ],
        "status": final_release["status"],
        "strategy": final_release["strategy"],
        "stability_score": final_release[
            "stability_score"
        ],
        "decision": final_release["decision"],
        "checks": final_release["checks"],
    }

    print(
        json.dumps(
            output,
            indent=2,
            default=str,
        )
    )

    if not final_release["final_release_ready"]:
        raise RuntimeError(
            "FINAL RELEASE BLOCKED"
        )

    print("FINAL RELEASE READY")


if __name__ == "__main__":
    main()
