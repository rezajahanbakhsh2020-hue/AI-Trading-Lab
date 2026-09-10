from __future__ import annotations

import json

from src.evaluation.production_end_to_end import (
    load_production_market_data,
    run_production_end_to_end,
)


DATA_PATH = "data/raw/xauusd_daily_2025.csv"
RESULTS_DIR = "results/production"
SYMBOL = "XAUUSD"
INTERVAL = "1d"


def main() -> None:
    data = load_production_market_data(DATA_PATH)

    result = run_production_end_to_end(
        data,
        results_dir=RESULTS_DIR,
        symbol=SYMBOL,
        interval=INTERVAL,
    )

    summary = {
        "end_to_end_ready": result["end_to_end_ready"],
        "decision": result["decision"]["decision"],
        "stable_strategy": result["production_selection"][
            "stable_strategy"
        ],
        "stability_score": result["production_selection"][
            "stability_score"
        ],
        "trend": result["decision"]["trend"],
        "entry_price": result["display"]["entry_price"],
        "stop_loss": result["display"]["stop_loss"],
        "tp1": result["display"]["tp1"],
        "tp2": result["display"]["tp2"],
        "tp3": result["display"]["tp3"],
        "release_ready": result["release_gate"][
            "release_ready"
        ],
    }

    print(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    if not result["end_to_end_ready"]:
        raise RuntimeError(
            "PRODUCTION END-TO-END FAILED"
        )

    print("PRODUCTION END-TO-END SUCCESS")


if __name__ == "__main__":
    main()
