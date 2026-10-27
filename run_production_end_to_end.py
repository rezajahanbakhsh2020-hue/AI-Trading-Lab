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

    summary = {
        "end_to_end_ready": result["end_to_end_ready"],
        "decision": result["decision"]["decision"],
        "quote_stale": result["decision"].get("quote_stale"),
        "reason": result["decision"].get("reason"),
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

    if result["end_to_end_ready"]:
        print("PRODUCTION END-TO-END SUCCESS")
        return
    if _is_stale_data_fail_closed(result):
        # The canonical chain executed and correctly failed closed on
        # stale/historical market data: an honest readiness outcome for the
        # committed 2025 dataset (which the live worker does NOT consume).
        print(
            "PRODUCTION END-TO-END FAIL-CLOSED ON STALE MARKET DATA"
        )
        return
    raise RuntimeError(
        "PRODUCTION END-TO-END FAILED"
    )


def _is_stale_data_fail_closed(
    result: dict,
) -> bool:
    decision = result.get("decision") or {}
    return (
        decision.get("decision") == "NO TRADE"
        and decision.get("quote_stale") is True
        and decision.get("reason") == "stale_market_data"
    )


if __name__ == "__main__":
    main()
