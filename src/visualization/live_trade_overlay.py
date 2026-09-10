from __future__ import annotations

from numbers import Real
from typing import Any

import pandas as pd


REQUIRED_DISPLAY_FIELDS = (
    "decision",
    "entry_price",
    "stop_loss",
    "tp1",
    "tp2",
    "tp3",
)


def _validate_display(display: dict[str, Any]) -> None:
    if not isinstance(display, dict):
        raise ValueError("display must be a dictionary.")

    missing = [
        field
        for field in REQUIRED_DISPLAY_FIELDS
        if field not in display
    ]

    if missing:
        raise ValueError(
            "display is missing required fields: "
            + ", ".join(missing)
        )


def _validate_price(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number.")

    value = float(value)

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def build_live_trade_overlay(
    data: pd.DataFrame,
    display: dict[str, Any],
) -> dict[str, Any]:
    """
    Build chart-ready trade overlay data.

    The function does not calculate a new signal or alter trade levels.
    It converts the existing live trade display into a normalized
    chart-overlay structure.

    BUY:
        Entry, SL, TP1, TP2 and TP3 are visible.

    NO TRADE:
        No trade price lines are emitted.

    The returned structure is intentionally independent of any
    charting library so it can be consumed by Plotly, Streamlit,
    or another visualization layer.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("data must not be empty.")

    if "timestamp" not in data.columns:
        raise ValueError("data must contain timestamp.")

    _validate_display(display)

    decision = display["decision"]

    if decision not in {"BUY", "NO TRADE"}:
        raise ValueError(
            "decision must be BUY or NO TRADE."
        )

    if decision == "NO TRADE":
        levels = {
            "entry": None,
            "stop_loss": None,
            "tp1": None,
            "tp2": None,
            "tp3": None,
        }
    else:
        levels = {
            "entry": _validate_price(
                "entry_price",
                display["entry_price"],
            ),
            "stop_loss": _validate_price(
                "stop_loss",
                display["stop_loss"],
            ),
            "tp1": _validate_price(
                "tp1",
                display["tp1"],
            ),
            "tp2": _validate_price(
                "tp2",
                display["tp2"],
            ),
            "tp3": _validate_price(
                "tp3",
                display["tp3"],
            ),
        }

        if not levels["stop_loss"] < levels["entry"]:
            raise ValueError(
                "BUY stop_loss must be below entry."
            )

        if not (
            levels["entry"]
            < levels["tp1"]
            < levels["tp2"]
            < levels["tp3"]
        ):
            raise ValueError(
                "BUY levels must satisfy "
                "SL < Entry < TP1 < TP2 < TP3."
            )

    timestamp = data["timestamp"].iloc[-1]

    return {
        "symbol": display.get("symbol"),
        "interval": display.get("interval"),
        "decision": decision,
        "trend": display.get("trend"),
        "stable_strategy": display.get("stable_strategy"),
        "stability_score": display.get("stability_score"),
        "signal_label": display.get("signal_label"),
        "timestamp": timestamp,
        "levels": levels,
        "lines": [
            {
                "name": "Entry",
                "price": levels["entry"],
                "role": "entry",
                "visible": levels["entry"] is not None,
            },
            {
                "name": "SL",
                "price": levels["stop_loss"],
                "role": "stop_loss",
                "visible": levels["stop_loss"] is not None,
            },
            {
                "name": "TP1",
                "price": levels["tp1"],
                "role": "take_profit_1",
                "visible": levels["tp1"] is not None,
            },
            {
                "name": "TP2",
                "price": levels["tp2"],
                "role": "take_profit_2",
                "visible": levels["tp2"] is not None,
            },
            {
                "name": "TP3",
                "price": levels["tp3"],
                "role": "take_profit_3",
                "visible": levels["tp3"] is not None,
            },
        ],
    }
