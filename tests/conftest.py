import os
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def isolate_live_decision_history_store():
    """Isolate or reset results/live/decision_history.json between test runs to prevent replay conflicts."""
    store_file = Path("results/live/decision_history.json")
    if store_file.exists():
        try:
            store_file.unlink()
        except OSError:
            pass
    yield
    if store_file.exists():
        try:
            store_file.unlink()
        except OSError:
            pass
