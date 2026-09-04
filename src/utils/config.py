from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "configs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
