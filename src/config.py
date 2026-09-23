"""Global configuration, paths, and business constants."""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# File Paths
KC_HOUSES_PATH = RAW_DATA_DIR / "kc_house_data.csv"
DEMOGRAPHICS_PATH = RAW_DATA_DIR / "zipcode_demographics.csv"
FUTURE_UNSEEN_PATH = RAW_DATA_DIR / "future_unseen_examples.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "model_pipeline.joblib"
DATA_DRIFT_REPORT_PATH = REPORTS_DIR / "data_drift_report.json"
BENCHMARK_RESULTS_PATH = REPORTS_DIR / "model_benchmark_results.json"
PREDICTIONS_OUTPUT_PATH = PREDICTIONS_DIR / "future_unseen_predictions.csv"

# Economic & Spatial Constants
# King County tech hubs
SEATTLE_DOWNTOWN = (47.6062, -122.3321)  # (lat, long)
BELLEVUE_DOWNTOWN = (47.6101, -122.2015) # (lat, long)

# Base Year for Age Calculation (Dataset sales are from May 2014 - May 2015)
BASE_YEAR = 2015

# Target column
TARGET_COL = "price"

# Identifier and temporal columns not directly used as numeric regression features
METADATA_COLS = ["id", "date"]
