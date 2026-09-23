"""Batch inference on future unseen data and automated data drift analysis."""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    MODEL_ARTIFACT_PATH,
    FUTURE_UNSEEN_PATH,
    PREDICTIONS_OUTPUT_PATH,
    DATA_DRIFT_REPORT_PATH,
    KC_HOUSES_PATH
)
from src.data.loader import load_future_unseen, load_kc_houses

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_drift_analysis(train_df: pd.DataFrame, future_df: pd.DataFrame) -> dict:
    """
    Computes statistical Kolmogorov-Smirnov distribution drift and summary shifts
    between the baseline training features and the future unseen records.
    """
    logger.info("Computing feature drift analysis (KS-Test & Distribution Shifts)...")
    drift_report = {
        "sample_sizes": {
            "training_samples": len(train_df),
            "future_unseen_samples": len(future_df)
        },
        "features": {},
        "overall_drift_detected": False
    }

    drift_count = 0
    common_numeric = [c for c in future_df.select_dtypes(include=[np.number]).columns if c in train_df.columns]

    for col in common_numeric:
        s_train = train_df[col].dropna()
        s_future = future_df[col].dropna()

        # Two-sample Kolmogorov-Smirnov test
        ks_stat, p_val = ks_2samp(s_train, s_future)
        is_drift = bool(p_val < 0.05)
        if is_drift:
            drift_count += 1

        drift_report["features"][col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 5),
            "drift_detected": is_drift,
            "train_stats": {
                "mean": round(float(s_train.mean()), 2),
                "std": round(float(s_train.std()), 2),
                "median": round(float(s_train.median()), 2)
            },
            "future_stats": {
                "mean": round(float(s_future.mean()), 2),
                "std": round(float(s_future.std()), 2),
                "median": round(float(s_future.median()), 2)
            }
        }

    drift_report["drift_feature_ratio"] = round(drift_count / max(1, len(common_numeric)), 3)
    drift_report["overall_drift_detected"] = bool(drift_report["drift_feature_ratio"] > 0.3)
    return drift_report


def run_batch_predictions():
    logger.info("=== Running Batch Predictions on future_unseen_examples.csv ===")
    
    # 1. Load Model Pipeline
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_ARTIFACT_PATH}. Please run training first.")
    
    pipeline = joblib.load(MODEL_ARTIFACT_PATH)
    logger.info("Loaded trained model pipeline successfully.")

    # 2. Load Unseen Data
    df_future = load_future_unseen(validate=True)
    logger.info(f"Loaded {len(df_future)} future unseen records.")

    # 3. Predict Prices
    preds = pipeline.predict(df_future)
    
    # 4. Format Predictions
    df_output = df_future.copy()
    df_output["predicted_price"] = np.round(preds, 2)
    df_output["predicted_price_formatted"] = [f"${p:,.2f}" for p in preds]

    # Save CSV
    PREDICTIONS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    logger.info(f"Saved predictions to {PREDICTIONS_OUTPUT_PATH}")

    # 5. Compute and Save Drift Report
    df_train = load_kc_houses(validate=True)
    drift_report = compute_drift_analysis(df_train, df_future)

    DATA_DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=4)
    logger.info(f"Saved drift report to {DATA_DRIFT_REPORT_PATH}")

    # Summary
    logger.info(f"Predictions complete! Mean predicted price: ${preds.mean():,.2f}, Median: ${np.median(preds):,.2f}")
    return df_output


if __name__ == "__main__":
    run_batch_predictions()
