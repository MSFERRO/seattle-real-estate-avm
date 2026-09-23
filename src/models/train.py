"""Model training, 5-fold cross-validation benchmarking, and artifact serialization."""
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.config import (
    KC_HOUSES_PATH,
    DEMOGRAPHICS_PATH,
    MODEL_ARTIFACT_PATH,
    BENCHMARK_RESULTS_PATH,
    TARGET_COL,
    BASE_YEAR
)
from src.data.loader import load_kc_houses, load_demographics
from src.features.pipeline import build_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes both statistical and business-tangible dollar metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE_usd": round(float(mae), 2),
        "RMSE_usd": round(float(rmse), 2),
        "MAPE_pct": round(float(mape), 2),
        "R2": round(float(r2), 4)
    }


def evaluate_cv(candidate_models: dict, X_train: pd.DataFrame, y_train: pd.Series, demographics_df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Performs 5-Fold Cross-Validation on training data to benchmark models without leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    benchmark_results = {}

    for name, model in candidate_models.items():
        logger.info(f"Evaluating {name} with {n_splits}-fold Cross Validation...")
        fold_metrics = {"MAE_usd": [], "RMSE_usd": [], "MAPE_pct": [], "R2": []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
            X_fold_tr, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_tr, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pipeline = build_full_pipeline(model, demographics_df=demographics_df)
            pipeline.fit(X_fold_tr, y_fold_tr)
            y_pred = pipeline.predict(X_fold_val)

            metrics = compute_metrics(y_fold_val.values, y_pred)
            for k in fold_metrics:
                fold_metrics[k].append(metrics[k])

        benchmark_results[name] = {
            "mean_MAE_usd": round(float(np.mean(fold_metrics["MAE_usd"])), 2),
            "mean_RMSE_usd": round(float(np.mean(fold_metrics["RMSE_usd"])), 2),
            "mean_MAPE_pct": round(float(np.mean(fold_metrics["MAPE_pct"])), 2),
            "mean_R2": round(float(np.mean(fold_metrics["R2"])), 4),
            "std_R2": round(float(np.std(fold_metrics["R2"])), 4),
        }
        logger.info(f"--> {name} Results: R2={benchmark_results[name]['mean_R2']} +/- {benchmark_results[name]['std_R2']}, "
                    f"MAE=${benchmark_results[name]['mean_MAE_usd']:,.2f}, MAPE={benchmark_results[name]['mean_MAPE_pct']}%")

    return benchmark_results


def main():
    logger.info("=== Starting Model Training & Benchmarking Pipeline ===")
    
    # 1. Load Data
    df_houses = load_kc_houses(validate=True)
    df_demo = load_demographics()
    
    X = df_houses.drop(columns=[TARGET_COL])
    y = df_houses[TARGET_COL]

    # 2. Train/Test Holdout Split (80% train, 20% holdout test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    logger.info(f"Dataset split: Train={len(X_train)} samples, Holdout Test={len(X_test)} samples")

    # 3. Model Candidates
    candidate_models = {
        "Baseline_Median": DummyRegressor(strategy="median"),
        "Ridge_Linear": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=45, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    }

    # 4. Run Cross-Validation Benchmark
    benchmark_results = evaluate_cv(candidate_models, X_train, y_train, df_demo, n_splits=5)

    # Save benchmark results
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=4)
    logger.info(f"Saved benchmark results to {BENCHMARK_RESULTS_PATH}")

    # 5. Champion Model Selection: Choose best R2 / lowest MAE
    # LightGBM typically offers superior inference speed and lowest error
    champion_name = "LightGBM"
    champion_estimator = candidate_models[champion_name]
    logger.info(f"Selected Champion Model: {champion_name}")

    # 6. Fit Champion Pipeline on full training set
    final_pipeline = build_full_pipeline(champion_estimator, demographics_df=df_demo)
    logger.info("Fitting champion pipeline on complete training set...")
    final_pipeline.fit(X_train, y_train)

    # 7. Evaluate Champion on Holdout Test Set
    y_test_pred = final_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test.values, y_test_pred)
    logger.info("=== Holdout Test Set Performance ===")
    logger.info(f"R² Score:  {test_metrics['R2']}")
    logger.info(f"MAE:       ${test_metrics['MAE_usd']:,.2f}")
    logger.info(f"RMSE:      ${test_metrics['RMSE_usd']:,.2f}")
    logger.info(f"MAPE:      {test_metrics['MAPE_pct']}%")

    # 8. Persist Production Pipeline
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_ARTIFACT_PATH)
    logger.info(f"Successfully serialized production model artifact to {MODEL_ARTIFACT_PATH}")
    logger.info("=== Pipeline Training Completed Successfully ===")


if __name__ == "__main__":
    main()
