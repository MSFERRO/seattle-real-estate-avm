"""Data ingestion and validation routines."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from src.config import KC_HOUSES_PATH, DEMOGRAPHICS_PATH, FUTURE_UNSEEN_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def validate_house_dataframe(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Performs vectorized validation and cleaning of house properties data.
    Ensures non-negative values, realistic boundaries, and fixes known anomalies.
    """
    df = df.copy()

    # Fix known King County dataset typo: house with 33 bedrooms and 1.75 baths, 1620 sqft
    if "bedrooms" in df.columns:
        typo_mask = (df["bedrooms"] == 33) & (df["sqft_living"] < 2000)
        if typo_mask.any():
            logger.info("Fixing known anomaly: house with 33 bedrooms corrected to 3 bedrooms.")
            df.loc[typo_mask, "bedrooms"] = 3

        # Impute/correct 0 bedrooms or 0 bathrooms if present using median of small houses
        zero_bed = df["bedrooms"] == 0
        if zero_bed.any():
            df.loc[zero_bed, "bedrooms"] = 1
        
        zero_bath = df["bathrooms"] == 0
        if zero_bath.any():
            df.loc[zero_bath, "bathrooms"] = 1.0

    # Ensure physical sqft columns are positive
    for col in ["sqft_living", "sqft_lot", "sqft_above"]:
        if col in df.columns:
            if (df[col] <= 0).any():
                raise ValueError(f"Feature {col} contains non-positive values.")

    # Target validation in training
    if is_training and "price" in df.columns:
        if (df["price"] <= 0).any():
            raise ValueError("Training data contains invalid non-positive prices.")

    # Ensure zipcode is integer
    if "zipcode" in df.columns:
        df["zipcode"] = df["zipcode"].astype(str).str.strip().str.replace('"', '').astype(int)

    return df


def load_kc_houses(path: Optional[str] = None, validate: bool = True) -> pd.DataFrame:
    """Load and validate King County residential sales training data."""
    target_path = path or KC_HOUSES_PATH
    logger.info(f"Loading KC Houses dataset from {target_path}")
    df = pd.read_csv(target_path)
    if validate:
        df = validate_house_dataframe(df, is_training=True)
    return df


def load_demographics(path: Optional[str] = None) -> pd.DataFrame:
    """Load demographic statistics per zipcode."""
    target_path = path or DEMOGRAPHICS_PATH
    logger.info(f"Loading demographics dataset from {target_path}")
    df = pd.read_csv(target_path)
    df["zipcode"] = df["zipcode"].astype(int)
    return df


def load_future_unseen(path: Optional[str] = None, validate: bool = True) -> pd.DataFrame:
    """Load and validate future unseen houses dataset."""
    target_path = path or FUTURE_UNSEEN_PATH
    logger.info(f"Loading future unseen dataset from {target_path}")
    df = pd.read_csv(target_path)
    if validate:
        df = validate_house_dataframe(df, is_training=False)
    return df
