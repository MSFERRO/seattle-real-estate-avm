"""Custom Scikit-Learn transformers for spatial, temporal, and structural feature engineering."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import logging
from src.config import SEATTLE_DOWNTOWN, BELLEVUE_DOWNTOWN, BASE_YEAR

logger = logging.getLogger(__name__)


def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Computes great-circle distance between coordinates in kilometers."""
    R = 6371.0 # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Constructs high-signal domain features from physical characteristics,
    geographic coordinates, and demographic indicators.
    """
    def __init__(self, base_year: int = BASE_YEAR):
        self.base_year = base_year
        self.feature_names_: list = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Determine reference evaluation year
        if "valuation_date" in df.columns and df["valuation_date"].notna().any():
            eval_year = pd.to_datetime(df["valuation_date"], errors="coerce").dt.year.fillna(self.base_year).values
        else:
            eval_year = self.base_year

        # 1. Temporal & Age Features (deterministic based on reference year or valuation date)
        df["house_age"] = np.maximum(0, eval_year - df["yr_built"])
        df["is_renovated"] = (df["yr_renovated"] > 0).astype(int)
        df["years_since_renovation"] = np.where(
            df["yr_renovated"] > 0,
            np.maximum(0, eval_year - df["yr_renovated"]),
            df["house_age"]
        )

        # 2. Structural & Space Ratios
        df["total_baths_beds"] = df["bedrooms"] + df["bathrooms"]
        df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 0.1)
        df["sqft_per_room"] = df["sqft_living"] / (df["total_baths_beds"] + 1.0)
        df["living_to_lot_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1.0)
        df["has_basement"] = (df["sqft_basement"] > 0).astype(int)
        df["basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1.0)
        
        # Neighborhood comparative ratios (pre-computed King County neighborhood features)
        df["living_vs_neighbor_ratio"] = df["sqft_living"] / (df["sqft_living15"] + 1.0)
        df["lot_vs_neighbor_ratio"] = df["sqft_lot"] / (df["sqft_lot15"] + 1.0)

        # 3. Spatial & Commute Features
        lat = df["lat"].values
        lon = df["long"].values
        df["dist_seattle_km"] = haversine_distance(lat, lon, SEATTLE_DOWNTOWN[0], SEATTLE_DOWNTOWN[1])
        df["dist_bellevue_km"] = haversine_distance(lat, lon, BELLEVUE_DOWNTOWN[0], BELLEVUE_DOWNTOWN[1])
        df["min_dist_tech_hub_km"] = np.minimum(df["dist_seattle_km"], df["dist_bellevue_km"])

        # 4. Socioeconomic Interactivity Features (if demographics present)
        if "hous_val_amt" in df.columns:
            # Composite education index
            if "per_bchlr" in df.columns and "per_prfsnl" in df.columns:
                df["high_edctn_ratio"] = df["per_bchlr"] + df["per_prfsnl"]
            
            # Affluence score
            if "medn_hshld_incm_amt" in df.columns:
                high_ed = df.get("high_edctn_ratio", 20.0)
                df["affluence_score"] = df["medn_hshld_incm_amt"] * (1.0 + high_ed / 100.0)

            # Property size scaled by neighborhood house valuation
            df["living_x_zip_val"] = df["sqft_living"] * np.log1p(df["hous_val_amt"])

        # 5. Clean non-feature and identifier columns
        cols_to_drop = [c for c in ["id", "date", "price", "valuation_date"] if c in df.columns]
        df = df.drop(columns=cols_to_drop)

        self.feature_names_ = list(df.columns)
        return df


def build_full_pipeline(regressor, demographics_df: pd.DataFrame = None):
    """
    Assembles complete end-to-end pipeline:
    Demographics Enrichment -> Feature Engineering -> Target Log-Transformed Regressor
    """
    from src.data.enricher import DemographicsEnricher

    preprocessing_pipeline = Pipeline([
        ("enricher", DemographicsEnricher(demographics_df=demographics_df)),
        ("engineer", HouseFeatureEngineer(base_year=BASE_YEAR))
    ])

    # Target log-transform: log1p(y) in training, expm1(y) in inference
    model_pipeline = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

    full_pipeline = Pipeline([
        ("prep", preprocessing_pipeline),
        ("model", model_pipeline)
    ])

    return full_pipeline
