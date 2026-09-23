"""Demographic enrichment transformer with robust fallback for unseen zipcodes."""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
from typing import Optional
from src.data.loader import load_demographics

logger = logging.getLogger(__name__)


class DemographicsEnricher(BaseEstimator, TransformerMixin):
    """
    Enriches house records with census demographic indicators based on zipcode.
    Includes a learned county-level fallback (median vector) to guarantee
    zero inference failure when predicting unseen or future zipcodes.
    """
    def __init__(self, demographics_df: Optional[pd.DataFrame] = None):
        self.demographics_df = demographics_df
        self.lookup_table_: Optional[pd.DataFrame] = None
        self.fallback_values_: Optional[pd.Series] = None
        self.feature_columns_: Optional[list] = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.demographics_df is None:
            self.demographics_df = load_demographics()

        demo = self.demographics_df.copy()
        demo["zipcode"] = demo["zipcode"].astype(int)
        
        # Demographic feature columns (all columns except zipcode)
        self.feature_columns_ = [c for c in demo.columns if c != "zipcode"]
        
        # Deduplicate on zipcode if needed
        self.lookup_table_ = demo.drop_duplicates(subset=["zipcode"]).set_index("zipcode")
        
        # Learn county-level median fallback for each demographic feature
        self.fallback_values_ = self.lookup_table_[self.feature_columns_].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.lookup_table_ is None:
            raise RuntimeError("DemographicsEnricher is not fitted yet.")

        X_df = X.copy()
        original_index = X_df.index
        zip_series = X_df["zipcode"].astype(str).str.strip().str.replace('"', '').astype(int)

        # Merge with lookup table
        enriched = zip_series.to_frame().join(self.lookup_table_, on="zipcode", how="left")
        
        # Identify missing zipcodes and fill with learned fallback
        missing_mask = enriched[self.feature_columns_[0]].isna()
        if missing_mask.any():
            missing_count = missing_mask.sum()
            logger.warning(f"Encountered {missing_count} records with unmapped zipcodes. Imputing with learned county medians.")
            enriched.loc[missing_mask, self.feature_columns_] = self.fallback_values_.values

        # Attach demographic features back to X_df
        for col in self.feature_columns_:
            X_df[col] = enriched[col].values

        X_df.index = original_index
        return X_df
