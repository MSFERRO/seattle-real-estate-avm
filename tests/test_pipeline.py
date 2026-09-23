"""Unit tests for feature engineering and demographics enricher."""
import pandas as pd
import numpy as np
from src.data.enricher import DemographicsEnricher
from src.features.pipeline import HouseFeatureEngineer


def test_demographics_enricher_with_unseen_zipcode():
    enricher = DemographicsEnricher()
    enricher.fit(pd.DataFrame())

    # 99999 is an unmapped zipcode
    test_df = pd.DataFrame([{"zipcode": 99999, "sqft_living": 2000}])
    transformed = enricher.transform(test_df)
    
    assert "hous_val_amt" in transformed.columns
    # Check that fallback imputed non-null value
    assert not pd.isna(transformed.loc[0, "hous_val_amt"])


def test_house_feature_engineer():
    engineer = HouseFeatureEngineer(base_year=2015)
    sample_df = pd.DataFrame([{
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": 2000,
        "sqft_lot": 5000,
        "sqft_above": 1500,
        "sqft_basement": 500,
        "sqft_living15": 1800,
        "sqft_lot15": 4800,
        "yr_built": 1995,
        "yr_renovated": 0,
        "lat": 47.6062,
        "long": -122.3321,
        "hous_val_amt": 300000,
        "per_bchlr": 25.0,
        "per_prfsnl": 15.0,
        "medn_hshld_incm_amt": 80000
    }])
    transformed = engineer.transform(sample_df)

    assert "house_age" in transformed.columns
    assert transformed.loc[0, "house_age"] == 20
    assert transformed.loc[0, "has_basement"] == 1
    assert "dist_seattle_km" in transformed.columns
    assert "affluence_score" in transformed.columns
