"""Unit tests for data loading and schema validation."""
import pytest
import pandas as pd
import numpy as np
from src.data.loader import validate_house_dataframe, load_demographics


def test_validate_house_dataframe_bedrooms_typo():
    sample_df = pd.DataFrame([{
        "bedrooms": 33,
        "bathrooms": 1.75,
        "sqft_living": 1620,
        "sqft_lot": 5000,
        "sqft_above": 1620,
        "zipcode": "98178",
        "price": 640000
    }])
    cleaned = validate_house_dataframe(sample_df, is_training=True)
    assert cleaned.loc[0, "bedrooms"] == 3
    assert cleaned.loc[0, "zipcode"] == 98178


def test_validate_house_dataframe_non_positive_sqft():
    sample_df = pd.DataFrame([{
        "bedrooms": 3,
        "bathrooms": 2.0,
        "sqft_living": -50,
        "sqft_lot": 5000,
        "sqft_above": 1000,
        "zipcode": 98178,
        "price": 500000
    }])
    with pytest.raises(ValueError):
        validate_house_dataframe(sample_df, is_training=True)


def test_demographics_has_zipcode():
    demo = load_demographics()
    assert "zipcode" in demo.columns
    assert len(demo) == 70
