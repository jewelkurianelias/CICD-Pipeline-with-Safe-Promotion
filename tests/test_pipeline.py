import os
import pandas as pd
from src.generate_data import generate_data

def test_data_generation_creates_files():
    """Simulates a CI pipeline test ensuring data extraction works."""
    generate_data()
    assert os.path.exists("data/train.csv")
    assert os.path.exists("data/holdout.csv")

def test_data_has_correct_features():
    """Ensures upstream schema hasn't unexpectedly changed."""
    df = pd.read_csv("data/train.csv")
    expected_columns = ["date", "marketing_spend", "holiday", "orders"]
    assert all(col in df.columns for col in expected_columns)