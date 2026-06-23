import pandas as pd
import numpy as np
import os

def generate_data():
    """Generates synthetic daily demand data for a small online store."""
    np.random.seed(42)
    
    # 365 days of data
    dates = pd.date_range(start="2023-01-01", periods=365)
    
    # Features
    marketing_spend = np.random.uniform(100, 1000, size=365)
    holiday = np.random.choice([0, 1], size=365, p=[0.9, 0.1])
    
    # Target: Orders (Formula: Base + Marketing impact + Holiday impact + Noise)
    orders = 50 + (marketing_spend * 0.1) + (holiday * 50) + np.random.normal(0, 10, size=365)
    
    df = pd.DataFrame({
        "date": dates, 
        "marketing_spend": marketing_spend, 
        "holiday": holiday, 
        "orders": orders
    })
    
    os.makedirs("data", exist_ok=True)
    
    # Split into training set and a strict holdout set for the evaluation gate
    train_df = df.iloc[:-30]
    holdout_df = df.iloc[-30:]
    
    train_df.to_csv("data/train.csv", index=False)
    holdout_df.to_csv("data/holdout.csv", index=False)
    
    print("✅ Synthetic data generated in data/ directory.")

if __name__ == "__main__":
    generate_data()