import argparse
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from mlflow.models import infer_signature

def train(degrade=False):
    """Trains a model and registers it as a candidate."""
    mlflow.set_experiment("demand_forecasting")
    
    train_df = pd.read_csv("data/train.csv")
    X_train = train_df[["marketing_spend", "holiday"]]
    y_train = train_df["orders"]
    
    with mlflow.start_run():
        if degrade:
            print("⚠️ Training intentionally degraded model (Max Depth=1)...")
            # A terrible model that will fail the evaluation gate
            model = RandomForestRegressor(max_depth=1, random_state=42)
        else:
            print("🧠 Training standard model...")
            # A good model
            model = RandomForestRegressor(max_depth=10, random_state=42)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        mae = mean_absolute_error(y_train, preds)
        
        mlflow.log_param("degraded_run", degrade)
        mlflow.log_metric("train_mae", mae)
        
        signature = infer_signature(X_train, preds)
        # 1. Log the model (without registering it yet)
        model_info = mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature
        )
        
        # 2. Explicitly register the model to safely get the version object
        model_version = mlflow.register_model(model_info.model_uri, "DemandForecaster")
        
        client = mlflow.tracking.MlflowClient()
        # 3. Mark this newly trained model as the latest "@candidate"
        client.set_registered_model_alias("DemandForecaster", "candidate", model_version.version)
        print(f"✅ Model trained. Version {model_version.version} marked as @candidate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--degrade", action="store_true", help="Intentionally train a worse model")
    args = parser.parse_args()
    train(args.degrade)