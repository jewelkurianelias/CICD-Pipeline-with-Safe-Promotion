import pandas as pd
import mlflow
from sklearn.metrics import mean_absolute_error
import datetime
from mlflow.exceptions import MlflowException

def evaluate():
    """Compares candidate model against production. Promotes if better."""
    client = mlflow.tracking.MlflowClient()
    model_name = "DemandForecaster"
    
    # Load the strict holdout set
    holdout_df = pd.read_csv("data/holdout.csv")
    X_test = holdout_df[["marketing_spend", "holiday"]]
    y_test = holdout_df["orders"]
    
    # 1. Load Candidate Model
    print("🔍 Evaluating @candidate model...")
    try:
        candidate_model = mlflow.sklearn.load_model(f"models:/{model_name}@candidate")
        candidate_version = client.get_model_version_by_alias(model_name, "candidate").version
    except MlflowException:
        print("❌ No @candidate model found. Run train.py first.")
        return

    candidate_preds = candidate_model.predict(X_test)
    candidate_mae = mean_absolute_error(y_test, candidate_preds)
    print(f"   ↳ Candidate MAE: {candidate_mae:.2f}")

    # 2. Load Production Model (if exists)
    try:
        prod_model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
        prod_preds = prod_model.predict(X_test)
        prod_mae = mean_absolute_error(y_test, prod_preds)
        print(f"   ↳ Current Production MAE: {prod_mae:.2f}")
    except MlflowException:
        print("   ↳ No existing @production model found. Candidate will be promoted automatically.")
        prod_mae = float("inf")
    
    # 3. The Gate Logic
    if candidate_mae < prod_mae:
        print(f"\n🚀 SUCCESS: Candidate is better ({candidate_mae:.2f} < {prod_mae:.2f}).")
        print(f"   ↳ Promoting Version {candidate_version} to @production.")
        client.set_registered_model_alias(model_name, "production", candidate_version)
        log_deployment(candidate_version, candidate_mae, "Promoted")
    else:
        print(f"\n🛑 REJECTED: Candidate is worse ({candidate_mae:.2f} >= {prod_mae:.2f}).")
        print("   ↳ The production environment has been protected.")
        log_deployment(candidate_version, candidate_mae, "Rejected (Gate Failed)")

def log_deployment(version, mae, status):
    """Maintains a simple, auditable deployment log."""
    with open("deployment_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Version: {version} | Holdout MAE: {mae:.2f} | Action: {status}\n")

if __name__ == "__main__":
    evaluate()