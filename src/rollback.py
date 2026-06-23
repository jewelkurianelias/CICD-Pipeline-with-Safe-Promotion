import argparse
import mlflow
import datetime

def rollback(target_version=None):
    """Reverts the @production alias to a previous safe version."""
    client = mlflow.tracking.MlflowClient()
    model_name = "DemandForecaster"
    
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production").version
        print(f"Current @production version is: {prod_version}")
    except Exception:
        print("❌ No production model found to roll back from.")
        return

    # Auto-rollback: Find the version right before current production
    if not target_version:
        versions = client.search_model_versions(f"name='{model_name}'")
        sorted_versions = sorted([int(v.version) for v in versions])
        
        current_idx = sorted_versions.index(int(prod_version))
        if current_idx == 0:
            print("❌ This is the first version. There is no older version to roll back to.")
            return
        target_version = str(sorted_versions[current_idx - 1])

    print(f"⏪ Rolling back to Version {target_version}...")
    
    # The Rollback is just shifting the alias pointer!
    client.set_registered_model_alias(model_name, "production", target_version)
    
    with open("deployment_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏪ ROLLED BACK from Version {prod_version} to Version {target_version}\n")
        
    print(f"✅ Successfully rolled back @production to Version {target_version}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, help="Specific version to roll back to", default=None)
    args = parser.parse_args()
    rollback(args.version)