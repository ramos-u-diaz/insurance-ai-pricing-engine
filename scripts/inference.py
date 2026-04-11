# scripts/inference.py
import os
import json
import tarfile
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys

# Install xgboost in the inference container
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "xgboost==1.7.6", "-q"
])

import xgboost

# ── These four functions are the SageMaker inference contract ─────────────────
# SageMaker calls them in this order:
#   model_fn      → load your model when the endpoint starts
#   input_fn      → parse incoming request data
#   predict_fn    → run the model
#   output_fn     → format the response

def model_fn(model_dir):
    """Load model and feature columns from model directory."""
    print("Loading model...")
    
    # Extract model.tar.gz if needed
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(model_dir)
    
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    with open(os.path.join(model_dir, "feature_columns.json"), "r") as f:
        feature_columns = json.load(f)
    
    print(f"Model loaded with {len(feature_columns)} features")
    return {"model": model, "feature_columns": feature_columns}


def input_fn(request_body, content_type="application/json"):
    """Parse incoming request into a dataframe."""
    if content_type == "application/json":
        data = json.loads(request_body)
        # Accept either a single policy dict or a list of policies
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_df, model_artifacts):
    """Score the input dataframe."""
    model          = model_artifacts["model"]
    feature_columns = model_artifacts["feature_columns"]
    
    xgb_features = [
        'DrivAge', 'VehAge', 'VehPower', 'BonusMalus', 'Area', 'VehGas',
        'Density', 'DrivAge_Group', 'VehAge_Group', 'High_Power', 'Log_Density'
    ]
    
    # Apply same feature engineering as training
    input_df = input_df.copy()
    
    if 'Log_Density' not in input_df.columns and 'Density' in input_df.columns:
        input_df['Log_Density'] = np.log1p(input_df['Density'])
    
    if 'High_Power' not in input_df.columns and 'VehPower' in input_df.columns:
        input_df['High_Power'] = (input_df['VehPower'] >= 9).astype(int)
    
    # One-hot encode and align to training columns
    X = pd.get_dummies(input_df[xgb_features], drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)
    
    predictions = model.predict(X)
    return predictions


def output_fn(predictions, accept="application/json"):
    """Format predictions as JSON response."""
    result = {
        "predictions": predictions.tolist(),
        "unit": "expected_claims_per_year"
    }
    return json.dumps(result), "application/json"