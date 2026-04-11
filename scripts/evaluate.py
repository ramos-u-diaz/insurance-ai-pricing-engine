# scripts/evaluate.py
import subprocess
import sys

# Install dependencies not in the base sklearn container
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "xgboost==1.7.6", "joblib==1.2.0", "-q"
])

import argparse
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--test-data",  type=str, default="/opt/ml/processing/test")
parser.add_argument("--model-dir",  type=str, default="/opt/ml/processing/model")
parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
args = parser.parse_args()

# ── Load test data ────────────────────────────────────────────────────────────
print("Loading test data...")
test_df = pd.read_csv(os.path.join(args.test_data, "test.csv"))

TARGET = "Frequency"
xgb_features = ['DrivAge', 'VehAge', 'VehPower', 'BonusMalus', 'Area', 'VehGas',
                 'Density', 'DrivAge_Group', 'VehAge_Group', 'High_Power', 'Log_Density']

X_test = test_df[xgb_features]
y_test = test_df[TARGET]

# ── Load model and feature columns ────────────────────────────────────────────
print("Loading model...")

# TrainingJob outputs a model.tar.gz — extract it first
import tarfile

model_path = os.path.join(args.model_dir, "model.tar.gz")
if os.path.exists(model_path):
    print("Extracting model.tar.gz...")
    with tarfile.open(model_path) as tar:
        tar.extractall(args.model_dir)
    print("Extraction complete")

model = joblib.load(os.path.join(args.model_dir, "model.joblib"))

with open(os.path.join(args.model_dir, "feature_columns.json"), "r") as f:
    feature_columns = json.load(f)

print(f"Model loaded, {len(feature_columns)} features")

# ── Encode test set using saved column order ──────────────────────────────────
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_test_enc = X_test_enc.reindex(columns=feature_columns, fill_value=0)

# ── Score ─────────────────────────────────────────────────────────────────────
print("Scoring test set...")
y_pred = model.predict(X_test_enc)

# ── Metrics ───────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mean_freq = float(y_test.mean())

print(f"\nTest set results:")
print(f"  MAE          : {mae:.4f}")
print(f"  RMSE         : {rmse:.4f}")
print(f"  Mean frequency (actual) : {mean_freq:.4f}")

# ── Save metrics as JSON ──────────────────────────────────────────────────────
# This is what SageMaker Model Registry reads to decide if the model
# is good enough to register — a critical production gate
metrics = {
    "regression_metrics": {
        "mae":  {"value": round(mae,  4)},
        "rmse": {"value": round(rmse, 4)},
        "mean_frequency": {"value": round(mean_freq, 4)},
    }
}

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, "evaluation.json")
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Evaluation complete — metrics saved to {output_path}")