# scripts/train.py
import argparse
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, default="/opt/ml/input/data/train")
parser.add_argument("--val-data",   type=str, default="/opt/ml/input/data/val")
parser.add_argument("--model-dir",  type=str, default="/opt/ml/model")
# XGBoost hyperparameters — same values you used in your notebook
parser.add_argument("--n-estimators",    type=int,   default=300)
parser.add_argument("--max-depth",       type=int,   default=8)
parser.add_argument("--learning-rate",   type=float, default=0.05)
parser.add_argument("--subsample",       type=float, default=0.8)
parser.add_argument("--colsample-bytree",type=float, default=0.8)
args = parser.parse_args()

# ── Load train and validation splits ─────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv(os.path.join(args.train_data, "train.csv"))
val_df   = pd.read_csv(os.path.join(args.val_data,   "val.csv"))

print(f"Train shape : {train_df.shape}")
print(f"Val shape   : {val_df.shape}")

# ── Separate features and target ──────────────────────────────────────────────
TARGET = "Frequency"

xgb_features = ['DrivAge', 'VehAge', 'VehPower', 'BonusMalus', 'Area', 'VehGas',
                 'Density', 'DrivAge_Group', 'VehAge_Group', 'High_Power', 'Log_Density']

X_train = train_df[xgb_features]
y_train = train_df[TARGET]
X_val   = val_df[xgb_features]
y_val   = val_df[TARGET]

# ── One-hot encode categoricals — same as your notebook ──────────────────────
X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_val_enc   = pd.get_dummies(X_val,   drop_first=True)

# Align columns — val may be missing a dummy column if a category didn't appear
X_train_enc, X_val_enc = X_train_enc.align(
    X_val_enc, join='left', axis=1, fill_value=0)

print(f"Encoded feature count: {X_train_enc.shape[1]}")

# ── Save the column order — critical for inference later ──────────────────────
# When a new policy comes in for scoring, we need to encode it
# in exactly the same column order the model was trained on
os.makedirs(args.model_dir, exist_ok=True)
feature_columns = X_train_enc.columns.tolist()
with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
    json.dump(feature_columns, f)
print(f"✓ Saved {len(feature_columns)} feature column names")

# ── Train XGBoost — your exact same parameters ────────────────────────────────
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators     = args.n_estimators,
    max_depth        = args.max_depth,
    learning_rate    = args.learning_rate,
    subsample        = args.subsample,
    colsample_bytree = args.colsample_bytree,
    random_state     = 42,
    eval_metric      = "rmse",
)

model.fit(
    X_train_enc, y_train,
    eval_set=[(X_val_enc, y_val)],
    verbose=50,   # print progress every 50 rounds
)

# ── Validate ──────────────────────────────────────────────────────────────────
y_pred_val = model.predict(X_val_enc)
mae  = mean_absolute_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"\nValidation results:")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  Mean frequency (actual): {y_val.mean():.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(args.model_dir, "model.joblib")
joblib.dump(model, model_path)
print(f"✓ Model saved to {model_path}")