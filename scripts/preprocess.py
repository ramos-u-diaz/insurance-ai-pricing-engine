# scripts/preprocess.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── SageMaker passes input/output paths as arguments ─────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input-data",  type=str, default="/opt/ml/processing/input")
parser.add_argument("--output-train",type=str, default="/opt/ml/processing/train")
parser.add_argument("--output-val",  type=str, default="/opt/ml/processing/val")
parser.add_argument("--output-test", type=str, default="/opt/ml/processing/test")
args = parser.parse_args()

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading raw data...")
input_path = os.path.join(args.input_data, "freMTPL2freq.csv")
df = pd.read_csv(input_path)
print(f"Raw shape: {df.shape}")

# ── Your existing feature engineering (copied directly from your notebook) ────
df['Frequency'] = df['ClaimNb'] / df['Exposure']
df['Frequency'] = df['Frequency'].clip(upper=df['Frequency'].quantile(0.99))

df['DrivAge_Group'] = pd.cut(df['DrivAge'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['18-25', '26-35', '36-50', '51-65', '65+'])

df['VehAge_Group'] = pd.cut(df['VehAge'],
    bins=[-1, 1, 5, 10, 20, 100],
    labels=['New', '1-5', '6-10', '11-20', 'Old'])

df['BonusMalus_Group'] = pd.cut(df['BonusMalus'],
    bins=[49, 60, 70, 80, 90, 100, 120, 150, 230],
    labels=['50-59', '60-69', '70-79', '80-89', '90-99',
            '100-119', '120-149', '150+'])

df['High_Power']   = (df['VehPower'] >= 9).astype(int)
df['Log_Density']  = np.log1p(df['Density'])

# ── Features + target ────────────────────────────────────────────────────────
features = ['DrivAge', 'VehAge', 'VehPower', 'BonusMalus', 'Area', 'VehGas',
            'Density', 'DrivAge_Group', 'VehAge_Group', 'BonusMalus_Group',
            'High_Power', 'Log_Density', 'Exposure']

df_model = df[features + ['Frequency', 'ClaimNb']].copy()

# ── Your existing 3-way stratified split ─────────────────────────────────────
X = df_model[features]
y = df_model['Frequency']
stratify_col = (df_model['ClaimNb'] > 0).astype(int)

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=stratify_col)

stratify_temp = stratify_col.loc[X_temp.index]
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=stratify_temp)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── Save splits to their output directories ───────────────────────────────────
os.makedirs(args.output_train, exist_ok=True)
os.makedirs(args.output_val,   exist_ok=True)
os.makedirs(args.output_test,  exist_ok=True)

train_df = X_train.copy(); train_df['Frequency'] = y_train
val_df   = X_val.copy();   val_df['Frequency']   = y_val
test_df  = X_test.copy();  test_df['Frequency']  = y_test

train_df.to_csv(os.path.join(args.output_train, "train.csv"), index=False)
val_df.to_csv(  os.path.join(args.output_val,   "val.csv"),   index=False)
test_df.to_csv( os.path.join(args.output_test,  "test.csv"),  index=False)

print("✓ Preprocessing complete — train/val/test saved")