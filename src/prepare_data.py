import pandas as pd
from preprocessing import (DATA_DIR, WARMUP_DAYS, STARTUP_MINUTES,
                           RUNNING_THRESHOLD, load_plant_file, day_split, steady_state_mask)

FAULT_FREE_FILE = f"{DATA_DIR}/ChillerPlant.csv"

# --- Step 1: confirm completeness of the fault-free file ---
print("Loading fault-free file...")
# load_plant_file also swaps OA_TEMP / OA_TEMP_WB back (see preprocessing.py)
df = load_plant_file(FAULT_FREE_FILE)
assert (df["OA_TEMP_WB"] <= df["OA_TEMP"] + 0.5).all(), "wet-bulb above dry-bulb"

print(f"Shape: {df.shape}")
print(f"Total missing values: {df.isnull().sum().sum()}")

if df.isnull().sum().sum() == 0:
    print("OK - no missing values. Data is complete.")
else:
    print("WARNING - found missing values:")
    print(df.isnull().sum())

# --- Step 2: catalog all files and parse their labels ---
import os
import re

def parse_filename(filename):
    """Given a CSV filename, return (label, fault_type, severity)."""
    name = filename.replace(".csv", "")

    # The fault-free file is just "ChillerPlant"
    if name == "ChillerPlant":
        return ("normal", "none", None)

    # Fault files look like: ChillerPlant_<type>_<severity>
    # e.g. ChillerPlant_coolingtower_fouling_095
    rest = name.replace("ChillerPlant_", "")

    # Severity is the trailing number (e.g. 095, 050, -1, 020)
    match = re.search(r"_(-?\d+)$", rest)
    if match:
        severity = match.group(1)
        fault_type = rest[:match.start()]
    else:
        severity = None            # e.g. coolingtower_PI (no numeric severity)
        fault_type = rest

    return ("fault", fault_type, severity)


print("\n--- Cataloging all files ---")
all_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
print(f"Found {len(all_files)} CSV files.\n")

for f in all_files:
    label, fault_type, severity = parse_filename(f)
    print(f"{f:60s} -> {label:7s} | type={fault_type:20s} | severity={severity}")


# --- Step 2b: build a catalog table (instead of just printing) ---
catalog = []
for f in all_files:
    label, fault_type, severity = parse_filename(f)
    catalog.append({
        "filename": f,
        "label": label,
        "fault_type": fault_type,
        "severity": severity,
    })

catalog_df = pd.DataFrame(catalog)

print("\n--- Catalog summary ---")
print(f"Total files:       {len(catalog_df)}")
print(f"Normal files:      {(catalog_df['label'] == 'normal').sum()}")
print(f"Fault files:       {(catalog_df['label'] == 'fault').sum()}")

print("\nFault files by type:")
print(catalog_df[catalog_df['label'] == 'fault']['fault_type'].value_counts())

# Save the catalog for later steps
catalog_df.to_csv("data/file_catalog.csv", index=False)
print("\nCatalog saved to data/file_catalog.csv")

# --- Step 3: split the fault-free data by whole days (70/15/15) ---
# Whole days go to train/val/test at random, and only steady-state rows are
# kept (see day_split and steady_state_mask in preprocessing.py).
import numpy as np

print("\n--- Splitting fault-free data (whole days, 70/15/15) ---")

row_split = day_split(len(df))
keep, running, startup, warmup = steady_state_mask(df)

print(f"Rows with chiller running (> {RUNNING_THRESHOLD} kW): "
      f"{running.sum()} of {len(df)} ({running.mean()*100:.0f}%)")
print(f"  skipped - first {WARMUP_DAYS} day(s) (simulation warm-up): {warmup.sum()} rows")
print(f"  skipped - first {STARTUP_MINUTES} min after a chiller start: {(startup & ~warmup).sum()} rows")

train_df = df[keep & (row_split == "train")]
val_df   = df[keep & (row_split == "val")]
test_normal_df = df[keep & (row_split == "test")]

n = keep.sum()
print(f"Total fault-free steady-state rows: {n}")
print(f"  Train:       {len(train_df):>7} rows ({len(train_df)/n*100:.0f}%)")
print(f"  Validation:  {len(val_df):>7} rows ({len(val_df)/n*100:.0f}%)")
print(f"  Test-normal: {len(test_normal_df):>7} rows ({len(test_normal_df)/n*100:.0f}%)")

# Sanity check: the three parts must add up to the whole, with no overlap
assert len(train_df) + len(val_df) + len(test_normal_df) == n
print("OK - splits add up correctly, no overlap.")

# --- Step 3b: drop sensors that never change in the training data ---
# A constant sensor (std = 0) carries no information, and StandardScaler
# turns it into all zeros, which drags the overall std below 1.
constant_cols = [c for c in train_df.columns if train_df[c].std() == 0]
if constant_cols:
    print(f"\nDropping constant sensors (no variation in train): {constant_cols}")
    train_df = train_df.drop(columns=constant_cols)
    val_df = val_df.drop(columns=constant_cols)
    test_normal_df = test_normal_df.drop(columns=constant_cols)
print(f"Sensors used for the model: {len(train_df.columns)}")

# --- Step 4: normalize with StandardScaler (fit on train only) ---
from sklearn.preprocessing import StandardScaler
import joblib

print("\n--- Normalizing (StandardScaler, fit on train only) ---")

scaler = StandardScaler()

# FIT only on training data - this is what prevents data leakage
scaler.fit(train_df)

# TRANSFORM all splits using the same scaler
train_scaled = scaler.transform(train_df)
val_scaled   = scaler.transform(val_df)
test_normal_scaled = scaler.transform(test_normal_df)

# Quick check: after scaling, train should have mean ~0 and std ~1
# Check per sensor (not one number over the whole array)
means = train_scaled.mean(axis=0)
stds  = train_scaled.std(axis=0)
print(f"Train mean after scaling (should be ~0): min={means.min():.4f} max={means.max():.4f}")
print(f"Train std  after scaling (should be ~1): min={stds.min():.4f} max={stds.max():.4f}")

# Save the scaler for later use (on fault files, and at inference)
joblib.dump(scaler, "models/scaler.joblib")
print("Scaler saved to models/scaler.joblib")

# --- Step 5: save the processed splits for Week 3 ---
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

np.save("data/processed/train.npy", train_scaled)
np.save("data/processed/val.npy", val_scaled)
np.save("data/processed/test_normal.npy", test_normal_scaled)

# Row numbers in the original CSV, so later analysis can go back to raw values
np.save("data/processed/train_idx.npy", train_df.index.to_numpy())
np.save("data/processed/val_idx.npy", val_df.index.to_numpy())
np.save("data/processed/test_normal_idx.npy", test_normal_df.index.to_numpy())

print("\n--- Saved processed splits ---")
print(f"  train.npy:       {train_scaled.shape}")
print(f"  val.npy:         {val_scaled.shape}")
print(f"  test_normal.npy: {test_normal_scaled.shape}")
print("\nWeek 2 preprocessing complete!")