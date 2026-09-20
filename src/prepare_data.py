import pandas as pd

DATA_DIR = "data/LBNL_FDD_Data_Sets_Chiller_Plant"
FAULT_FREE_FILE = f"{DATA_DIR}/ChillerPlant.csv"

SELECTED_SENSORS = [
    "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1",
    "OA_TEMP", "OA_TEMP_WB", "CHL_POW_1", "CT_POW_1",
    "CHL_CW_FLOW_1", "CWL_SEC_DP", "CHL_STA_1",
]

# --- Step 1: confirm completeness of the fault-free file ---
print("Loading fault-free file...")
df = pd.read_csv(FAULT_FREE_FILE, usecols=SELECTED_SENSORS)

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

# --- Step 3: chronological split of the fault-free data ---
print("\n--- Splitting fault-free data (chronological 70/15/15) ---")

# df already holds the fault-free data (loaded in Step 1), in time order
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)   # 70% + 15%

# Chronological slices - NO shuffling (order = time)
train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_normal_df = df.iloc[val_end:]

print(f"Total fault-free rows: {n}")
print(f"  Train:       {len(train_df):>7} rows ({len(train_df)/n*100:.0f}%)")
print(f"  Validation:  {len(val_df):>7} rows ({len(val_df)/n*100:.0f}%)")
print(f"  Test-normal: {len(test_normal_df):>7} rows ({len(test_normal_df)/n*100:.0f}%)")

# Sanity check: the three parts must add up to the whole, with no overlap
assert len(train_df) + len(val_df) + len(test_normal_df) == n
print("OK - splits add up correctly, no overlap.")

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
print(f"Train mean after scaling (should be ~0): {train_scaled.mean():.4f}")
print(f"Train std  after scaling (should be ~1): {train_scaled.std():.4f}")

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

print("\n--- Saved processed splits ---")
print(f"  train.npy:       {train_scaled.shape}")
print(f"  val.npy:         {val_scaled.shape}")
print(f"  test_normal.npy: {test_normal_scaled.shape}")
print("\nWeek 2 preprocessing complete!")