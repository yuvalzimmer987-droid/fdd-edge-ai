import pandas as pd

FILE = "data/LBNL_FDD_Data_Sets_Chiller_Plant/ChillerPlant.csv"

# In the scaler, CHL_POW_1 has mean=35 but std=57. That usually means many
# rows at ~0 (chiller off) mixed with rows at high power (chiller running).
# This script checks how much of the data is "off", and where it sits in time.

OFF_THRESHOLD = 1.0   # kW - below this we treat the chiller as off

df = pd.read_csv(FILE)
# Same fix as in prepare_data.py (the two columns are swapped in the file)
df = df.rename(columns={"OA_TEMP": "OA_TEMP_WB", "OA_TEMP_WB": "OA_TEMP"})
pow_ = df["CHL_POW_1"]

# --- Check 1: distribution of chiller power ---
print("--- CHL_POW_1 distribution ---")
print(pow_.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
print(f"Rows with power == 0:          {(pow_ == 0).mean() * 100:.1f}%")
print(f"Rows with power < {OFF_THRESHOLD} (off):   {(pow_ < OFF_THRESHOLD).mean() * 100:.1f}%")

# --- Check 2: does CHL_STA_1 (status) agree with the power? ---
print("\n--- CHL_STA_1 (status) vs power ---")
print(df.groupby("CHL_STA_1")["CHL_POW_1"].agg(["count", "mean", "min", "max"]))

# --- Check 3: how is "off" spread over the chronological splits? ---
n = len(df)
off = pow_ < OFF_THRESHOLD
splits = {
    "Train (0-70%)":        off.iloc[: int(n * 0.70)],
    "Validation (70-85%)":  off.iloc[int(n * 0.70): int(n * 0.85)],
    "Test-normal (85-100%)": off.iloc[int(n * 0.85):],
}
print("\n--- Share of 'off' rows in each split ---")
for name, s in splits.items():
    print(f"  {name:24s} {s.mean() * 100:5.1f}% off")

# --- Check 4: how "off" is spread over the year (10 equal time chunks) ---
print("\n--- Share of 'off' rows per 10% of time, with outdoor temp ---")
chunk = n // 10
for i in range(10):
    part = df.iloc[i * chunk:(i + 1) * chunk]
    print(f"  rows {i*10:3d}-{(i+1)*10:3d}%:  off={(part['CHL_POW_1'] < OFF_THRESHOLD).mean()*100:5.1f}%"
          f"   OA_TEMP mean={part['OA_TEMP'].mean():5.1f}")

# --- Check 5: what do the other sensors look like when off vs on? ---
print("\n--- Sensor means: off vs on ---")
cols = ["CHL_POW_1", "CT_POW_1", "CHL_SW_TEMP_1", "CHL_RW_TEMP_1",
        "CHL_CW_FLOW_1", "CT_SW_TEMP_1", "OA_TEMP"]
print(df.groupby(off.map({True: "off", False: "on"}))[cols].mean().T)
