import pandas as pd

FILE = "data/LBNL_FDD_Data_Sets_Chiller_Plant/ChillerPlant.csv"

# Physics: wet-bulb temperature can never be higher than dry-bulb temperature.
# In the scaler, OA_TEMP_WB has a HIGHER mean (53.0 vs 47.4) and a HIGHER std
# (21.7 vs 19.1) than OA_TEMP. This script checks what OA_TEMP_WB really is.

df = pd.read_csv(FILE)
db = df["OA_TEMP"]
wb = df["OA_TEMP_WB"]

# --- Check 1: basic ranges ---
print("--- Ranges ---")
print(pd.DataFrame({"OA_TEMP": db.describe(), "OA_TEMP_WB": wb.describe()}))

# --- Check 2: how often is wet-bulb > dry-bulb? (should be 0%) ---
bad = (wb > db + 0.5).mean() * 100
print(f"\nRows where OA_TEMP_WB > OA_TEMP: {bad:.1f}%  (should be ~0%)")

# --- Check 3: are the two columns swapped? ---
bad_swapped = (db > wb + 0.5).mean() * 100
print(f"Rows where OA_TEMP > OA_TEMP_WB: {bad_swapped:.1f}%")
if bad > 50 and bad_swapped < 5:
    print("-> Looks SWAPPED: OA_TEMP_WB is probably the dry-bulb, and OA_TEMP the wet-bulb.")

# --- Check 4: is it relative humidity (0-100 %) instead of a temperature? ---
if wb.min() >= 0 and wb.max() <= 100 and db.corr(wb) < 0.5:
    print("-> Values fit 0-100 and barely follow OA_TEMP: may be RELATIVE HUMIDITY (%), not wet-bulb.")
print(f"Correlation OA_TEMP vs OA_TEMP_WB: {db.corr(wb):.3f}  (a real wet-bulb is usually > 0.9)")

# --- Check 5: other outdoor-air columns that could help ---
print("\nOther outdoor-air / humidity columns in the file:")
for c in df.columns:
    if c.startswith("OA_") or "HUM" in c.upper() or "RH" in c.upper():
        print(f"   {c:20s} mean={df[c].mean():9.3f}  min={df[c].min():9.3f}  max={df[c].max():9.3f}")

# --- Check 6: sample rows ---
print("\nSample rows (every ~10%):")
print(df[["OA_TEMP", "OA_TEMP_WB"]].iloc[:: max(1, len(df) // 10)])
