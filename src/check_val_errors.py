import numpy as np
import pandas as pd
import joblib
import torch
from model import Autoencoder   # defined in src/model.py

FILE = "data/LBNL_FDD_Data_Sets_Chiller_Plant/ChillerPlant.csv"
ROWS_PER_DAY = 1440
RUNNING_THRESHOLD = 5.0   # kW - same as prepare_data.py

# After training: val loss (0.062) was ~2x train loss (0.027), and the 99th
# percentile error on validation (0.99) was ~3x the one on test (0.35), while
# the medians were almost equal. So a small group of rows has very high error.
# This script finds out which rows those are, and why.

# --- Load model, scaler and data ---
scaler = joblib.load("models/scaler.joblib")
sensors = list(scaler.feature_names_in_)

model = Autoencoder(len(sensors))
model.load_state_dict(torch.load("models/autoencoder.pt"))
model.eval()

raw = pd.read_csv(FILE)
# Same fix as in prepare_data.py (the two columns are swapped in the file)
raw = raw.rename(columns={"OA_TEMP": "OA_TEMP_WB", "OA_TEMP_WB": "OA_TEMP"})

# How many minutes the chiller has been running in a row (0 = off/standby)
running = (raw["CHL_POW_1"] > RUNNING_THRESHOLD).to_numpy()
run_id = np.cumsum(~running)
minutes_running = pd.Series(running.astype(int)).groupby(run_id).cumsum().to_numpy()


def row_errors(split):
    """Per-row and per-sensor squared error for one split."""
    x = np.load(f"data/processed/{split}.npy")
    idx = np.load(f"data/processed/{split}_idx.npy")
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        sq = ((model(t) - t) ** 2).numpy()   # shape (rows, sensors)
    return idx, sq


for split in ["val", "test_normal", "train"]:
    idx, sq = row_errors(split)
    err = sq.mean(axis=1)
    top = err >= np.percentile(err, 99)       # worst 1% of rows

    print(f"\n==================== {split} ({len(err)} rows) ====================")
    print(f"Error: mean={err.mean():.5f}  median={np.median(err):.5f}  "
          f"99th pct={np.percentile(err, 99):.5f}")
    print(f"Share of the TOTAL error coming from the worst 1% of rows: "
          f"{err[top].sum() / err.sum() * 100:.0f}%")

    # Check 1: which sensors cause the error in the worst rows?
    print("\nWhich sensors cause the error in the worst 1% (share of error):")
    share = sq[top].sum(axis=0) / sq[top].sum() * 100
    for s, v in sorted(zip(sensors, share), key=lambda p: -p[1])[:5]:
        print(f"   {s:18s} {v:5.1f}%")

    # Check 2: are the worst rows right after the chiller starts?
    m = minutes_running[idx]
    print("\nMinutes since the chiller started:")
    print(f"   worst 1%:  median={np.median(m[top]):6.0f}   "
          f"in first 30 min: {(m[top] <= 30).mean() * 100:5.1f}%")
    print(f"   all rows:  median={np.median(m):6.0f}   "
          f"in first 30 min: {(m <= 30).mean() * 100:5.1f}%")

    # Check 3: low power, just above the threshold?
    p = raw["CHL_POW_1"].to_numpy()[idx]
    print("\nCHL_POW_1 (kW):")
    print(f"   worst 1%:  median={np.median(p[top]):6.1f}   below 20 kW: {(p[top] < 20).mean() * 100:5.1f}%")
    print(f"   all rows:  median={np.median(p):6.1f}   below 20 kW: {(p < 20).mean() * 100:5.1f}%")

    # Check 4: are the worst rows packed into a few days?
    days = pd.Series(idx[top] // ROWS_PER_DAY).value_counts()
    print(f"\nWorst rows are spread over {len(days)} days. Top 5 days (day of year: rows):")
    print("   " + ", ".join(f"{d}: {c}" for d, c in days.head(5).items()))
