import json
import numpy as np
import pandas as pd
import joblib
import torch
from model import Autoencoder                 # defined in src/model.py
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

THRESHOLD_PERCENTILE = 99   # 1% of normal validation rows are allowed above it

# --- Step 1: load the trained model and the scaler ---
scaler = joblib.load("models/scaler.joblib")
sensors = list(scaler.feature_names_in_)      # the 14 sensors the model uses

model = Autoencoder(len(sensors))
model.load_state_dict(torch.load("models/autoencoder.pt"))
model.eval()


def reconstruction_errors(x):
    """Per-row error (mean over sensors) and per-sensor squared error."""
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        sq = ((model(t) - t) ** 2).numpy()
    return sq.mean(axis=1), sq


# --- Step 2: choose the threshold from normal validation data ---
val_err, _ = reconstruction_errors(np.load("data/processed/val.npy"))
threshold = float(np.percentile(val_err, THRESHOLD_PERCENTILE))

with open("models/threshold.json", "w") as f:
    json.dump({"threshold": threshold, "percentile": THRESHOLD_PERCENTILE}, f, indent=2)

print(f"Threshold = {THRESHOLD_PERCENTILE}th percentile of validation error = {threshold:.5f}")
print("Saved to models/threshold.json")

# --- Step 3: false alarms on normal test data ---
test_err, _ = reconstruction_errors(np.load("data/processed/test_normal.npy"))
false_alarm_rate = (test_err > threshold).mean() * 100
print(f"\nFalse alarms on normal test data: {false_alarm_rate:.2f}% of rows "
      f"(expected ~{100 - THRESHOLD_PERCENTILE}%)")

# --- Step 4: detection rate on every fault file ---
# Fault files are simulations of the same year as the fault-free file. To
# avoid rewarding the model for days it saw (in normal form) during training,
# only the TEST days are scored, i.e. the same days as test_normal above.
# Each file gets exactly the same preprocessing as the training data.
catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
fault_files = catalog[catalog["label"] == "fault"]

print("\n--- Detection per fault file (test days, steady state only) ---")
print(f"{'fault type':40s} {'severity':>8s} {'rows':>7s} {'detected':>9s}   top sensor")

results = []
for _, row in fault_files.iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}")
    keep, _, _, _ = steady_state_mask(df)
    mask = keep & (day_split(len(df)) == "test")

    x = scaler.transform(df.loc[mask, sensors])
    err, sq = reconstruction_errors(x)
    flagged = err > threshold
    detection = flagged.mean() * 100 if len(err) else float("nan")

    # Which sensor contributes most to the error in the flagged rows?
    top_sensor = sensors[sq[flagged].mean(axis=0).argmax()] if flagged.any() else "-"

    severity = row["severity"] if isinstance(row["severity"], str) else "-"
    print(f"{row['fault_type']:40s} {severity:>8s} {len(err):>7d} {detection:8.1f}%   {top_sensor}")
    results.append({
        "filename": row["filename"],
        "fault_type": row["fault_type"],
        "severity": severity,
        "rows": len(err),
        "detection_pct": round(detection, 2),
        "top_sensor": top_sensor,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("data/fault_results.csv", index=False)

# --- Step 5: summary per fault type ---
print("\n--- Summary per fault type (mean detection over severities) ---")
summary = results_df.groupby("fault_type")["detection_pct"].agg(["mean", "min", "max"])
print(summary.sort_values("mean", ascending=False).round(1).to_string())

print(f"\nFalse alarm rate (normal test data): {false_alarm_rate:.2f}%")
print(f"Mean detection rate (all fault files): {results_df['detection_pct'].mean():.1f}%")
print("\nResults saved to data/fault_results.csv")
