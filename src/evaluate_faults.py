import json
import numpy as np
import pandas as pd
import joblib
import torch
from model import Autoencoder                 # defined in src/model.py
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

THRESHOLD_PERCENTILE = 99   # 1% of normal validation rows/windows are allowed above it
WINDOW_MINUTES = 60         # window-level scores use 1-hour windows
MIN_ROWS_PER_WINDOW = 30    # skip windows with too few steady-state rows

# --- Step 1: load the trained model and the scaler ---
scaler = joblib.load("models/scaler.joblib")
sensors = list(scaler.feature_names_in_)      # the 14 sensors the model uses

model = Autoencoder(len(sensors))
model.load_state_dict(torch.load("models/autoencoder.pt"))
model.eval()


def residuals(x):
    """Signed reconstruction residual (reconstruction - input), per row and sensor."""
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        return (model(t) - t).numpy()


def window_means(idx, resid):
    """Mean signed residual per sensor, for each 1-hour window of rows.

    idx = row numbers in the original file, used to group rows into hours.
    Small, persistent faults (e.g. a 1-2 degF sensor bias) are lost in the
    noise of single rows, but they push the residual of some sensor in the
    same direction for the whole hour, while noise averages out.
    """
    df = pd.DataFrame(resid, columns=sensors)
    window = idx // WINDOW_MINUTES
    counts = df.groupby(window).size()
    means = df.groupby(window).mean()
    return means[counts >= MIN_ROWS_PER_WINDOW].to_numpy()


# --- Step 2: thresholds from normal validation data ---
val_x = np.load("data/processed/val.npy")
val_resid = residuals(val_x)

# (a) Row score: mean squared residual of one row
val_row_err = (val_resid ** 2).mean(axis=1)
row_threshold = float(np.percentile(val_row_err, THRESHOLD_PERCENTILE))

# (b) Window score: how far the most-shifted sensor's hourly mean residual is
#     from normal, in units of its normal spread (a z-score)
val_win = window_means(np.load("data/processed/val_idx.npy"), val_resid)
win_center = val_win.mean(axis=0)
win_spread = val_win.std(axis=0)


def window_scores(win):
    """Per window: largest |z| over sensors, and which sensor it was."""
    z = np.abs((win - win_center) / win_spread)
    return z.max(axis=1), z.argmax(axis=1)


val_win_score, _ = window_scores(val_win)
win_threshold = float(np.percentile(val_win_score, THRESHOLD_PERCENTILE))

with open("models/threshold.json", "w") as f:
    json.dump({
        "percentile": THRESHOLD_PERCENTILE,
        "row_threshold": row_threshold,
        "window_minutes": WINDOW_MINUTES,
        "min_rows_per_window": MIN_ROWS_PER_WINDOW,
        "window_threshold": win_threshold,
        "window_center": win_center.tolist(),
        "window_spread": win_spread.tolist(),
        "sensors": sensors,
    }, f, indent=2)

print(f"Row threshold    (99th pct of validation rows):    {row_threshold:.5f}")
print(f"Window threshold (99th pct of validation windows): {win_threshold:.3f}  "
      f"({len(val_win)} one-hour windows)")
print("Saved to models/threshold.json")


def evaluate(idx, x):
    """Row detection %, window detection %, number of windows, top sensor."""
    resid = residuals(x)
    row_flag = (resid ** 2).mean(axis=1) > row_threshold
    score, top = window_scores(window_means(idx, resid))
    win_flag = score > win_threshold
    # The sensor that is most often the most-shifted one in flagged windows
    top_sensor = sensors[np.bincount(top[win_flag]).argmax()] if win_flag.any() else "-"
    return row_flag.mean() * 100, win_flag.mean() * 100, len(score), top_sensor


# --- Step 3: false alarms on normal test data ---
fa_row, fa_win, n_win, _ = evaluate(np.load("data/processed/test_normal_idx.npy"),
                                    np.load("data/processed/test_normal.npy"))
print(f"\nFalse alarms on normal test data (expected ~{100 - THRESHOLD_PERCENTILE}%):")
print(f"   rows:    {fa_row:.2f}%")
print(f"   windows: {fa_win:.2f}%  ({n_win} one-hour windows)")

# --- Step 4: detection on every fault file ---
# Fault files are simulations of the same year as the fault-free file. To
# avoid rewarding the model for days it saw (in normal form) during training,
# only the TEST days are scored, i.e. the same days as test_normal above.
# Each file gets exactly the same preprocessing as the training data.
catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
fault_files = catalog[catalog["label"] == "fault"]

print("\n--- Detection per fault file (test days, steady state only) ---")
print(f"{'fault type':40s} {'severity':>8s} {'rows':>7s} {'row det.':>9s} "
      f"{'hour det.':>10s}   top sensor (hours)")

results = []
for _, row in fault_files.iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}")
    keep, _, _, _ = steady_state_mask(df)
    mask = keep & (day_split(len(df)) == "test")

    idx = np.flatnonzero(mask)
    x = scaler.transform(df.loc[mask, sensors])
    row_det, win_det, _, top_sensor = evaluate(idx, x)

    severity = row["severity"] if isinstance(row["severity"], str) else "-"
    print(f"{row['fault_type']:40s} {severity:>8s} {len(idx):>7d} {row_det:8.1f}% "
          f"{win_det:9.1f}%   {top_sensor}")
    results.append({
        "filename": row["filename"],
        "fault_type": row["fault_type"],
        "severity": severity,
        "rows": len(idx),
        "row_detection_pct": round(row_det, 2),
        "window_detection_pct": round(win_det, 2),
        "top_sensor": top_sensor,
    })

results_df = pd.DataFrame(results)
results_df.to_csv("data/fault_results.csv", index=False)

# --- Step 5: summary per fault type ---
print("\n--- Summary per fault type (mean over severities) ---")
summary = results_df.groupby("fault_type")[["row_detection_pct", "window_detection_pct"]].mean()
summary.columns = ["row det. %", "hour det. %"]
print(summary.sort_values("hour det. %", ascending=False).round(1).to_string())

print(f"\nFalse alarm rate:  rows {fa_row:.2f}%   windows {fa_win:.2f}%")
print(f"Mean detection:    rows {results_df['row_detection_pct'].mean():.1f}%   "
      f"windows {results_df['window_detection_pct'].mean():.1f}%")
print("\nResults saved to data/fault_results.csv")
