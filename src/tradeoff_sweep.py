import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model import Autoencoder                 # defined in src/model.py
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Trade-off sweep, no retraining: how much detection can we gain by
#   - lowering the threshold (more false alarms), or
#   - using longer windows (the alarm comes later)?
# Same method as evaluate_faults.py (signed residual averaged per window,
# max-z score over sensors), only the threshold percentile and window length
# change. Thresholds always come from NORMAL validation data.
#
# Detection is reported two ways:
#   - all 23 fault files
#   - the 19 "detectable" files: without the 4 that check_fault_signatures.py
#     showed do not change any measured column beyond noise (fouling 095) or
#     only show in noisy pump signals (positive pressure bias)
#
# Run: python src/tradeoff_sweep.py              (full sweep + figure)
#      python src/tradeoff_sweep.py --plot-only  (redraw the figure from
#                                                 data/tradeoff_results.csv)

PERCENTILES = [95, 97, 98, 99, 99.5]
WINDOWS = [60, 120, 180, 360, 1440]           # minutes
MIN_ROWS_PER_WINDOW = 30
NOT_DETECTABLE = [
    "ChillerPlant_chiller_fouling_095.csv",
    "ChillerPlant_coolingtower_fouling_095.csv",
    "ChillerPlant_secondary_chilled_water_pressure_bias_010.csv",
    "ChillerPlant_secondary_chilled_water_pressure_bias_020.csv",
]

if "--plot-only" in sys.argv:
    res = pd.read_csv("data/tradeoff_results.csv")
else:
    # --- Step 1: model, scaler, data (residuals computed once) ---
    scaler = joblib.load("models/scaler.joblib")
    sensors = list(scaler.feature_names_in_)
    model = Autoencoder(len(sensors))
    model.load_state_dict(torch.load("models/autoencoder.pt"))
    model.eval()


    def residuals(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            return (model(t) - t).numpy()


    normal = {
        "val":  (np.load("data/processed/val_idx.npy"), residuals(np.load("data/processed/val.npy"))),
        "test": (np.load("data/processed/test_normal_idx.npy"),
                 residuals(np.load("data/processed/test_normal.npy"))),
    }

    print("Loading fault files...")
    catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
    faults = []
    for _, row in catalog[catalog["label"] == "fault"].iterrows():
        df = load_plant_file(f"{DATA_DIR}/{row['filename']}")
        keep, _, _, _ = steady_state_mask(df)
        split = day_split(len(df))
        parts = {}
        for part in ["val", "test"]:
            mask = keep & (split == part)
            parts[part] = (np.flatnonzero(mask), residuals(scaler.transform(df.loc[mask, sensors])))
        faults.append({"file": row["filename"], **parts})
    detectable = np.array([f["file"] not in NOT_DETECTABLE for f in faults])
    print(f"{len(faults)} fault files, {detectable.sum()} counted as detectable.\n")


    def window_means(idx, resid, minutes):
        df = pd.DataFrame(resid)
        window = idx // minutes
        counts = df.groupby(window).size()
        return df.groupby(window).mean()[counts >= MIN_ROWS_PER_WINDOW].to_numpy()


    # --- Step 2: sweep ---
    rows = []
    for minutes in WINDOWS:
        val_win = window_means(*normal["val"], minutes)
        test_win = window_means(*normal["test"], minutes)
        center, spread = val_win.mean(axis=0), val_win.std(axis=0)
        score = lambda w: np.abs((w - center) / spread).max(axis=1)
        fault_scores = {p: [score(window_means(*f[p], minutes)) for f in faults] for p in ["val", "test"]}

        for pct in PERCENTILES:
            threshold = np.percentile(score(val_win), pct)
            det = {p: np.array([(s > threshold).mean() * 100 for s in fault_scores[p]]) for p in ["val", "test"]}
            rows.append({
                "window_min": minutes,
                "percentile": pct,
                "normal_test_windows": len(test_win),
                "false_alarm_pct": round((score(test_win) > threshold).mean() * 100, 2),
                "detection_all_test_pct": round(det["test"].mean(), 1),
                "detection_detectable_test_pct": round(det["test"][detectable].mean(), 1),
                "detection_all_val_pct": round(det["val"].mean(), 1),
            })

    res = pd.DataFrame(rows)
    os.makedirs("reports/figures", exist_ok=True)
    res.to_csv("data/tradeoff_results.csv", index=False)

    print("Detection on TEST days. 'all' = 23 files, 'detectable' = 19 files.")
    print("False alarms = % of normal test windows flagged.\n")
    print(res.rename(columns={
        "window_min": "window", "percentile": "pct", "normal_test_windows": "n windows",
        "false_alarm_pct": "false al. %", "detection_all_test_pct": "det. all %",
        "detection_detectable_test_pct": "det. detectable %", "detection_all_val_pct": "det. all (val) %",
    }).to_string(index=False))

# --- Step 3: figure - detection vs false alarms, one line per window ---
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK_2, GRID, SURFACE = "#52514e", "#e4e3df", "#fcfcfb"
colors = dict(zip(WINDOWS, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
names = {60: "1 h", 120: "2 h", 180: "3 h", 360: "6 h", 1440: "1 day"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.labelcolor": INK_2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, col, title in [(axes[0], "detection_all_test_pct", "All 23 fault files"),
                       (axes[1], "detection_detectable_test_pct", "19 detectable fault files")]:
    for minutes in WINDOWS:
        # Connect points in threshold order (strict -> loose)
        d = res[res["window_min"] == minutes].sort_values("percentile", ascending=False)
        ax.plot(d["false_alarm_pct"], d[col], color=colors[minutes], lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=names[minutes])
    # Mark the setting used in the final model: 1-hour window, 99th percentile
    cur = res[(res["window_min"] == 60) & (res["percentile"] == 99)].iloc[0]
    ax.plot(cur["false_alarm_pct"], cur[col], marker="o", ms=13, mfc="none",
            mec="#0b0b0b", mew=1.5, zorder=5)
    ax.annotate(f"current setting\n(1 h, 99th pct): {cur[col]:.1f}%",
                xy=(cur["false_alarm_pct"], cur[col]), xytext=(70, -85),
                textcoords="offset points", fontsize=8, color="#0b0b0b",
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.axhline(85, color=INK_2, lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 85.5, "85% target", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.set_title(title)
    ax.set_xlabel("False alarms (% of normal test windows)")
axes[0].set_ylabel("Mean detection, test days (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(WINDOWS), title="Window length",
           fontsize=9, bbox_to_anchor=(0.5, 0))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig("reports/figures/7_tradeoff.png", dpi=200)

print("\nSaved data/tradeoff_results.csv and reports/figures/7_tradeoff.png")
