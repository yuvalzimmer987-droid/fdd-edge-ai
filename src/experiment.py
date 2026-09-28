import sys
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.covariance import LedoitWolf
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Experiment: which model size, window length and window score detect the
# most faults at the same false-alarm level?
#
# - Model size: (hidden units, bottleneck units). (8, 4) is the current model.
#   A bigger model reconstructs normal data better, so faults stand out more,
#   but too big a model also learns to reconstruct the faults.
# - Window length: longer windows average out more noise (better for small,
#   persistent faults) but raise the alarm later.
# - Window score:
#     max_z       - current method: the most-shifted sensor, on its own.
#     mahalanobis - uses all 14 sensors together, including how their
#                   residuals normally move together (covariance).
#
# Fair selection: every threshold comes from NORMAL validation data only.
# The best setting is chosen by detection on the VALIDATION days of the fault
# files; the TEST days are only used to report the final numbers.
#
# Run: python src/experiment.py           (full run, ~10-20 min)
#      python src/experiment.py --quick   (3 epochs per model, just to test the code)

CONFIGS = [(8, 4), (16, 4), (16, 6), (32, 6), (32, 8)]   # (hidden, bottleneck)
WINDOWS = [60, 180, 1440]                                # minutes
SCORES = ["max_z", "mahalanobis"]
PERCENTILE = 99
MIN_ROWS_PER_WINDOW = 30
MAX_EPOCHS = 3 if "--quick" in sys.argv else 200
PATIENCE = 10


class Autoencoder(nn.Module):
    """Same design as src/model.py, with the layer sizes as parameters."""
    def __init__(self, n_features, hidden, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# --- Step 1: load all data once ---
print("Loading data...")
scaler = joblib.load("models/scaler.joblib")
sensors = list(scaler.feature_names_in_)

train = torch.tensor(np.load("data/processed/train.npy"), dtype=torch.float32)
normal = {
    "val":  (np.load("data/processed/val_idx.npy"), np.load("data/processed/val.npy")),
    "test": (np.load("data/processed/test_normal_idx.npy"), np.load("data/processed/test_normal.npy")),
}

catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
faults = []
for _, row in catalog[catalog["label"] == "fault"].iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}")
    keep, _, _, _ = steady_state_mask(df)
    split = day_split(len(df))
    parts = {}
    for part in ["val", "test"]:
        mask = keep & (split == part)
        parts[part] = (np.flatnonzero(mask), scaler.transform(df.loc[mask, sensors]))
    severity = row["severity"] if isinstance(row["severity"], str) else "-"
    faults.append({"name": f"{row['fault_type']} {severity}", **parts})
print(f"Loaded {len(faults)} fault files.")


# --- Helpers ---
def train_model(hidden, bottleneck):
    """Train with early stopping (same settings as train_model.py)."""
    torch.manual_seed(42)
    model = Autoencoder(train.shape[1], hidden, bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(train, train), batch_size=256, shuffle=True)
    val = torch.tensor(normal["val"][1], dtype=torch.float32)

    best_loss, best_state, waited = float("inf"), None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val), val).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, waited = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def residuals(model, x):
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        return (model(t) - t).numpy()


def window_means(idx, resid, minutes):
    """Mean signed residual per sensor for each window of `minutes` minutes."""
    df = pd.DataFrame(resid)
    window = idx // minutes
    counts = df.groupby(window).size()
    return df.groupby(window).mean()[counts >= MIN_ROWS_PER_WINDOW].to_numpy()


def make_scorer(normal_windows, kind):
    """Fit a window score on normal validation windows."""
    center = normal_windows.mean(axis=0)
    if kind == "max_z":
        spread = normal_windows.std(axis=0)
        return lambda w: np.abs((w - center) / spread).max(axis=1)
    precision = LedoitWolf().fit(normal_windows).precision_   # shrunk inverse covariance
    return lambda w: np.sqrt(np.einsum("ij,jk,ik->i", w - center, precision, w - center))


# --- Step 2: run every combination ---
results = []
per_fault = {}
for hidden, bottleneck in CONFIGS:
    start = time.time()
    model, val_loss = train_model(hidden, bottleneck)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel hidden={hidden:2d} bottleneck={bottleneck}: {n_params} parameters, "
          f"val loss {val_loss:.5f}, trained in {time.time() - start:.0f} s")

    # Residuals are computed once per model and reused for every window/score
    res_normal = {p: (idx, residuals(model, x)) for p, (idx, x) in normal.items()}
    res_faults = [{p: (f[p][0], residuals(model, f[p][1])) for p in ["val", "test"]} for f in faults]

    for minutes in WINDOWS:
        win_normal = {p: window_means(idx, r, minutes) for p, (idx, r) in res_normal.items()}
        win_faults = [{p: window_means(*f[p], minutes) for p in ["val", "test"]} for f in res_faults]

        for kind in SCORES:
            score = make_scorer(win_normal["val"], kind)
            threshold = np.percentile(score(win_normal["val"]), PERCENTILE)
            false_alarm = (score(win_normal["test"]) > threshold).mean() * 100
            det = {p: [(score(w[p]) > threshold).mean() * 100 if len(w[p]) else np.nan
                       for w in win_faults] for p in ["val", "test"]}

            key = (hidden, bottleneck, minutes, kind)
            per_fault[key] = det["test"]
            results.append({
                "hidden": hidden, "bottleneck": bottleneck, "params": n_params,
                "val_loss": round(val_loss, 5), "window_min": minutes, "score": kind,
                "normal_test_windows": len(win_normal["test"]),
                "false_alarm_pct": round(false_alarm, 2),
                "detection_val_days_pct": round(np.nanmean(det["val"]), 1),
                "detection_test_days_pct": round(np.nanmean(det["test"]), 1),
            })
            print(f"   window {minutes:4d} min  {kind:11s}  false alarms {false_alarm:5.2f}%  "
                  f"detection (val days) {np.nanmean(det['val']):5.1f}%  "
                  f"(test days) {np.nanmean(det['test']):5.1f}%")

results_df = pd.DataFrame(results)
results_df.to_csv("data/experiment_results.csv", index=False)

# --- Step 3: the best setting for each window length ---
# Chosen on validation days; reported on test days. Different window lengths
# are not directly comparable (a longer window raises the alarm later), so
# the best setting is shown for each one.
print("\n\n=== Best setting per window length (chosen on validation days) ===")
cols = ["hidden", "bottleneck", "params", "score", "normal_test_windows",
        "false_alarm_pct", "detection_test_days_pct"]
best_rows = []
for minutes in WINDOWS:
    sub = results_df[results_df["window_min"] == minutes]
    best = sub.loc[sub["detection_val_days_pct"].idxmax()]
    best_rows.append(best)
    print(f"\nWindow {minutes} min:")
    print(best[cols].to_string())

# --- Step 4: per-fault comparison, current setup vs best 1-hour setting ---
current = (8, 4, 60, "max_z")
b = best_rows[0]
best_key = (int(b["hidden"]), int(b["bottleneck"]), 60, b["score"])
print(f"\n=== Per fault (test days): current {current} vs best 1-hour {best_key} ===")
for f, old, new in zip(faults, per_fault[current], per_fault[best_key]):
    print(f"   {f['name']:45s} {old:6.1f}%  ->  {new:6.1f}%")

print("\nAll results saved to data/experiment_results.csv")
