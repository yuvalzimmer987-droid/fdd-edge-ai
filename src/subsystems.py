import sys
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# One detector per subsystem, as suggested by the supervisor: the chiller,
# the cooling tower and the chilled-water pumping loop are separate systems,
# so each gets its own small autoencoder, its own hourly score and its own
# threshold. An hour is a fault if ANY subsystem raises an alarm, and the
# alarm also says WHICH subsystem.
#
# The dataset has no pump inlet/outlet pressure and no AHU pressure. The
# pumping detector uses what exists: the secondary-loop differential pressure
# and its setpoint, pump speeds and powers, loop flows, temperatures and load.
#
# Fair evaluation, same as before: thresholds from normal validation days,
# decisions compared on validation days, final numbers on test days.
#
# Run: python src/subsystems.py              (chiller model includes delta T)
#      python src/subsystems.py --no-delta   (same, without delta T)
#      python src/subsystems.py --hybrid     (the existing single model for
#                                             chiller + tower, plus a separate
#                                             pump detector)
#      add --sweep to also try false-alarm budgets of 1.1, 2, 3 and 5%

USE_DELTA_T = "--no-delta" not in sys.argv
HYBRID = "--hybrid" in sys.argv

SUBSYSTEMS = {
    "chiller": [
        "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
        "CHL_POW_1", "CHL_CW_FLOW_1", "CHL_CD_FLOW_1", "CHL_COMP_SPD_CTRL_1",
        "CWL_PRI_SW_TEMPSPT", "CWL_SEC_LOAD",
    ],
    "cooling_tower": [
        "CT_SW_TEMP_1", "CT_RW_TEMP_1", "CT_SW_TEMPSPT", "CT_FAN_SPD_1", "CT_POW_1",
        "CT_FLOW_1", "CDWL_SW_TEMP", "CDWL_RW_TEMP", "TWV_CTRL", "OA_TEMP", "OA_TEMP_WB",
    ],
    "pumps": [
        "CWL_SEC_DP", "CWL_SEC_DPSPT", "CWL_SEC_PM_SPD_1", "CWL_SEC_PM_SPD_2",
        "CWL_SEC_PM_POW_1", "CWL_SEC_PM_POW_2", "CWL_SEC_CW_FLOW", "CWL_PRI_CW_FLOW",
        "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CWL_SEC_LOAD",
    ],
}
# Hybrid: the per-subsystem run showed that chiller and cooling-tower faults
# are detected better by the single model (the two systems interact through
# the condenser loop, which a joint model sees), while the pumping loop
# benefits from its own detector. So keep the single model as "main" and add
# only the pump detector.
MAIN_SCALER = joblib.load("models/scaler.joblib") if HYBRID else None
MAIN_COLUMNS = list(MAIN_SCALER.feature_names_in_) if HYBRID else []
if HYBRID:
    SUBSYSTEMS = {"pumps": SUBSYSTEMS["pumps"]}

# CHL_POW_1 is always loaded: the steady-state mask needs it
ALL_COLUMNS = sorted({c for cols in SUBSYSTEMS.values() for c in cols}
                     | set(MAIN_COLUMNS) | {"CHL_POW_1"})

# False-alarm budget on normal validation hours, all detectors together. The
# single model's threshold (99th percentile of 289 validation hours) flags
# 3 of them, 1.04%; a 1.0% budget allows only 2 hours, which pushed all
# thresholds to the maximum normal hour in the first run and made the
# comparison unfair. 1.1% gives the detectors together the same 3 hours.
TARGET_FALSE_ALARM = 1.1
WINDOW_MINUTES = 60
MIN_ROWS_PER_WINDOW = 30
MAX_EPOCHS = 200
PATIENCE = 10
NOT_DETECTABLE = [           # same 4 files as in tradeoff_sweep.py
    "ChillerPlant_chiller_fouling_095.csv",
    "ChillerPlant_coolingtower_fouling_095.csv",
    "ChillerPlant_secondary_chilled_water_pressure_bias_010.csv",
    "ChillerPlant_secondary_chilled_water_pressure_bias_020.csv",
]


def subsystem_frame(df, name):
    """The columns of one subsystem (plus chiller delta T if enabled)."""
    if name == "main":
        return df[MAIN_COLUMNS].copy()
    x = df[SUBSYSTEMS[name]].copy()
    if name == "chiller" and USE_DELTA_T:
        # Chilled-water delta T: return minus supply temperature of the chiller
        x["CHL_DELTA_T"] = df["CHL_RW_TEMP_1"] - df["CHL_SW_TEMP_1"]
    return x


class Autoencoder(nn.Module):
    def __init__(self, n_features, hidden, bottleneck):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(train, val):
    """Same training as train_model.py: Adam, early stopping on val loss."""
    torch.manual_seed(42)
    n = train.shape[1]
    model = Autoencoder(n, hidden=16, bottleneck=max(2, n // 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_t = torch.tensor(train, dtype=torch.float32)
    val_t = torch.tensor(val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_t, train_t), batch_size=256, shuffle=True)

    best_loss, best_state, waited = float("inf"), None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_t), val_t).item()
        if val_loss < best_loss - 1e-5:
            best_loss, best_state, waited = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_loss


def window_means(idx, resid):
    """Mean signed residual per column for each 1-hour window (indexed by window id)."""
    df = pd.DataFrame(resid)
    window = idx // WINDOW_MINUTES
    counts = df.groupby(window).size()
    return df.groupby(window).mean()[counts >= MIN_ROWS_PER_WINDOW]


# --- Step 1: fault-free data, split and steady-state mask (as prepare_data.py) ---
print(f"Delta T in chiller model: {'yes' if USE_DELTA_T else 'no'}")
print("Loading fault-free file...")
normal = load_plant_file(f"{DATA_DIR}/ChillerPlant.csv", ALL_COLUMNS)
keep, _, _, _ = steady_state_mask(normal)
split = day_split(len(normal))
rows = {p: np.flatnonzero(keep & (split == p)) for p in ["train", "val", "test"]}

# --- Step 2: one model per subsystem ---
models = {}
for name in SUBSYSTEMS:
    frame = subsystem_frame(normal, name)
    # Drop columns that never change in training (e.g. a fixed setpoint)
    frame = frame.loc[:, frame.iloc[rows["train"]].std() > 0]
    scaler = StandardScaler().fit(frame.iloc[rows["train"]])
    x = {p: scaler.transform(frame.iloc[r]).astype(np.float32) for p, r in rows.items()}
    model, val_loss = train_autoencoder(x["train"], x["val"])

    def resid_of(arr, model=model):
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32)
            return (model(t) - t).numpy()

    val_win = window_means(rows["val"], resid_of(x["val"]))
    center, spread = val_win.mean().to_numpy(), val_win.std().to_numpy()
    models[name] = dict(columns=list(frame.columns), scaler=scaler, model=model,
                        resid_of=resid_of, center=center, spread=spread)
    print(f"  {name:13s} {len(frame.columns):2d} inputs  val loss {val_loss:.5f}")

    torch.save(model.state_dict(), f"models/subsys_{name}.pt")
    joblib.dump(scaler, f"models/subsys_{name}_scaler.joblib")

if HYBRID:
    # The existing single model, trained by train_model.py
    from model import Autoencoder as MainAutoencoder
    main_model = MainAutoencoder(len(MAIN_COLUMNS))
    main_model.load_state_dict(torch.load("models/autoencoder.pt"))
    main_model.eval()

    def main_resid_of(arr):
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32)
            return (main_model(t) - t).numpy()

    x_val = MAIN_SCALER.transform(normal[MAIN_COLUMNS].iloc[rows["val"]]).astype(np.float32)
    val_win = window_means(rows["val"], main_resid_of(x_val))
    models = {"main": dict(columns=MAIN_COLUMNS, scaler=MAIN_SCALER, model=main_model,
                           resid_of=main_resid_of, center=val_win.mean().to_numpy(),
                           spread=val_win.std().to_numpy()), **models}
    print(f"  {'main':13s} {len(MAIN_COLUMNS):2d} inputs  (existing single model)")


def subsystem_scores(df_rows, idx):
    """Hourly max-z score per subsystem, one column each, aligned on window id."""
    out = {}
    for name, m in models.items():
        x = m["scaler"].transform(subsystem_frame(df_rows, name)[m["columns"]]).astype(np.float32)
        win = window_means(idx, m["resid_of"](x))
        out[name] = (np.abs((win.to_numpy() - m["center"]) / m["spread"])).max(axis=1)
        out[name] = pd.Series(out[name], index=win.index)
    return pd.DataFrame(out)


# --- Step 3: thresholds - one per subsystem, set together so that all three
# combined raise an alarm in about 1% of normal validation hours ---
def find_thresholds(val_scores, budget):
    """Lowest common percentile whose thresholds flag <= budget % of normal val hours."""
    for pct in np.append(np.arange(95.0, 100.0, 0.01), 100.0):   # 100 = highest normal hour
        thresholds = val_scores.quantile(min(pct / 100, 1.0))
        any_rate = (val_scores > thresholds).any(axis=1).mean() * 100
        if any_rate <= budget:
            return thresholds, pct, any_rate


val_scores = subsystem_scores(normal.iloc[rows["val"]], rows["val"])
thresholds, pct, any_rate = find_thresholds(val_scores, TARGET_FALSE_ALARM)
print(f"\nThresholds at the {pct:.2f}th percentile of each subsystem "
      f"-> {any_rate:.2f}% of normal validation hours flagged")

test_scores = subsystem_scores(normal.iloc[rows["test"]], rows["test"])
fa_any = (test_scores > thresholds).any(axis=1).mean() * 100
fa_each = ((test_scores > thresholds).mean() * 100).round(2).to_dict()
print(f"False alarms on normal TEST hours: {fa_any:.2f}%  (per subsystem: {fa_each})")

with open("models/subsys_thresholds.json", "w") as f:
    json.dump({name: {"columns": m["columns"], "threshold": float(thresholds[name]),
                      "center": m["center"].tolist(), "spread": m["spread"].tolist()}
               for name, m in models.items()} | {"percentile": float(pct),
                                                 "delta_t": USE_DELTA_T}, f, indent=2)

# --- Step 4: every fault file, scored on validation and test days ---
catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
names = list(models)
print(f"\n{'fault':45s} {'any':>6s} " + " ".join(f"{n[:8]:>8s}" for n in names) + "   main subsystem")
results = []
fault_scores = []      # kept for the --sweep below
for _, row in catalog[catalog["label"] == "fault"].iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}", ALL_COLUMNS)
    f_keep, _, _, _ = steady_state_mask(df)
    f_split = day_split(len(df))
    det, scores = {}, {"file": row["filename"]}
    for part in ["val", "test"]:
        idx = np.flatnonzero(f_keep & (f_split == part))
        scores[part] = subsystem_scores(df.iloc[idx], idx)
        flags = scores[part] > thresholds
        det[part] = (flags.any(axis=1).mean() * 100, (flags.mean() * 100).to_dict())
    fault_scores.append(scores)
    any_test, each_test = det["test"]
    main = max(each_test, key=each_test.get) if any_test > 2 else "-"
    severity = row["severity"] if isinstance(row["severity"], str) else "-"
    name = f"{row['fault_type']} {severity}"
    print(f"{name:45s} {any_test:5.1f}% " + " ".join(f"{each_test[n]:7.1f}%" for n in names)
          + f"   {main}")
    results.append({"filename": row["filename"], "fault": name,
                    "detection_test_pct": round(any_test, 2),
                    "detection_val_pct": round(det["val"][0], 2),
                    **{f"{k}_test_pct": round(v, 2) for k, v in each_test.items()},
                    "main_subsystem": main})

res = pd.DataFrame(results)
res.to_csv("data/subsystem_results.csv", index=False)
detectable = ~res["filename"].isin(NOT_DETECTABLE)

print("\n=== Summary (compare with the single model: 72.8% test, 72.5% val, 0.34% false alarms) ===")
print(f"False alarms (test):                  {fa_any:.2f}%")
print(f"Mean detection, all 23 (test):        {res['detection_test_pct'].mean():.1f}%")
print(f"Mean detection, 19 detectable (test): {res.loc[detectable, 'detection_test_pct'].mean():.1f}%")
print(f"Mean detection, all 23 (VALIDATION):  {res['detection_val_pct'].mean():.1f}%   <- use this to decide")
print("\nSaved models/subsys_*.pt, models/subsys_thresholds.json, data/subsystem_results.csv")

# --- Step 5 (--sweep): same detectors, different false-alarm budgets ---
# How much detection do we gain if we allow more false alarms?
if "--sweep" in sys.argv:
    INVISIBLE = ["ChillerPlant_chiller_fouling_095.csv", "ChillerPlant_coolingtower_fouling_095.csv"]
    visible = np.array([f["file"] not in INVISIBLE for f in fault_scores])
    rows_out = []
    for budget in [1.1, 2.0, 3.0, 5.0]:
        thr, p, val_fa = find_thresholds(val_scores, budget)
        det = {part: np.array([(f[part] > thr).any(axis=1).mean() * 100 for f in fault_scores])
               for part in ["val", "test"]}
        rows_out.append({
            "val_budget_pct": budget,
            "percentile": round(p, 2),
            "false_alarm_val_pct": round(val_fa, 2),
            "false_alarm_test_pct": round((test_scores > thr).any(axis=1).mean() * 100, 2),
            "detection_all_val_pct": round(det["val"].mean(), 1),
            "detection_all_test_pct": round(det["test"].mean(), 1),
            "detection_21_test_pct": round(det["test"][visible].mean(), 1),
        })
    sweep = pd.DataFrame(rows_out)
    sweep.to_csv("data/hybrid_tradeoff.csv", index=False)
    print("\n=== False-alarm budget sweep ('21' = without the two fouling 095 files) ===")
    print(sweep.to_string(index=False))
    print("Saved data/hybrid_tradeoff.csv")
