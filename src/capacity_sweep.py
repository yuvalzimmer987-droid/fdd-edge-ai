import sys
import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from preprocessing import DATA_DIR, load_plant_file, day_split, steady_state_mask

# Does a bigger main model detect more faults?
#
# experiment.py already compared sizes up to 1364 parameters (16,6 -> 32,6:
# +0.7 points on validation days). This script goes much further: wider
# hidden layers, a wider bottleneck and two hidden layers per side, up to
# ~30x the current 692 parameters. Each main model is evaluated in the full
# detector it would be used in: main model + the separate pump detector,
# thresholds set together for a 1.1% and a 2% false-alarm budget on normal
# validation hours (same as subsystems.py --hybrid --sweep).
#
# Fair selection, as always: thresholds from normal validation days, the
# decision on the VALIDATION days of the fault files, TEST days only reported.
#
# Run: python src/capacity_sweep.py           (full run, ~20-40 min)
#      python src/capacity_sweep.py --quick   (3 epochs per model, just to test the code)

# (hidden layer widths per side, bottleneck). ((16,), 6) is the current model.
CONFIGS = [
    ((16,), 6), ((32,), 6), ((64,), 6), ((128,), 6),
    ((64,), 8), ((128,), 8),
    ((64, 32), 6), ((128, 64), 8),
]
BUDGETS = [1.1, 2.0]           # false-alarm budgets, % of normal validation hours
PERCENTILE = 99                # main model alone, as in evaluate_faults.py
WINDOW_MINUTES = 60
MIN_ROWS_PER_WINDOW = 30
MAX_EPOCHS = 3 if "--quick" in sys.argv else 200
PATIENCE = 10

PUMP_COLUMNS = [               # same as subsystems.py
    "CWL_SEC_DP", "CWL_SEC_DPSPT", "CWL_SEC_PM_SPD_1", "CWL_SEC_PM_SPD_2",
    "CWL_SEC_PM_POW_1", "CWL_SEC_PM_POW_2", "CWL_SEC_CW_FLOW", "CWL_PRI_CW_FLOW",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CWL_SEC_LOAD",
]
INVISIBLE = ["ChillerPlant_chiller_fouling_095.csv", "ChillerPlant_coolingtower_fouling_095.csv"]


class Autoencoder(nn.Module):
    """Same design as src/model.py, with any number of hidden layers per side."""
    def __init__(self, n_features, hidden, bottleneck):
        super().__init__()
        enc, size = [], n_features
        for h in hidden:
            enc += [nn.Linear(size, h), nn.ReLU()]
            size = h
        enc.append(nn.Linear(size, bottleneck))       # linear bottleneck
        dec, size = [], bottleneck
        for h in reversed(hidden):
            dec += [nn.Linear(size, h), nn.ReLU()]
            size = h
        dec.append(nn.Linear(size, n_features))       # linear output
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(train, val, hidden, bottleneck):
    """Same training as train_model.py: Adam, early stopping on val loss."""
    torch.manual_seed(42)
    model = Autoencoder(train.shape[1], hidden, bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    train_t, val_t = torch.tensor(train), torch.tensor(val)
    loader = DataLoader(TensorDataset(train_t, train_t), batch_size=256, shuffle=True)

    best_loss, best_state, waited, epochs = float("inf"), None, 0, 0
    for epochs in range(1, MAX_EPOCHS + 1):
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
    return model, best_loss, epochs


def window_means(idx, resid):
    """Mean signed residual per column for each 1-hour window (indexed by window id)."""
    df = pd.DataFrame(resid)
    window = idx // WINDOW_MINUTES
    counts = df.groupby(window).size()
    return df.groupby(window).mean()[counts >= MIN_ROWS_PER_WINDOW]


class HourlyScorer:
    """Hourly max-z score of one autoencoder, z-scores fitted on normal validation hours."""
    def __init__(self, model, val_idx, val_x):
        self.model = model
        win = window_means(val_idx, self.resid(val_x))
        self.center, self.spread = win.mean().to_numpy(), win.std().to_numpy()

    def resid(self, x):
        with torch.no_grad():
            t = torch.tensor(x)
            return (self.model(t) - t).numpy()

    def __call__(self, idx, x):
        win = window_means(idx, self.resid(x))
        z = np.abs((win.to_numpy() - self.center) / self.spread).max(axis=1)
        return pd.Series(z, index=win.index)


def find_thresholds(val_scores, budget):
    """Lowest common percentile whose thresholds flag <= budget % of normal val hours
    (same as subsystems.py)."""
    for pct in np.append(np.arange(95.0, 100.0, 0.01), 100.0):
        thresholds = val_scores.quantile(min(pct / 100, 1.0))
        rate = (val_scores > thresholds).any(axis=1).mean() * 100
        if rate <= budget:
            return thresholds, rate


# --- Step 1: load everything once ---
print("Loading data...")
main_scaler = joblib.load("models/scaler.joblib")
main_columns = list(main_scaler.feature_names_in_)
all_columns = sorted(set(main_columns) | set(PUMP_COLUMNS) | {"CHL_POW_1"})

normal = load_plant_file(f"{DATA_DIR}/ChillerPlant.csv", all_columns)
keep, _, _, _ = steady_state_mask(normal)
split = day_split(len(normal))
rows = {p: np.flatnonzero(keep & (split == p)) for p in ["train", "val", "test"]}

pump_frame = normal[PUMP_COLUMNS]
pump_columns = list(pump_frame.columns[pump_frame.iloc[rows["train"]].std() > 0])
pump_scaler = StandardScaler().fit(normal[pump_columns].iloc[rows["train"]])


def inputs(df, idx):
    """Scaled inputs of the main model and of the pump detector for the given rows."""
    part = df.iloc[idx]
    return (main_scaler.transform(part[main_columns]).astype(np.float32),
            pump_scaler.transform(part[pump_columns]).astype(np.float32))


normal_x = {p: inputs(normal, r) for p, r in rows.items()}

catalog = pd.read_csv("data/file_catalog.csv", dtype={"severity": str})
faults = []
for _, row in catalog[catalog["label"] == "fault"].iterrows():
    df = load_plant_file(f"{DATA_DIR}/{row['filename']}", all_columns)
    f_keep, _, _, _ = steady_state_mask(df)
    f_split = day_split(len(df))
    fault = {"file": row["filename"]}
    for part in ["val", "test"]:
        idx = np.flatnonzero(f_keep & (f_split == part))
        fault[part] = (idx, *inputs(df, idx))
    faults.append(fault)
visible = np.array([f["file"] not in INVISIBLE for f in faults])
print(f"Loaded {len(faults)} fault files.")

# --- Step 2: the pump detector, trained once (same as subsystems.py --hybrid) ---
n_pump = len(pump_columns)
pump_model, _, _ = train_autoencoder(normal_x["train"][1], normal_x["val"][1],
                                     (16,), max(2, n_pump // 2))
pump = HourlyScorer(pump_model, rows["val"], normal_x["val"][1])
pump_scores = {p: pump(rows[p], normal_x[p][1]) for p in ["val", "test"]}
pump_fault_scores = [{p: pump(f[p][0], f[p][2]) for p in ["val", "test"]} for f in faults]
print(f"Pump detector: {n_pump} inputs.")

# --- Step 3: every main model size ---
results = []
for hidden, bottleneck in CONFIGS:
    start = time.time()
    model, val_loss, epochs = train_autoencoder(normal_x["train"][0], normal_x["val"][0],
                                                hidden, bottleneck)
    n_params = sum(p.numel() for p in model.parameters())
    main = HourlyScorer(model, rows["val"], normal_x["val"][0])

    def both(main_s, pump_s):
        return pd.DataFrame({"main": main_s, "pumps": pump_s})

    val_scores = both(main(rows["val"], normal_x["val"][0]), pump_scores["val"])
    test_scores = both(main(rows["test"], normal_x["test"][0]), pump_scores["test"])
    fault_scores = [{p: both(main(f[p][0], f[p][1]), pf[p]) for p in ["val", "test"]}
                    for f, pf in zip(faults, pump_fault_scores)]

    # Main model alone at the 99th percentile (comparable to the 72.5% val / 72.8% test)
    thr = np.percentile(val_scores["main"].dropna(), PERCENTILE)
    alone = {p: np.mean([(s[p]["main"].dropna() > thr).mean() * 100 for s in fault_scores])
             for p in ["val", "test"]}

    row = {"hidden": "-".join(map(str, hidden)), "bottleneck": bottleneck, "params": n_params,
           "epochs": epochs, "val_loss": round(val_loss, 5),
           "train_s": round(time.time() - start),
           "main_alone_val_pct": round(alone["val"], 1), "main_alone_test_pct": round(alone["test"], 1)}
    for budget in BUDGETS:
        thresholds, _ = find_thresholds(val_scores, budget)
        det = {p: np.array([(s[p] > thresholds).any(axis=1).mean() * 100 for s in fault_scores])
               for p in ["val", "test"]}
        tag = f"{budget:g}"
        row[f"hybrid{tag}_val_pct"] = round(det["val"].mean(), 1)
        row[f"hybrid{tag}_test_pct"] = round(det["test"].mean(), 1)
        row[f"hybrid{tag}_test21_pct"] = round(det["test"][visible].mean(), 1)
        row[f"hybrid{tag}_fa_test_pct"] = round((test_scores > thresholds).any(axis=1).mean() * 100, 2)
    results.append(row)
    print(f"\nhidden {row['hidden']:>6s}  bottleneck {bottleneck}  {n_params:6d} params  "
          f"val loss {val_loss:.5f}  ({epochs} epochs, {row['train_s']} s)")
    print(f"   main alone          val {alone['val']:5.1f}%   test {alone['test']:5.1f}%")
    for budget in BUDGETS:
        tag = f"{budget:g}"
        print(f"   + pumps, {tag:>3s}% budget  val {row[f'hybrid{tag}_val_pct']:5.1f}%   "
              f"test {row[f'hybrid{tag}_test_pct']:5.1f}%   "
              f"false alarms (test) {row[f'hybrid{tag}_fa_test_pct']:.2f}%")

res = pd.DataFrame(results)
res.to_csv("data/capacity_results.csv", index=False)

print("\n\n=== Summary (decide on the 'val' columns; current model = first row) ===")
cols = ["hidden", "bottleneck", "params", "val_loss", "main_alone_val_pct",
        "hybrid1.1_val_pct", "hybrid1.1_test_pct", "hybrid2_val_pct", "hybrid2_test_pct"]
print(res[cols].to_string(index=False))
best = res.loc[res["hybrid2_val_pct"].idxmax()]
print(f"\nBest on validation days (2% budget): hidden {best['hidden']}, bottleneck "
      f"{best['bottleneck']}, {best['params']} params -> {best['hybrid2_val_pct']}% val, "
      f"{best['hybrid2_test_pct']}% test")
print("Saved data/capacity_results.csv")
