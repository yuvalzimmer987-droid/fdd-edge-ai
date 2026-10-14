import json
import numpy as np
import pandas as pd
import joblib
import torch
from model import Autoencoder                 # defined in src/model.py
from preprocessing import steady_state_mask

# The final fault detector in one place, used by export_model.py and the
# dashboard: the main model (chiller + cooling tower, 14 sensors) plus the
# separate pump detector, each with its own hourly max-z score and threshold.
#
# For each hour it answers:
#   - is there a fault? (any detector above its threshold)
#   - where? (the detector furthest above its threshold)
#   - which sensor? (the most-shifted sensor of that detector)
#
# The models and thresholds come from train_model.py and
# subsystems.py --hybrid. The models are plain functions (scaled input ->
# reconstruction), so the same detector runs with PyTorch or with ONNX.
#
# Use:
#   from detector import Detector
#   detector = Detector.from_saved()
#   hours = detector.hourly_file(df)     # df = a plant CSV, see load_columns()

WINDOW_MINUTES = 60
MIN_ROWS_PER_WINDOW = 30
THRESHOLDS_FILE = "models/subsys_thresholds.json"
MODEL_FILES = {               # name -> (PyTorch weights, scaler)
    "main":  ("models/autoencoder.pt", "models/scaler.joblib"),
    "pumps": ("models/subsys_pumps.pt", "models/subsys_pumps_scaler.joblib"),
}
LABELS = {"main": "chiller / cooling tower", "pumps": "pumping loop"}


def build_model(name, n_features):
    """The architecture of each detector (must match how it was trained)."""
    if name == "main":
        return Autoencoder(n_features)                       # sizes from model.py
    return Autoencoder(n_features, hidden=16, bottleneck=max(2, n_features // 2))   # as subsystems.py


def load_torch_models():
    """The trained PyTorch models and their scalers."""
    models, scalers = {}, {}
    for name, (weights, scaler_file) in MODEL_FILES.items():
        scalers[name] = joblib.load(scaler_file)
        models[name] = build_model(name, len(scalers[name].feature_names_in_))
        models[name].load_state_dict(torch.load(weights))
        models[name].eval()
    return models, scalers


def torch_predictor(model):
    def predict(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()
    return predict


def window_means(idx, resid, columns):
    """Mean signed residual per sensor for each 1-hour window (indexed by hour of the year)."""
    df = pd.DataFrame(resid, columns=columns)
    window = idx // WINDOW_MINUTES
    counts = df.groupby(window).size()
    return df.groupby(window).mean()[counts >= MIN_ROWS_PER_WINDOW]


def find_thresholds(val_scores, budget):
    """Lowest common percentile whose thresholds flag <= budget % of normal
    validation hours, all detectors together (same as subsystems.py)."""
    for pct in np.append(np.arange(95.0, 100.0, 0.01), 100.0):
        thresholds = val_scores.quantile(min(pct / 100, 1.0))
        if (val_scores > thresholds).any(axis=1).mean() * 100 <= budget:
            return thresholds


class Detector:
    def __init__(self, predictors, scalers):
        self.predictors = predictors          # name -> function(scaled x) -> reconstruction
        self.scalers = scalers
        self.columns = {name: list(s.feature_names_in_) for name, s in scalers.items()}
        self.center, self.spread, self.thresholds = {}, {}, None

    @classmethod
    def from_saved(cls):
        """PyTorch models with the thresholds saved by subsystems.py --hybrid."""
        models, scalers = load_torch_models()
        detector = cls({name: torch_predictor(m) for name, m in models.items()}, scalers)
        with open(THRESHOLDS_FILE) as f:
            saved = json.load(f)
        for name in models:
            if saved[name]["columns"] != detector.columns[name]:
                raise ValueError(f"{THRESHOLDS_FILE} does not match the {name} model: "
                                 "run python src/subsystems.py --hybrid again")
            detector.center[name] = np.array(saved[name]["center"])
            detector.spread[name] = np.array(saved[name]["spread"])
        detector.thresholds = pd.Series({name: saved[name]["threshold"] for name in models})
        detector.budget = saved.get("budget", 1.1)
        return detector

    def load_columns(self):
        """Columns to read from a plant CSV (CHL_POW_1 for the steady-state mask)."""
        return sorted({c for cols in self.columns.values() for c in cols} | {"CHL_POW_1"})

    def _z(self, frame, idx):
        """Per detector: |z| of every sensor's hourly mean residual."""
        z = {}
        for name, predict in self.predictors.items():
            x = self.scalers[name].transform(frame[self.columns[name]]).astype(np.float32)
            win = window_means(idx, predict(x) - x, self.columns[name])
            z[name] = ((win - self.center[name]) / self.spread[name]).abs()
        return z

    def calibrate(self, frame, idx, budget):
        """Fit the z-scores and thresholds on normal validation rows (steady state)."""
        for name, predict in self.predictors.items():
            x = self.scalers[name].transform(frame[self.columns[name]]).astype(np.float32)
            win = window_means(idx, predict(x) - x, self.columns[name])
            self.center[name], self.spread[name] = win.mean().to_numpy(), win.std().to_numpy()
        scores = pd.DataFrame({name: z.max(axis=1) for name, z in self._z(frame, idx).items()})
        self.thresholds = find_thresholds(scores, budget)
        self.budget = budget

    def hourly(self, frame, idx):
        """One row per hour: score and alarm of each detector, and the overall verdict.

        frame = steady-state rows of a plant file, idx = their row numbers in
        the file (used to group them into hours).
        """
        z = self._z(frame, idx)
        out = pd.DataFrame(index=next(iter(z.values())).index)
        out.index.name = "hour"
        for name, zn in z.items():
            out[f"score_{name}"] = zn.max(axis=1)
            out[f"alarm_{name}"] = out[f"score_{name}"] > self.thresholds[name]
            out[f"sensor_{name}"] = zn.idxmax(axis=1)
        names = list(z)
        out["alarm"] = out[[f"alarm_{n}" for n in names]].any(axis=1)

        # Where: the detector furthest above its own threshold
        ratio = pd.DataFrame({n: out[f"score_{n}"] / self.thresholds[n] for n in names})
        worst = ratio.idxmax(axis=1)
        out["subsystem"] = np.where(out["alarm"], worst.map(LABELS), "")
        sensor = [out.at[h, f"sensor_{w}"] for h, w in worst.items()]
        out["suspect_sensor"] = np.where(out["alarm"], sensor, "")
        return out

    def hourly_file(self, df, rows=None):
        """hourly() for a whole plant file, after the steady-state filter.
        rows: optional boolean mask to keep only some rows (e.g. test days)."""
        keep, _, _, _ = steady_state_mask(df)
        if rows is not None:
            keep = keep & rows
        idx = np.flatnonzero(keep)
        return self.hourly(df.iloc[idx], idx)
