import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# --- load model ---
class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 8), nn.ReLU(),
            nn.Linear(8, 4),          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),           nn.ReLU(),
            nn.Linear(8, n_features),
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

SELECTED_SENSORS = [
    "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1",
    "OA_TEMP", "OA_TEMP_WB", "CHL_POW_1", "CT_POW_1",
    "CHL_CW_FLOW_1", "CWL_SEC_DP", "CHL_STA_1",
]

THRESHOLD_PERCENTILE = 95


def add_delta_features(df):
    df = df.copy()
    df["CT_delta"]  = df["CT_RW_TEMP_1"]   - df["CT_SW_TEMP_1"]
    df["CHL_delta"] = df["CHL_RW_TEMP_1"]  - df["CHL_SW_TEMP_1"]
    df["CD_delta"]  = df["CHL_RWCD_TEMP_1"]- df["CHL_SWCD_TEMP_1"]
    df["CWL_delta"] = df["CWL_SEC_RW_TEMP"]- df["CWL_SEC_SW_TEMP"]
    return df


n_features = len(SELECTED_SENSORS) + 4  # 15 מקוריים + 4 deltas
model = Autoencoder(n_features)
model.load_state_dict(torch.load("models/autoencoder_best.pt"))
model.eval()

scaler = joblib.load("models/scaler.joblib")


def get_per_sensor_errors(tensor):
    """מחזיר שגיאה נפרדת לכל חיישן לכל שורה — shape: (N, 15)"""
    with torch.no_grad():
        recon = model(tensor)
        errors = ((tensor - recon) ** 2).numpy()  # לא עושים mean על הצירים
    return errors


# --- כיול סף נפרד לכל חיישן על validation ---
print("Calibrating per-sensor thresholds on validation set...")

val = np.load("data/processed/val.npy")
val_tensor = torch.tensor(val, dtype=torch.float32)
val_errors = get_per_sensor_errors(val_tensor)  # (N, 19)

# לכל חיישן — סף בנפרד על האחוזון ה-95
thresholds = np.percentile(val_errors, THRESHOLD_PERCENTILE, axis=0)  # (19,)

ALL_FEATURES = SELECTED_SENSORS + ["CT_delta", "CHL_delta", "CD_delta", "CWL_delta"]

print(f"\nPer-sensor thresholds (p{THRESHOLD_PERCENTILE}):")
for name, thr in zip(ALL_FEATURES, thresholds):
    print(f"  {name:25s}: {thr:.6f}")

# --- הערכה על test_normal ---
test_normal = np.load("data/processed/test_normal.npy")
test_tensor = torch.tensor(test_normal, dtype=torch.float32)
test_errors = get_per_sensor_errors(test_tensor)  # (N, 15)

# שורה = תקלה אם לפחות חיישן אחד חרג מהסף שלו
flagged = (test_errors > thresholds).any(axis=1)
false_alarms = flagged.sum()
print(f"\nTest-normal: {len(flagged)} samples | false alarm: {false_alarms} ({false_alarms/len(flagged)*100:.1f}%)\n")

# --- הערכה על כל קבצי התקלה ---
catalog = pd.read_csv("data/file_catalog.csv")
fault_files = catalog[catalog["label"] == "fault"]
DATA_DIR = "data/LBNL_FDD_Data_Sets_Chiller_Plant"

results = []

for _, row in fault_files.iterrows():
    path = os.path.join(DATA_DIR, row["filename"])
    if not os.path.exists(path):
        continue

    df     = add_delta_features(pd.read_csv(path, usecols=SELECTED_SENSORS))
    scaled = scaler.transform(df)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    errors   = get_per_sensor_errors(tensor)        # (N, 15)
    flagged  = (errors > thresholds).any(axis=1)    # True אם אחד מ-15 חרג
    detected = flagged.sum()
    detection_rate = detected / len(flagged) * 100

    results.append({
        "fault_type":  row["fault_type"],
        "severity":    row["severity"],
        "detected":    int(detected),
        "detection_%": round(detection_rate, 1),
    })

    print(f"{row['fault_type']:45s} | severity={str(row['severity']):5s} | detected={detection_rate:5.1f}%")

# --- טבלת סיכום ---
results_df = pd.DataFrame(results).sort_values("detection_%", ascending=False)
print("\n--- Summary (sorted by detection rate) ---")
print(results_df.to_string(index=False))
