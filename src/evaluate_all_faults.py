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

n_features = len(SELECTED_SENSORS)
model = Autoencoder(n_features)
model.load_state_dict(torch.load("models/autoencoder_best.pt"))
model.eval()

scaler    = joblib.load("models/scaler.joblib")
threshold = float(np.load("models/threshold.npy")[0])

print(f"Model loaded. Threshold = {threshold:.6f}\n")

# --- load catalog ---
catalog = pd.read_csv("data/file_catalog.csv")
fault_files = catalog[catalog["label"] == "fault"]

DATA_DIR = "data/LBNL_FDD_Data_Sets_Chiller_Plant"

results = []

for _, row in fault_files.iterrows():
    path = os.path.join(DATA_DIR, row["filename"])
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path, usecols=SELECTED_SENSORS)
    scaled = scaler.transform(df)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    with torch.no_grad():
        recon  = model(tensor)
        errors = ((tensor - recon) ** 2).mean(dim=1).numpy()

    detected = (errors > threshold).sum()
    detection_rate = detected / len(errors) * 100

    results.append({
        "fault_type": row["fault_type"],
        "severity":   row["severity"],
        "samples":    len(errors),
        "detected":   int(detected),
        "detection_%": round(detection_rate, 1),
        "mean_error":  round(float(errors.mean()), 4),
    })

    print(f"{row['fault_type']:45s} | severity={str(row['severity']):5s} | detected={detection_rate:5.1f}%")

# --- summary table ---
results_df = pd.DataFrame(results).sort_values("detection_%", ascending=False)
print("\n--- Summary (sorted by detection rate) ---")
print(results_df.to_string(index=False))
