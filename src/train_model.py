import numpy as np

# --- Step 1: load the processed data from Week 2 ---
print("Loading processed data...")

train = np.load("data/processed/train.npy")
val   = np.load("data/processed/val.npy")
test_normal = np.load("data/processed/test_normal.npy")

print(f"Train:       {train.shape}")
print(f"Validation:  {val.shape}")
print(f"Test-normal: {test_normal.shape}")

# The number of sensors = number of columns = the input size for the model
n_features = train.shape[1]
print(f"\nNumber of features (sensors): {n_features}")


# --- Step 2: build the autoencoder architecture ---
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # Encoder: compress 15 -> 8 -> 4 (the bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )

        # Decoder: reconstruct 4 -> 8 -> 15
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, n_features),
            # no activation on the last layer (linear output)
        )

    def forward(self, x):
        encoded = self.encoder(x)      # compress to the bottleneck
        decoded = self.decoder(encoded)  # reconstruct back to 15
        return decoded


# Create the model
model = Autoencoder(n_features)
print("\n--- Autoencoder architecture ---")
print(model)

# --- Step 3: prepare for training ---
from torch.utils.data import DataLoader, TensorDataset

# Convert the numpy data to PyTorch tensors (the format the model expects)
train_tensor = torch.tensor(train, dtype=torch.float32)
val_tensor   = torch.tensor(val, dtype=torch.float32)

# Wrap in a DataLoader that feeds batches of 256 rows
# For an autoencoder, input and target are the SAME (it reconstructs its input)
train_loader = DataLoader(
    TensorDataset(train_tensor, train_tensor),
    batch_size=256,
    shuffle=True,
)

# Loss function: MSE (measures reconstruction error)
loss_fn = nn.MSELoss()

# Optimizer: Adam, with learning rate 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining setup ready.")


# --- Step 4: training loop ---
import os

NUM_EPOCHS   = 100
PRINT_EVERY  = 10   # print loss every N epochs
PATIENCE     = 10   # early-stopping patience (epochs with no val improvement)

print(f"\n--- Training for up to {NUM_EPOCHS} epochs ---")

best_val_loss  = float("inf")
epochs_no_improve = 0
train_losses   = []
val_losses     = []

for epoch in range(1, NUM_EPOCHS + 1):

    # --- training phase ---
    model.train()
    batch_losses = []
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        recon = model(x_batch)
        loss  = loss_fn(recon, y_batch)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    train_loss = sum(batch_losses) / len(batch_losses)
    train_losses.append(train_loss)

    # --- validation phase ---
    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor)
        val_loss  = loss_fn(val_recon, val_tensor).item()
    val_losses.append(val_loss)

    if epoch % PRINT_EVERY == 0:
        print(f"  Epoch {epoch:3d} | train loss: {train_loss:.6f} | val loss: {val_loss:.6f}")

    # --- early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # save the best weights so far
        torch.save(model.state_dict(), "models/autoencoder_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
            break

print(f"\nBest validation loss: {best_val_loss:.6f}")
print("Best model weights saved to models/autoencoder_best.pt")

# reload the best weights before evaluation
model.load_state_dict(torch.load("models/autoencoder_best.pt"))
model.eval()


# --- Step 5: calibrate the anomaly detection threshold on the validation set ---
print("\n--- Setting anomaly threshold on validation set ---")

# Compute per-sample reconstruction error (MSE for each row)
with torch.no_grad():
    val_recon    = model(val_tensor)
    # shape: (N, n_features) -> mean over features -> (N,)
    val_errors   = ((val_tensor - val_recon) ** 2).mean(dim=1).numpy()

# Use the 95th percentile of normal validation errors as the threshold.
# Samples above this will be flagged as anomalies.
THRESHOLD_PERCENTILE = 95
threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))

print(f"Validation reconstruction errors:")
print(f"  min  : {val_errors.min():.6f}")
print(f"  mean : {val_errors.mean():.6f}")
print(f"  p95  : {threshold:.6f}  <- threshold")
print(f"  max  : {val_errors.max():.6f}")

# Save threshold alongside the model weights
np.save("models/threshold.npy", np.array([threshold]))
print(f"Threshold saved to models/threshold.npy")


# --- Step 6: evaluate on test_normal (should be mostly below threshold) ---
print("\n--- Evaluation on test_normal split ---")

test_tensor = torch.tensor(test_normal, dtype=torch.float32)

with torch.no_grad():
    test_recon  = model(test_tensor)
    test_errors = ((test_tensor - test_recon) ** 2).mean(dim=1).numpy()

flags_normal = (test_errors > threshold).sum()
print(f"Test-normal samples:  {len(test_errors)}")
print(f"Flagged as anomaly:   {flags_normal} ({flags_normal / len(test_errors) * 100:.1f}%)")
print(f"  (ideal: close to {100 - THRESHOLD_PERCENTILE}% false-alarm rate)")


# --- Step 7: evaluate on a sample fault file ---
import joblib
import pandas as pd

FAULT_FILE = "data/LBNL_FDD_Data_Sets_Chiller_Plant/ChillerPlant_bypass_leakage_075.csv"

SELECTED_SENSORS = [
    "CHL_SW_TEMP_1", "CHL_RW_TEMP_1", "CHL_SWCD_TEMP_1", "CHL_RWCD_TEMP_1",
    "CWL_SEC_SW_TEMP", "CWL_SEC_RW_TEMP", "CT_SW_TEMP_1", "CT_RW_TEMP_1",
    "OA_TEMP", "OA_TEMP_WB", "CHL_POW_1", "CT_POW_1",
    "CHL_CW_FLOW_1", "CWL_SEC_DP", "CHL_STA_1",
]

def add_delta_features(df):
    df = df.copy()
    df["CT_delta"]  = df["CT_RW_TEMP_1"]   - df["CT_SW_TEMP_1"]
    df["CHL_delta"] = df["CHL_RW_TEMP_1"]  - df["CHL_SW_TEMP_1"]
    df["CD_delta"]  = df["CHL_RWCD_TEMP_1"]- df["CHL_SWCD_TEMP_1"]
    df["CWL_delta"] = df["CWL_SEC_RW_TEMP"]- df["CWL_SEC_SW_TEMP"]
    return df

if os.path.exists(FAULT_FILE):
    scaler = joblib.load("models/scaler.joblib")
    fault_df = add_delta_features(pd.read_csv(FAULT_FILE, usecols=SELECTED_SENSORS))
    fault_scaled = scaler.transform(fault_df)
    fault_tensor = torch.tensor(fault_scaled, dtype=torch.float32)

    with torch.no_grad():
        fault_recon  = model(fault_tensor)
        fault_errors = ((fault_tensor - fault_recon) ** 2).mean(dim=1).numpy()

    flags_fault = (fault_errors > threshold).sum()
    print(f"\n--- Evaluation on fault file: {os.path.basename(FAULT_FILE)} ---")
    print(f"Fault samples:        {len(fault_errors)}")
    print(f"Detected as anomaly:  {flags_fault} ({flags_fault / len(fault_errors) * 100:.1f}%)")
    print(f"Mean fault error:     {fault_errors.mean():.6f} (vs threshold {threshold:.6f})")
else:
    print(f"\nFault file not found, skipping fault evaluation: {FAULT_FILE}")

print("\nWeek 4 training and evaluation complete!")