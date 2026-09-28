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
from model import Autoencoder   # defined in src/model.py

# Create the model (seed first, so the starting weights are the same every run)
torch.manual_seed(42)
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

# --- Step 4: training loop with early stopping ---
from codecarbon import EmissionsTracker
import copy

MAX_EPOCHS = 200
PATIENCE   = 10      # stop if val loss does not improve for this many epochs
MIN_DELTA  = 1e-5    # smaller improvements than this do not count



def evaluate(model, data):
    """Mean reconstruction loss on a whole dataset (no training)."""
    model.eval()
    with torch.no_grad():
        return loss_fn(model(data), data).item()


print("\n--- Training ---")
tracker = EmissionsTracker(project_name="fdd_training", log_level="error")
tracker.start()

best_val_loss = float("inf")
best_state = None
best_epoch = 0
epochs_without_improvement = 0
history = []   # (train_loss, val_loss) per epoch

for epoch in range(1, MAX_EPOCHS + 1):
    # Train on all batches
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()                     # clear old gradients
        loss = loss_fn(model(x_batch), y_batch)   # how bad is the reconstruction
        loss.backward()                           # compute gradients
        optimizer.step()                          # update the weights
        total_loss += loss.item() * len(x_batch)
    train_loss = total_loss / len(train_tensor)

    # Check on validation data (never used for training)
    val_loss = evaluate(model, val_tensor)
    history.append((train_loss, val_loss))

    # Early stopping: keep the best model, stop when it stops improving
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        epochs_without_improvement = 0
        mark = "  <- best"
    else:
        epochs_without_improvement += 1
        mark = ""

    print(f"Epoch {epoch:3d} | train loss {train_loss:.5f} | val loss {val_loss:.5f}{mark}")

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping: no improvement for {PATIENCE} epochs.")
        break

tracker.stop()
emissions = tracker.final_emissions_data

print(f"\nBest epoch: {best_epoch} (val loss {best_val_loss:.5f})")
print(f"Training time:   {emissions.duration:.1f} seconds")
print(f"Energy consumed: {emissions.energy_consumed:.6f} kWh")

# --- Step 5: save the best model ---
model.load_state_dict(best_state)
torch.save(model.state_dict(), "models/autoencoder.pt")
np.save("data/processed/loss_history.npy", np.array(history))
print("\nBest model saved to models/autoencoder.pt")
print("Loss history saved to data/processed/loss_history.npy")

# --- Step 6: quick check of the reconstruction error ---
# Per-row error = mean squared error over the 14 sensors of that row.
# Next step: use the validation errors to choose the fault threshold.
test_tensor = torch.tensor(test_normal, dtype=torch.float32)

model.eval()
with torch.no_grad():
    for name, data in [("Validation", val_tensor), ("Test-normal", test_tensor)]:
        errors = ((model(data) - data) ** 2).mean(dim=1).numpy()
        print(f"{name:12s} error: mean={errors.mean():.5f}  "
              f"median={np.median(errors):.5f}  99th pct={np.percentile(errors, 99):.5f}")

    # A bottleneck unit that barely changes between rows is not being used
    codes = model.encoder(torch.tensor(train, dtype=torch.float32)).numpy()
    unit_std = codes.std(axis=0)
    print(f"\nBottleneck units in use: {(unit_std > 1e-3).sum()} of {len(unit_std)} "
          f"(std per unit: {np.round(unit_std, 3)})")

print("\nWeek 3 training complete!")