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