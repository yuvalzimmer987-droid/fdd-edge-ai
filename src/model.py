import torch.nn as nn

# The autoencoder architecture, in its own file so that training and
# evaluation scripts use exactly the same model.

class Autoencoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # Encoder: compress n_features -> 8 -> 4 (the bottleneck)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )

        # Decoder: reconstruct 4 -> 8 -> n_features
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, n_features),
            # no activation on the last layer (linear output)
        )

    def forward(self, x):
        encoded = self.encoder(x)      # compress to the bottleneck
        decoded = self.decoder(encoded)  # reconstruct back to n_features
        return decoded
