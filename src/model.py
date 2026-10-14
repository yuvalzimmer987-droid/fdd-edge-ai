import torch.nn as nn

# The autoencoder architecture, in its own file so that training and
# evaluation scripts use exactly the same model.

class Autoencoder(nn.Module):
    # Sizes chosen with capacity_sweep.py --seeds (5 random seeds per model,
    # main model + pump detector, decided on validation days):
    #   (16, 6),    692 params: 82.5 +- 2.2% (2% budget, val) - depends on the seed;
    #                           seed 42 happened to be its best run (85.5%)
    #   (128, 8), 5,910 params: 85.6 +- 0.2% - the same result for every seed
    # More capacity does not raise the ceiling (~85%, limited by the data), but
    # reaches it reliably. 5,910 parameters (~23 KB) is still tiny for an edge device.
    def __init__(self, n_features, hidden=128, bottleneck=8):
        super().__init__()

        # Encoder: compress n_features -> hidden -> bottleneck
        # No ReLU on the bottleneck: a ReLU unit that gets stuck at 0 ("dead")
        # throws away part of the model's capacity.
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )

        # Decoder: reconstruct bottleneck -> hidden -> n_features
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
            # no activation on the last layer (linear output)
        )

    def forward(self, x):
        encoded = self.encoder(x)      # compress to the bottleneck
        decoded = self.decoder(encoded)  # reconstruct back to n_features
        return decoded
