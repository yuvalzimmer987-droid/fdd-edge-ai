import torch.nn as nn

# The autoencoder architecture, in its own file so that training and
# evaluation scripts use exactly the same model.

class Autoencoder(nn.Module):
    # Sizes chosen with experiment.py: (16, 6) detected as many faults as the
    # best setting on the validation days (72.5% vs 73.2% of 1-hour windows)
    # with half the parameters (692 vs 1364), and works with the per-sensor
    # max-z score, which also names the faulty sensor.
    def __init__(self, n_features, hidden=16, bottleneck=6):
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
